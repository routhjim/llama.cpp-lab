# Re-fitting the Flash-Next MTP head to a requantized target

Lab PR #50 left one caveat open: after the dense-path requant (C1..C23T8), draft acceptance fell
2-3 points, and re-fitting the MTP head to the requantized target "was not tried". This is that
experiment. It worked, and the re-fit head is now the production drafter on this box.

Weights: https://huggingface.co/Jrouth/Qwen3.8-Flash-Next-MTP-C2T8-refit-GGUF

## Result

Live A/B, production shape (C2T8, 84/16 iGPU/XTX split, MTP depth 3 with the drafter on the XTX,
coupled sampling, np1), the 12 real Terminal-Bench 2.1 requests `fn-replay.py` uses in #50, same seeds.
Same binary and launcher; only the `-md` file differs.

| | original Q4DRAFT head | re-fit head |
|---|---|---|
| draft acceptance (server counters) | 0.651 (9241 / 14190) | **0.692** (8215 / 11871) |
| decode, time-weighted | 38.7 tok/s | 40.0 tok/s (+3.4%) |
| decode, median request | 40.7 tok/s | 41.6 tok/s (+2.2%) |
| requests with higher acceptance | | 9 of 12 |

The acceptance gain is about ten times its standard error. The decode gain is inside the ~3%
run-to-run noise on this box, but in the size the acceptance gain predicts (each verify step returns
accepted + 1 tokens: 2.98 -> 3.12). Output is unchanged by construction: the target verifies every
draft.

Offline, on 45k held-out assistant positions (expected acceptance per draft position):

| head | pos 1 | pos 2 | pos 3 | accepted per 3-token round |
|---|---|---|---|---|
| original, q8_0 experts | 0.841 | 0.783 | 0.744 | 1.988 |
| original, as served (Q4DRAFT) | 0.839 | 0.781 | 0.741 | 1.979 |
| re-fit, as served | 0.871 | 0.810 | 0.771 | 2.119 |

The re-fit head accepts 0.707 per drafted token offline, above the 0.687 the original head reached
against UD-Q4_K_XL in #50. So it does not just recover the requant loss.

## Why a re-fit helps

The MTP head was trained against the bf16 model. A requant moves the target's distribution a little,
and the head keeps predicting the old one. At draft position 1 the effect is small; at positions 2
and 3 the head runs on its own output streams, and the mismatch compounds. That is where #50 saw the
loss.

## Method

1. **Corpus** (`scripts/mtp-refit/build_corpus.py`). The model's own agent traffic, rendered through
   the server's chat template (`/apply-template`): 115 terminus trajectories (JSON replies rebuilt as
   the model wrote them) and Claude Code sessions served by Flash-Next (native tool calls). 100
   training documents, 11 held out. About 85% of the text is terminal-agent traffic.
2. **Dump** (`tools/mtp-dump`). For every position, the wide hyper-connection residual the MTP head
   seeds from (`l_last-47`, 4 x 2560) and the target's top-20 log-probs. It uses only the public
   `cb_eval` hook, like `moe-trace`; no model changes. Documents are fed in 1024-token steps with the
   KV cache kept, capped at 16k tokens. 20.6 KB per position; about 13k positions a minute on the iGPU.
3. **Torch port of the head** (`scripts/mtp-refit/mtp_torch.py`), line for line from `graph_mtp` in
   `src/models/qwen4exp.cpp`, with weights loaded from the draft gguf so the converter's transforms
   are already applied. Chained draft steps attend to the true-seed entries up to t plus their own
   earlier draft entries, as the draft KV cache does.
   **Validation:** offline, the original head gives 0.663 accepted per drafted token on held-out
   assistant spans; the live server measured 0.662 in #50. The port, the chaining and the scoring
   agree with production to three decimals, so offline gains are trustworthy.
4. **Training** (`scripts/mtp-refit/train.py`). Soft cross-entropy against the target's renormalized
   top-20 at draft depths 1..3, counted only where the whole chain is generated text (assistant
   turns). Trainable: attention, `nextn.eh_proj`, hyper-connection mixers, norms, router and shared
   expert, 89M parameters with fp32 master weights. Frozen: the 512 experts, `token_embd` and `output`.
   AdamW, lr 2e-5, 20-step warmup, cosine decay, 2 epochs over 338 windows of 1024 positions
   (~400k positions): 20 minutes on the Strix Halo iGPU, 16 GB peak.
5. **Export** (`scripts/mtp-refit/export_gguf.py`). Copies the served drafter's metadata and tensors,
   replacing the 25 trained tensors re-quantized to their original types. An identity export is
   byte-identical to the source; re-running the export on the checkpoint reproduces the deployed file
   (same SHA-256).

## Reproduce

    # build llama-mtp-dump (tools/mtp-dump) with the rest of the tree, then:
    MTP_TB_GLOB='runs/**/agent/trajectory.json' python3 scripts/mtp-refit/build_corpus.py corpus.txt http://127.0.0.1:8081
    MODEL=<target gguf> DEV=Vulkan1 scripts/mtp-refit/dump.sh corpus.txt train.bin          # target must not be served meanwhile
    MODEL=<target gguf> DEV=Vulkan1 scripts/mtp-refit/dump.sh corpus.txt.heldout heldout.bin
    export MTP_Q8_GGUF=mtp-Qwen3.8-Flash-Next-Q8_0.gguf HIP_VISIBLE_DEVICES=<iGPU>
    python3 scripts/mtp-refit/eval_chain.py heldout.bin                                        # baseline
    python3 scripts/mtp-refit/train.py train.bin refit.pt --lr 2e-5 --epochs 2
    python3 scripts/mtp-refit/eval_chain.py heldout.bin "" 3 refit.pt
    MTP_BASE_GGUF=mtp-Qwen3.8-Flash-Next-Q4DRAFT.gguf python3 scripts/mtp-refit/export_gguf.py refit.pt refit.gguf

torch on gfx1151 needs a ROCm build with gfx1151 kernels (TheRock); it has none for the 7900 XTX.

## Caveats and next steps

- One live A/B of 12 requests. The decode gain is inside the noise band; acceptance is the solid number.
- The corpus is mostly terminal-agent traffic plus some benchmark sessions; other workloads are unmeasured.
- Only half the prepared corpus was dumped and used. More data, more epochs and an lr sweep are untried.
- The head was fitted to C2T8. It will need a re-fit (about an hour of dump plus 20 minutes of
  training) after any future requant of the target.
