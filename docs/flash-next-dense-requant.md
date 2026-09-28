# Requantizing Flash-Next's dense path: what the MTP drafter does to the gain

Findings from 2026-09-27/28 on Qwen3.8-Flash-Next (qwen4exp, 180B sparse MoE, 512 experts, top-10)
served by llama-server from this repo on the Radeon 8060S iGPU plus a Radeon RX 7900 XTX over USB4
(Vulkan/RADV). No code changes: this is a quantization recipe and a set of measurements. The recipe
is in [`scripts/flash-next-dense-requant.py`](../scripts/flash-next-dense-requant.py).

**Short version.** Requantizing the 8-bit dense tensors of unsloth's UD-Q4_K_XL to 4/5-bit buys
+18.5% decode with no drafter, but only +2 to +4% in the production configuration with the MTP
drafter. Quality can be kept equal to UD (perplexity ratio 0.997), but the drafter's acceptance
drops 2-3 points on every variant tried, and only at draft positions 2 and 3. We run the best
variant (C2T8) as the default, knowing the gain is at the edge of run-to-run noise.

## Why the dense path

At np1, 57% of a Flash-Next decode step is the 8-bit dense path and 21% is the routed experts
(see the README findings). UD-Q4_K_XL keeps `attn_qkv`, `attn_q`, `attn_gate` and `ssm_out` at q8_0
in all 48 blocks. Per token that is 4,606 MiB of dense weights against 1,435 MiB of routed experts
(10 of 512), so the dense path is where the bytes are. An earlier naive q4_K requant of it gave
+23% decode but +6.4% perplexity; these variants use unsloth's importance matrix and keep the
sensitive tensors at 8-bit.

## The variants

All are built from UD-Q4_K_XL with `llama-quantize --allow-requantize --imatrix <unsloth imatrix>
COPY` plus per-tensor overrides. Everything not listed stays as in UD. "Big dense" = `attn_qkv`,
`attn_q`, `attn_gate`, `ssm_out`.

| Name | Big dense tensors | `ssm_out` | Blocks 40-47 | `output` |
|---|---|---|---|---|
| C1 | q4_K | q4_K | as the rest | q6_K |
| C2 | q4_K | **q8_0** | as the rest | q6_K |
| T8 | q4_K | q4_K | **all big dense q8_0** | q6_K |
| C3 | **q5_K** | q5_K | as the rest | q6_K |
| C2T8 | q4_K | **q8_0** | **all big dense q8_0** | q6_K |
| C23T8 | **q5_K** | **q8_0** | **all big dense q8_0** | q6_K |

Dense bytes read per token: UD 4,606 MiB, C1 3,289 MiB, C2T8 3,709 MiB. Files are about 103 GiB
either way; the experts dominate the file, the dense path dominates the step.

## Quick screen: quality and a short drafter test

KL divergence against UD's logits on wikitext-2 (4 x 2048 tokens), plus greedy decode through the
production launcher on 6 real agent prompts x 512 tokens. UD: 39.60 tok/s, acceptance 0.644.

| Variant | PPL ratio vs UD | Mean KLD | Same top token | Acceptance | Decode vs UD |
|---|---|---|---|---|---|
| C1 | 1.020 | 0.076 | 92.4% | 0.625 | +5.4% |
| C2 | 1.006 | 0.054 | 93.4% | 0.676 | +8.9% |
| T8 | 1.016 | 0.072 | 92.4% | 0.722 | +15.5% |
| C3 | 1.006 | 0.038 | 94.5% | 0.660 | +7.0% |
| C2T8 | 0.997 | 0.052 | 93.7% | 0.683 | +10.0% |
| C23T8 | 0.997 | 0.031 | 94.7% | (not screened) | |

C23T8's KLD was measured with the model split 84/16 across both GPUs, the others on the iGPU alone.

- `ssm_out` carries most of C1's quality loss: keeping it at 8-bit (C2) takes the perplexity ratio
  from 1.020 to 1.006.
- T8's acceptance jump did **not** reproduce once combined with C2 (0.683). Six prompts at 512
  tokens are too few for acceptance: each variant writes different text, and acceptance depends
  heavily on content. The quality columns are reliable because they score identical text; the
  acceptance column is not.

## Production shape: the gain mostly disappears

The deciding test: the production launcher (np1, MTP drafter on the XTX at depth 3, coupled
sampling, 84/16 iGPU/XTX layer split, q8_0 KV, temperature 1.0), replaying 12 real Terminal-Bench
2.1 agent requests (prompts 2.6k-19k tokens, up to 2,048 generated tokens each, fixed seeds).
Decode is token-weighted over the 12 requests.

| Model | Decode | vs UD | Acceptance | Prefill |
|---|---|---|---|---|
| UD-Q4_K_XL | 40.58 tok/s | | 0.687 | 237 tok/s |
| C1 | 42.20 tok/s | +4.0% | 0.642 | 245 tok/s |
| **C2T8** | 41.75 tok/s | **+2.9%** | 0.662 | 245 tok/s |
| C23T8 | 41.40 tok/s | +2.0% | 0.668 | 247 tok/s |

For scale, without a drafter at np1 C1 decodes at 28.15 tok/s against UD's 23.76 (+18.5%).

Two things eat the gain:

1. **Acceptance drops on every requant, even at equal quality.** C23T8 matches UD more closely
   than any other variant (mean KLD 0.031, 94.7% same top token) and still loses about 2 points.
   The MTP head reads the target's final hidden state; any change to the dense path moves that
   state slightly, and the head was fitted to UD's.
2. **The dense read is shared across the verify batch.** At depth 3 the dense weights are read once
   for 4 tokens, so saving on them matters less than at np1 without a drafter.

Prefill is about 4% faster on every requant (245-250 vs 237 tok/s), consistently across runs.

## The acceptance loss sits at draft positions 2 and 3

At np2 with draft depth 1 (same launcher, `-np 2`, two clients pulling the 12 requests from a
shared queue), the loss disappears:

| Model (np2, depth 1) | End-to-end, prefill included | Per-stream decode | Acceptance |
|---|---|---|---|
| UD-Q4_K_XL | 20.42 tok/s | 15.20 tok/s | 0.812 |
| C2T8 | 21.39 tok/s (+4.7%) | 15.77 tok/s | 0.821 |
| C23T8 | 21.17 tok/s (+3.7%) | 15.46 tok/s | 0.810 |

Wall time is not a speed signal here: at temperature 1.0 each model writes different lengths, and
C23T8's shorter wall came from one request ending at 224 tokens instead of 2,048.

For comparison, end-to-end on the same 12 requests at np1 depth 3 is 18.37 tok/s for UD (prefill
is about 55% of the time on these prompts), so np2 moves about 11% more total work, at about 15
tok/s per stream instead of about 41.

## Draft depth 2 does not help, and a simple cost model predicts it

Fit to the depth-3 runs: tokens per step = 1 + depth x acceptance, so a UD step takes 75.4 ms. A
no-drafter step is about 42 ms, so each draft position costs about (75.4 - 42) / 3 = 11 ms (drafter
step, verify token and the extra experts it touches). The dense savings come off every step at
about 4.4 us per MiB (UD - C2T8 = 3.9 ms, UD - C1 = 6.1 ms). With per-position acceptance p from
the depth-3 rate (p + p^2 + p^3 = 3 x acceptance), the model predicts depth 2:

| Model, depth 2 | Predicted | Measured |
|---|---|---|
| UD-Q4_K_XL | 38.8 tok/s | 38.44 tok/s (acceptance 0.754) |
| C2T8 | 40.7 tok/s | 41.39 tok/s (acceptance 0.767) |

Depth 2 loses for every model because the third draft token costs about 11 ms and is still
accepted about 53% of the time, about 21 ms per extra token against about 24 ms per token on an
average step. The requant's lead does grow at depth 2 (+7.7% over UD at depth 2), as the dense
read is shared by fewer tokens, but not enough to beat depth 3.

## Caveats

- Every production number is a single 12-request run. Same-config reruns on this box differ by
  about 3%, so +2 to +5% is at the edge of noise.
- Temperature 1.0 means each model generates different text; token-weighted decode rate is the
  comparable number, wall time is not.
- The fix that would change the verdict is re-fitting the MTP head to the requantized target.
  That was not attempted.
- One operational note: the host crashed during this sweep, possibly an out-of-memory event (unconfirmed: the journal lost its last minutes)
  when a freshly built 103 GiB model was loaded onto the iGPU alone while its just-written files
  still sat in page cache (128 GB unified memory). The later runs split the model across both GPUs,
  dropped the new files from page cache before loading, and ran a MemAvailable watchdog; none
  tripped.
