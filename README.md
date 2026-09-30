# llama.cpp-lab

**Measured llama.cpp changes for AMD Strix Halo (gfx1151) on Vulkan/RADV, plus a Radeon RX 7900 XTX over USB4.**

This is one person's working copy of llama.cpp, tuned against two models on one machine:
Qwen3.8-Flash-Next (a 180B sparse MoE with hyper-connections and a sparse-attention indexer)
on the integrated GPU, and dense Qwen3.8-27B on the eGPU. Every change here was merged through
a PR with a before/after measurement, and the negative results are recorded next to the wins.

It is not a GitHub fork of `ggml-org/llama.cpp`, and none of this work is in upstream llama.cpp.
The upstream README is preserved unchanged in [`UPSTREAM-README.md`](UPSTREAM-README.md); read it for
general llama.cpp usage, backends and bindings.

## The machine

| | |
|---|---|
| APU | AMD Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X unified memory |
| eGPU | Radeon RX 7900 XTX, 24 GB, USB4 dock |
| Backend | Vulkan (Mesa RADV), not ROCm. Vulkan decodes 1.64x faster than ROCm on this box |
| OS | Fedora 44, kernel 7.2, `amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856` |
| Decode roofline | about 215 GB/s on the iGPU; a Flash-Next token reads ~7 GB at np1 |

Numbers from a different power envelope, kernel command line or quant will not match these.

## Results

Each row is one merged change, measured on this machine with the other backend unloaded.

| Change | PR | Before | After | Conditions |
|---|---|---|---|---|
| MTP (nextn) speculative head for Qwen3.8-Flash-Next | [#1](https://github.com/routhjim/llama.cpp-lab/pull/1) | no MTP | **1.3x to 1.63x decode**, 62-86% acceptance | iGPU, np1, greedy |
| Pack GQA flash-attention tiles across tokens, 32/48-row coopmat1 tiles | [#41](https://github.com/routhjim/llama.cpp-lab/pull/41), [#42](https://github.com/routhjim/llama.cpp-lab/pull/42) | 8259 µs | **3573 µs (2.3x)** FA op, 8 rows | iGPU, q8_0 KV, 65k context |
| same, end to end | | 0.652 ms per 1k context | **0.167 ms** per 1k context | XTX, Qwen3.8-27B, real np2 traffic |
| QSA pooled key cache allocated per layer device | [#38](https://github.com/routhjim/llama.cpp-lab/pull/38) | 16.60 tok/s | **21.80 tok/s (+31%)** | Flash-Next split 88/12 iGPU/XTX, 60k context |
| MUL_MAT_ID GEMV threshold 8 → 32, with RDNA3 row counts sized to the column count | [#31](https://github.com/routhjim/llama.cpp-lab/pull/31) | 66.6 tok/s | **80.7 tok/s (+21%)** | Flash-Next batched decode, batch 16 |
| Prompt-cache disk tier made NVMe-resident, not RAM-gated | [#37](https://github.com/routhjim/llama.cpp-lab/pull/37) | 13% cache hit, task killed at 6 h | **98% hit at 114k context, task passed in 86 min** | Terminal-Bench 2.1 task, one run each |
| Hot slot packing: live sequences move to the lowest KV streams | [#16](https://github.com/routhjim/llama.cpp-lab/pull/16) | 38% per-slot penalty on a {0,3} pair | **penalty gone** (swap costs 0.7-4.3 ms) | Flash-Next, np4 |
| Flash-Next concurrency set: indexer pooling, pooled-key cache, sparse FA, upstream indexer fix | [#15](https://github.com/routhjim/llama.cpp-lab/pull/15)-[#18](https://github.com/routhjim/llama.cpp-lab/pull/18) | 227 ms/step | **156 ms/step** | Flash-Next, np4, one 59k slot + two short |
| Coupled (Gumbel-max) sampling between drafter and target | [#26](https://github.com/routhjim/llama.cpp-lab/pull/26) | 0.466 / 0.576 acceptance | **0.540 / 0.688** (code / thinking) | Qwen3.8-27B + DFlash2, temp 1.0 |
| Confidence-gated draft depth, ngram agreement gates, chunked ngram drafts, per-request `n_max`, mat-vec row override | [#43](https://github.com/routhjim/llama.cpp-lab/pull/43)-[#46](https://github.com/routhjim/llama.cpp-lab/pull/46) | 89.6 tok/s (best static MTP) | **109.2 tok/s** geomean over 5 content kinds | Qwen3.8-27B, XTX, np1 (autoregressive is 34.0) |
| Upstream PLE row prefetch ([ggml-org #29599](https://github.com/ggml-org/llama.cpp/pull/29599)), picked with #29638 (no draft accept after EOG) and #29280 (Vulkan descriptor reuse) | [#51](https://github.com/routhjim/llama.cpp-lab/pull/51) | 215 / 247 / 239 tok/s prefill | **262 / 289 / 272 (+22% / +17% / +14%)** at 512 / 4k / 16k | Flash-Next C2T8, 84/16 split, MTP 3, cold page cache, ABBA; decode unchanged |
| MTP draft head re-fit to the requantized target (C2T8): soft CE vs the target's top-20 at depths 1-3, unrolled like the draft KV | [#52](https://github.com/routhjim/llama.cpp-lab/pull/52) | 0.651 acceptance, 38.7 tok/s | **0.692 acceptance, 40.0 tok/s (+3.4%)** | Flash-Next C2T8, 84/16 split, MTP 3, 12 real TB2.1 requests; [weights](https://huggingface.co/Jrouth/Qwen3.8-Flash-Next-MTP-C2T8-refit-GGUF) |

The Flash-Next concurrency work (#15-#18) has its own write-up:
[`docs/flash-next-concurrency.md`](docs/flash-next-concurrency.md).
The dense-path requant study has its own too:
[`docs/flash-next-dense-requant.md`](docs/flash-next-dense-requant.md).
The MTP head re-fit that follows it:
[`docs/flash-next-mtp-refit.md`](docs/flash-next-mtp-refit.md).

## Findings that are not code

- **Ragged verify batches cost ~205 ms per step.** When slots in one batch draft different
  lengths, the decode splits into small ubatches. A per-token `p_min` gate "lost every test"
  on this box because of that, not because gating is bad.
- **Dense tensors, not experts, dominate Flash-Next decode at np1**: 57% of a step is the
  8-bit dense path, 21% the MoE experts. At np4 and above the expert union takes over.
- **Decode is linear in KV length** on Flash-Next at np1: 41.53 ms + 0.389 µs per token of context,
  with no cliffs.
- **A post-hoc n-gram memory table helps prose, not commands.** An Engram-style hashed 2/3-gram
  table (4M rows, gated injection after layer 3) trained for 3M tokens on a *frozen* Qwen3.8-27B,
  the post-hoc version of the table Flash-Next trains jointly: perplexity -5.7% on held-out text
  from its training mix and -5.1% on an agent's analysis prose, but -0.3% on the agent's own shell
  command blocks and -0.12% on wikitext. Only 1.6% of Terminal-Bench 2.1 steps were fixable by it.
  On gfx1151, fla's Triton gated-DeltaNet kernels run forward but return NaN gradients backward;
  train with the plain PyTorch path.
- **A target requant costs the MTP drafter, even at equal quality.** Requantizing Flash-Next's
  8-bit dense path with an importance matrix can keep perplexity equal to UD-Q4_K_XL (ratio 0.997),
  yet MTP acceptance still drops 2-3 points (0.687 to 0.662-0.668 at depth 3), and only at draft
  positions 2 and 3: at depth 1 it is unchanged. So C1's +18.5% without a drafter becomes +4.0%
  in production (C2T8: +2.9%). A no-drafter benchmark overstates any requant of a model served with MTP.
- **Speculative step cost is simple enough to predict.** On Flash-Next at np1: a no-drafter step is
  about 42 ms, each draft position adds about 11 ms, and dense bytes saved come off at about 4.4 us
  per MiB. Fitted on depth 3, it predicted depth 2 within 0.7 tok/s for both models tested.

## What did not work

Recorded so nobody has to measure them again.

| Idea | Result |
|---|---|
| ngram-mod drafting on Flash-Next | net negative in all 5 tests: -1.7% on prose, -48% on copy-heavy output. A 512-expert top-10 MoE pays for the union of every wide verify batch |
| Speculative lookahead (draft the next batch during verify) | -7.1% on real traffic, even after fixing four concurrency bugs |
| Sparse flash attention at prefill | -18% at 16k, -10% at 32k, +7% only at 64k |
| Drafter temperature calibration (0.7-1.5) and support truncation (top-k 10/15/32) | no effect on acceptance. Re-fitting the head to the target does help: see [#52](https://github.com/routhjim/llama.cpp-lab/pull/52) |
| Adaptive MTP depth on Flash-Next at np ≥ 2 | -5 to -9% per slot; fixed depth wins |
| Naive q4_K requant of Flash-Next's dense path | +23% decode, but +6.4% perplexity |
| Importance-matrix requant of the dense path, served with MTP | quality equal to UD, but +2 to +4% decode in production (C1: +18.5% without a drafter). We run C2T8 anyway; see [the write-up](docs/flash-next-dense-requant.md). A re-fit MTP head then adds +4 points of acceptance ([#52](https://github.com/routhjim/llama.cpp-lab/pull/52)). |
| MTP draft depth 2 on Flash-Next at np1 | -5.3% vs depth 3 on UD, -0.9% on C2T8 |
| Upstream MoE-aware `mul_mat_id` tile selection ([ggml-org #29182](https://github.com/ggml-org/llama.cpp/pull/29182)) | -6% prefill at 4k and 16k on Flash-Next (ABBA, same build otherwise); not picked |
| Scaling up a post-hoc n-gram table for agent work | -0.3% perplexity on agent command blocks after 14.7 h of training on the iGPU; not worth a bigger table |

## Build

```sh
cmake -B build -DGGML_VULKAN=ON -DGGML_VULKAN_MMV_MAX_COLS=32 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`-DGGML_VULKAN_MMV_MAX_COLS=32` matters: the default of 16 puts a decode cliff at verify batches above 16.

Flash-Next on the iGPU, the way it runs here:

```sh
build/bin/llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md mtp-Qwen3.8-Flash-Next-Q4DRAFT.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-coupled \
  -ngl 99 -ngld 99 -fa on -ctk q8_0 -ctv q8_0 -lm mmap -lzm auto -c 262144 -np 1 --jinja
```

Always pass `-lm mmap` for Flash-Next: `auto` disables mmap for every device when one device
lacks it, and the load then needs the whole file in RAM.

## Credits

- The Flash-Next MTP graph is derived from JJJYmmm's closed upstream PR #27739.
- The flash-attention dequant change this build carries is Nathanw1014's upstream PR #28190.
- Everything else builds on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT licensed.

## License

MIT, as upstream. See [LICENSE](LICENSE).
