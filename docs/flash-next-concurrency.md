# Flash-Next at np > 1 on Strix Halo: what costs a decode step, and what we did about it

Findings and changes from 2026-09-03/04 on Qwen3.8-Flash-Next (qwen4exp, UD-Q4_K_XL, 104 GB)
served by llama-server on the Radeon 8060S (Vulkan/RADV, unified memory). Everything below is
in `main` of this repo unless marked otherwise. Numbers are from real measurements on this box;
the harnesses live in `~/tbench` (see the end).

## The model, in one paragraph

Per decode step, four things scale with the *batch* rather than with the sequence:
the expert union of MoE verify batches (`np * (1 + n_max)` rows), the **range** of KV streams
in the batch, the **length** of the longest co-active KV, and the sparse indexer that scores
every block of every stream. Attention itself was never the big one on this model. Production
config: np=4, MTP n-max 2 + ngram-mod, q8_0 KV, 65536 tokens/slot, `--slot-pack`.

## What was fixed, in order

| # | Change | PR | Mechanism | Measured |
|---|---|---|---|---|
| 1 | Upstream sync incl. #27941 | #15 | the hybrid-idx update context never built `ctx_idx`, so a cross-stream `seq_cp` of the QSA indexer keys was never applied. Every slot-pack move/swap before this carried the previous occupant's indexer keys | silent wrong block selection after every relocation; fixed |
| 2 | Hot slot packing | #16 | `server_slot::seq` decouples the KV stream from the slot; `pack_streams()` runs between decode steps and keeps the k live sequences on streams {0..k-1}; KV/RS move with `seq_mv`, drafter state with `common_speculative_seq_swap`, backend sampler re-bound. Needs `--slot-pack` (one scratch stream) | pinned {0,3} pair 35-40 t/s agg vs {0,1} 31-34 (was -38%); 988 live relocations in a 13 h run, 0 structural failures, 1-30 ms each |
| 3 | Indexer block pooling via `ggml_pool_2d` | #17 | the r strided-slice `ggml_cont` copies per layer were the top op of a long-context step | 14.3 ms -> 3.5 ms at 30k x 3 streams; step 145 -> 134 ms |
| 4 | Incremental pooled-key cache | #17 | complete blocks are pooled/normed/roped once into a per-layer, per-stream store; per step only the freshly completed blocks are processed (dirty tables from `set_input_qsa`), scoring reads the store; watermarks per (ratio, stream) clamped by every seq op | 30k + short: 165 -> 158 ms; removes gather + norm + rope from every step |
| 5 | Vulkan sparse flash attention | #18 | `n_kv_max` -> per-row index list pre-pass + gathered scalar FA (staging loops); qwen4exp passes its top-k width, so QSA layers attend over ~2k selected cells | 59k + 2 short: 167 -> 156 ms; attention no longer grows with context |
| 6 | Per-stream FA KV bound (`GGML_VK_FA_KV_MAX=1`) | #17 | CUDA `KV_max` port; correct, but the scalar kernel already skipped masked tiles cheaply | ~4 ms gain vs ~10 ms dispatch cost here; **opt-in**, try on dense models |

Same-state totals, one 59k slot plus two short slots at np=4: **227 -> 156 ms/step**, short slots
**~11 -> ~15-17 t/s**. Weighted over last night's TB2.1 context timeline: ~16% more generated
tokens per hour, 30-45% on the 40-64k stretches where the long tasks time out.

## Negative results worth remembering

- **Adaptive MTP depth** (upstream PR #27210 and the LaurentZuijdwijk fork): on this MoE at np>=2
  the controller climbs to depth 3+ and loses 5-9%/slot; its objective is accepted tokens, but each
  depth position widens the verify batch 4 rows and the expert union with it. A cost-aware
  controller (branch `pr/adaptive-mtp`, `LLAMA_ADAPTIVE_ROI=1`) sits on the break-even line and
  collects nothing. **Fixed n-max 2 is optimal at np 1, 2 and 4.** Adaptive pays on the dense 27B.
- **Concurrency above 4** moves more requests, not more tokens: np=5 same aggregate at -20%/slot,
  np=6 loses outright; crossing the 16-column matvec limit (verify batch > 16) costs a further ~10%.
- **Attention was not where the batch-max cost lived**: the scalar FA already skipped fully
  masked tiles after a cheap mask load. Profile before building kernels
  (`GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_SHAPES=1`, `~/tbench/kvmax_prof_analyze.py`).
- A 99-minute thinking turn, a context-overflow summarization loop (72-min turn -> 67k > 65,536 ->
  280 failed summaries -> 6 h agent timeout) and a 150-frame vision request cost more TB2.1 tasks
  than any kernel. Fixes for the rerun: slot ctx 81920 with harness max_input 65536,
  `--reasoning-budget 16384`, `LOOKD_MAX_FRAMES=24`.

## Gotchas that bit

- Two GPUs since 2026-09-04 (RX 7900 XTX on an ORARA dock): ggml lists discrete first, so the iGPU
  index shifts and an unpinned launch splits the model across both (0.3 t/s). Pin by name:
  `-dev/-devd $(vkdev igpu $BIN) -mg 0`. Never combine with `GGML_VK_VISIBLE_DEVICES`, which
  renumbers the devices under `-dev`.
- Other sessions run servers here; kill only your own (`kill-my-answerer <port> <gpu-role>`).
- `--spec-type none` with `-md <MTP gguf>` segfaults (the MTP-only GGUF loaded as a plain draft).
- Output identity is not a usable gate on this model: cell layout and batch shape change FP
  summation order; expect rare late single-token flips between correct configurations.
- `wait` with no PIDs in a shell that launched the server blocks forever; `wait $pid`.

## Harnesses (`~/tbench`)

`answerer-tb21-ctrl.sh` (production launcher; `BIN NP CTX NMAX SPEC NGL GPU_ROLE` env),
`tb21v2-run.sh` (full TB2.1 with smoke gate, buckets, watchdog, tracker), `tb21np-run.sh`
(np/n-max arms), `kvmax_ab.sh` / `sparse_ab.sh` (mixed-length A/B), `kvmax_prof.sh` +
`kvmax_prof_analyze.py` (per-op profile), `qsa_cache_ab.sh` (cache correctness scenarios).
