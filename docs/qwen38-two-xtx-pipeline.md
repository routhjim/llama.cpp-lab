# Qwen3.8-27B on two 7900 XTX: one copy of the weights, pipelined instances

Dense Qwen3.8-27B (Q4_K_S) layer-split over two Radeon RX 7900 XTX in USB4 docks, serving many concurrent agent
streams (Terminal-Bench 2.1) with large KV reserved per stream. Code: [#54](https://github.com/routhjim/llama.cpp-lab/pull/54),
[#55](https://github.com/routhjim/llama.cpp-lab/pull/55), [#56](https://github.com/routhjim/llama.cpp-lab/pull/56).
Launcher and router: [`scripts/two-xtx/`](../scripts/two-xtx/).

## Result

8 concurrent real agent prompts (2 instances x 4 slots, 1500 tokens each, warm prompt cache), same harness throughout:

| | aggregate | GPU busy |
|---|---|---|
| 2 instances x np4, before | 115.4 t/s | 62 / 59 % |
| + host<->device rings, async scheduler inputs ([#55](https://github.com/routhjim/llama.cpp-lab/pull/55)) | 185.4 / 185.5 t/s | 88 / 84 % |
| + dock PCIe links retrained to Gen4 ([docs](egpu-dock-pcie-gen1.md)) | **268.8 / 259.0 t/s** | 89 / 87 % |

Draft acceptance stayed 0.85 / 0.78-0.80 per instance throughout.

## Design

- **Why one context is not enough.** A context decodes all its slots in lock-step. With a layer split, GPU 1 runs its
  layers, then GPU 2, so each card idles half the time. Splitting the batch into micro-batches re-reads the weights per
  micro-batch and was slower (np4 61.7 -> 39.4 t/s).
- **N contexts on one model** (`LLAMA_SERVER_INSTANCES`, #56): each instance decodes in its own thread, so instance A
  uses GPU 1 while instance B uses GPU 2. The weights are loaded once. pipe-router.py keeps each conversation on one
  instance (KV and prompt cache are per instance).
- **Balance the stages.** Card 2 holds the output head, so all MTP drafters go on card 1 and card 1 gets fewer layers
  (48/52 at np4, 45/55 at np2). Forced per-device ordering (`GGML_PIPE_ORDER`) gave nothing once balanced.
- **Reserve KV per stream.** 4 x 131072 tokens per instance, q4_0 KV (perplexity 5.7396 vs 5.7333 for q8_0, +0.11%).
  VRAM 21.9 / 19.3 GiB.

Early measurements, 4 streams: 1 instance x np4 69.3 t/s; 2 instances x np2 93.7; with drafters on card 1 at 45/55,
135-146 t/s, against 162 for two fully independent cards each holding its own weights.

## Where the time went

Stack sampling of the two decode threads under load (eu-stack, leaf frames classified):

| decode thread time, 2 x np4, before #55 | share |
|---|---|
| waiting on output readback (staging + full sync on card 2) | 38% |
| waiting before each cross-card copy | 33% |
| synchronous input uploads (staging + fence, device lock) | 12% |
| end-of-step synchronize (the thread waiting for its own GPU work) | 11% |

The Vulkan host buffer is pinned for device 0 only, so every logits read from card 2 took the synchronous staging
path; every small upload to a card without host-visible VRAM waited on a fence; and the scheduler synchronized the
destination before each split input. #55 moves small transfers into per-context rings and drops the per-input waits.

## What did not work

| Idea | Result |
|---|---|
| Device -> device copies with no CPU wait: dma-buf (or userptr) bridge + `sync_fd` semaphore, external queue-family barriers | 113 vs 191 t/s without it. A queue submission waiting on another device's fence stalls that device; the blocking host-staged copy is cheap once the rest is async |
| Import host buffers into every device (userptr) so any card can use pinned memory | amdgpu revalidates userptr pages on every command submission: ~26% of the decode threads' time in the CS ioctl |
| One compute queue per context | no gain (185 vs 190 t/s) |
| Micro-batching one context's decode (`LLAMA_DECODE_UBATCH`) | np4 61.7 -> 39.4 t/s |
| Forced per-device ordering of instances (`GGML_PIPE_ORDER`) | no gain once the stages are balanced |
| Drafters on the iGPU as a third stage; 3 instances x np3 | worse than 2 instances with drafters on card 1 |
| DFlash drafter instead of MTP | 64-77 t/s for one stream, but 65.6 at 2 streams on a card; MTP stays |
| Target GPU sampling (`-bs`) to cut the logits readback | 117.8 vs 115.4 t/s (noise), acceptance 0.81 -> 0.73 |

A bug on the way: dropping the scheduler's synchronize let Vulkan's zero-copy read of a CPU split's output run after
the next graph's CPU split had reused the buffer (one instance at acceptance 0.37). A toggle bisect on one binary
found it; host sources now go through the upload ring. A debug mode that synchronizes after every copy could not see
it. An earlier bug (#54): a cross-device copy's staging buffer refilled by another thread between its two steps.

## Terminal-Bench 2.1

Terminal-Bench 2.1, 83 tasks ordered shortest first, MEDIUM reasoning, 16 tasks live, 2 instances x np4 with 131072
tokens reserved per slot, same harness and instructions for all three runs. tb21x2b was stopped at 77.5 minutes
(2026-10-03).

| | one XTX, np2 (09-20) | two XTX, before #55, PCIe Gen1 links | **two XTX, #55 + Gen4 links** |
|---|---|---|---|
| tasks scored / passed in the first 60 min | 16 / 15 | 17 / 16 | **32 / 29** |
| tasks scored / passed at 77.5 min | 17 / 16 | 23 / 21 | **37 / 34** |
| decode per stream (server timings) | | 13.7 t/s | **23.7 t/s** |
| prefill per request | | 274 t/s | **434 t/s** |
| aggregate delivered | 67 -> 54 t/s | ~100 t/s | **~171 t/s** |
| draft acceptance | | 0.861 | 0.869 |

The three failures: cancel-async-tasks, query-optimize, mteb-retrieve.

Same outcomes, faster (compared at 68 minutes): on the 26 tasks both two-XTX runs have scored, every pass/fail matches; against the one-XTX
run (34 in common) one task differs each way (query-optimize passed only there, configure-git-webserver passes only
here). Median wall time per task: 14.6 min vs 28.6 for the Gen1 run.

## Running it

```sh
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
MODEL=Qwen3.8-27B-Q4_K_S.gguf DRAFT=mtp-Qwen3.8-27B-Q4_K_M.gguf TMPL=qwen38.jinja \
  scripts/two-xtx/run-qwen38-2x.sh --reasoning-format deepseek -rea on --reasoning-effort medium
# one endpoint on :8090, instances on :8080 / :8081
```

Defaults: 2 instances x np4, 524288 tokens per instance (131072 reserved per slot), split 48/52, MTP depth 2,
`-ub 256`. Check the dock links first ([egpu-dock-pcie-gen1.md](egpu-dock-pcie-gen1.md)).
