#!/usr/bin/env python3
"""gen_reasoning.py — generate chain-of-thought traces for the calibration corpus.

The reasoning slice is the one no public calibration corpus supplies, and for a
reasoning-tuned model it is the one most likely to matter: routing during extended
CoT plausibly differs from routing on flat prose, so calibrating on prose alone risks
pruning experts that only fire mid-reasoning.

Generation runs against any OpenAI-compatible endpoint. Point it at the fast
daily-driver model rather than the model being pruned — only the *text distribution*
matters here, not whose outputs they are, and V4 Flash generates at ~11 t/s while the
resident server is far quicker. Traces are appended as they land, so a long run
survives interruption and can be resumed by re-running with the same output file.

Usage
-----
    python3 gen_reasoning.py --n 400 -o traces.txt
    python3 gen_reasoning.py --endpoint http://127.0.0.1:13306/v1/chat/completions \\
        --model DeepSeek-V4-Flash-0731 --n 400 -o traces.txt
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

GSM8K_CACHE = "gsm8k_train.jsonl"

# Seeds spanning the reasoning modes this box actually sees. Math comes from GSM8K;
# these cover the code/systems/analysis side that math datasets miss.
SEEDS = [
    "Walk through what happens, step by step, when a process calls mmap() on a file "
    "larger than physical RAM and then reads it sequentially. Where does the memory "
    "pressure actually show up?",
    "A Mixture-of-Experts model has 256 experts per layer and routes top-6. Derive how "
    "the memory footprint and the per-token FLOPs each scale if you prune to 96 experts.",
    "Given a program that segfaults only under -O2 and never under -O0, reason through "
    "the most likely causes in order of probability, and how you would discriminate them.",
    "Explain why quantizing a transformer's router matrix hurts more per byte than "
    "quantizing its feed-forward weights. Reason from first principles.",
    "A service has p50 latency of 8ms and p99 of 2400ms. Work through what structural "
    "causes produce that shape, and what you would measure first.",
    "Derive the memory bandwidth required to generate one token from a 13B-active MoE "
    "at 30 tokens/sec in 4-bit, and say whether that is achievable on a 256 GB/s bus.",
    "Two sorted arrays of length n. Reason step by step to an O(log n) algorithm for "
    "the median of their union, and prove the bound.",
    "Explain how a doubly-stochastic mixing matrix keeps perturbations from amplifying "
    "across parallel residual streams. Work through the norm argument.",
    "You must cut 40% of a neural network's parameters. Compare pruning by activation "
    "frequency against pruning by gradient-based importance, and reason about when each "
    "is the wrong choice.",
    "Trace the sequence of events when a Linux system with GPU-pinned memory hits an "
    "allocation failure. Why can the OOM killer be unable to reclaim anything?",
    "Reason through why top-k routing over a pruned expert pool changes the effective "
    "sparsity a router was trained for, and what that does to output quality.",
    "A distributed job is 40x slower than single-machine despite 64 workers. Reason "
    "systematically through the candidate causes.",
]


def load_gsm8k_questions() -> list[str]:
    import os
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    p = cache / "moe-trace" / GSM8K_CACHE
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            q = json.loads(line).get("question", "")
        except json.JSONDecodeError:
            continue
        if q:
            out.append(q + "\n\nThink through this step by step.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default=None, help="model id; omitted lets the server pick")
    ap.add_argument("--n", type=int, default=400, help="number of traces")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("-o", "--output", default="traces.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prompts = SEEDS + load_gsm8k_questions()
    if len(prompts) == len(SEEDS):
        print("note: GSM8K not cached — run build_corpus.py once to populate it, "
              "for a wider math seed pool", file=sys.stderr)
    rng.shuffle(prompts)

    # Resume: count traces already present.
    out_path = Path(args.output)
    done = 0
    if out_path.exists():
        done = len([d for d in out_path.read_text(errors="replace").split("\f") if d.strip()])
        print(f"resuming — {done} traces already in {out_path}", file=sys.stderr)
    todo = args.n - done
    if todo <= 0:
        print("nothing to do", file=sys.stderr)
        return

    lock = threading.Lock()
    fout = out_path.open("a")
    written = 0
    no_think = [0]   # responses that carried no separate reasoning channel

    def one(i: int) -> int:
        prompt = prompts[(done + i) % len(prompts)]
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
        }
        if args.model:
            body["model"] = args.model
        try:
            r = requests.post(args.endpoint, json=body, timeout=900)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
        except Exception as e:
            print(f"  WARN: request failed: {e}", file=sys.stderr)
            return 0

        # Keep the reasoning channel when the server exposes it separately — that
        # text is the entire point of this slice. Field naming is NOT standardized:
        # llama.cpp and vLLM use "reasoning_content", Ollama uses "reasoning". Miss it
        # and the slice silently degrades to plain answers, which defeats the purpose.
        think = next((msg[k] for k in ("reasoning_content", "reasoning", "thinking")
                      if msg.get(k)), "")
        parts = [think, msg.get("content") or ""]
        text = "\n".join(p for p in parts if p).strip()
        if not text:
            return 0
        if not think:
            no_think[0] += 1

        with lock:
            fout.write(prompt + "\n\n" + text + "\f")
            fout.flush()
        return len(text)

    nonlocal_chars = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, i) for i in range(todo)]
        for n, f in enumerate(as_completed(futs), 1):
            nonlocal_chars += f.result()
            if n % 10 == 0 or n == todo:
                print(f"  {n}/{todo} traces, ~{nonlocal_chars/3.5:,.0f} est. tokens",
                      file=sys.stderr)
            written = n

    fout.close()
    print(f"\nwrote {written} traces to {out_path} "
          f"(~{nonlocal_chars/3.5:,.0f} est. tokens)", file=sys.stderr)
    if no_think[0]:
        pct = 100.0 * no_think[0] / max(1, written)
        lvl = "WARN" if pct > 20 else "note"
        print(f"{lvl}: {no_think[0]}/{written} ({pct:.0f}%) responses had no separate "
              f"reasoning channel — if that is most of them, the endpoint is not "
              f"returning CoT and this slice is just plain answers", file=sys.stderr)
    print("feed it to build_corpus.py with --reasoning-file", file=sys.stderr)


if __name__ == "__main__":
    main()
