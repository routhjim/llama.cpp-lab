#!/usr/bin/env python3
"""build_corpus.py — assemble a routing-calibration corpus for MoE expert pruning.

Why this is not just an imatrix corpus
--------------------------------------
An imatrix corpus estimates per-channel activation magnitudes; it is small and broad
on purpose. Expert-pruning calibration is a different job: an expert that looks cold
only because its domain is absent from the corpus gets pruned. Published results are
consistent on this — general tasks stay stable across general corpora, but math and
code need matched calibration data. So the mix is deliberately weighted toward what
this machine is actually used for, rather than toward generic diversity.

Mix (defaults; override with --mix)
-----------------------------------
  code       30%   local repositories — the language mix you actually write
  reasoning  25%   generated CoT traces (see gen_reasoning.py)
  prose      35%   bartowski calibration_datav3 — general + technical + multilingual
  math       10%   GSM8K train split

The reasoning slice matters most and no public corpus supplies it: V4-Flash-0731 is a
reasoning variant, and routing during extended chain-of-thought plausibly differs from
routing on flat prose. Calibrate on prose alone and you may delete experts that only
wake up mid-reasoning.

Sources are all fetchable without HF auth or parquet support. A source that fails is
warned about and its share redistributed proportionally, rather than aborting.

Usage
-----
    python3 build_corpus.py --model <gguf> --target-tokens 1000000 \\
        --code-root ~/llama.cpp --code-root ~/myproject \\
        --reasoning-file traces.txt -o corpus.txt

    python3 build_corpus.py --model <gguf> --dry-run     # report mix, write nothing
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

DATAV3_URL = ("https://gist.githubusercontent.com/bartowski1182/"
              "eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt")
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/train.jsonl")

# exintro is REQUIRED for bulk: MediaWiki caps full-article extracts at ONE page per
# request no matter what exlimit says, so without it this degrades to a request per
# document and rate-limits itself into uselessness. With exintro, exlimit=20 genuinely
# returns 20 lead sections per call.
WIKI_API = ("https://{lang}.wikipedia.org/w/api.php?action=query&generator=random"
            "&grnnamespace=0&grnlimit=20&prop=extracts&explaintext=1&exintro=1"
            "&exlimit=20&format=json")
ARXIV_API = ("http://export.arxiv.org/api/query?search_query={q}"
             "&start={start}&max_results=200")
ARXIV_CATS = ["cs.LG", "cs.CL", "cs.DC", "cs.OS", "math.OC", "stat.ML"]
WIKI_LANGS = ["de", "fr", "es", "ja", "zh", "ru", "pt", "it"]
UA = "moe-trace-corpus/1.0 (calibration corpus assembly; llama.cpp tooling)"

DEFAULT_MIX = {
    "code":         0.30,   # local repos — the language mix actually written here
    "reasoning":    0.25,   # generated CoT (gen_reasoning.py)
    "general":      0.15,   # English Wikipedia + calibration_datav3
    "technical":    0.15,   # arXiv abstracts across CS/math
    "math":         0.10,   # GSM8K
    "multilingual": 0.05,   # non-English Wikipedia
}

CODE_EXT = {".c", ".h", ".cpp", ".hpp", ".cc", ".cu", ".py", ".rs", ".go", ".js",
            ".ts", ".java", ".rb", ".sh", ".lua", ".sql", ".m", ".swift", ".kt"}
CODE_SKIP_DIRS = {".git", "build", "node_modules", "vendor", "third_party", "target",
                  "__pycache__", ".venv", "dist", "models", ".cache"}
CODE_MAX_BYTES = 200_000     # skip generated/minified monsters
DOC_SEP = "\n\n"


# ---------------------------------------------------------------- fetching

def cache_dir() -> Path:
    d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "moe-trace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(url: str, name: str) -> str | None:
    dst = cache_dir() / name
    if dst.exists() and dst.stat().st_size > 0:
        return dst.read_text(errors="replace")
    try:
        print(f"  fetching {name} ...", file=sys.stderr)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dst.write_text(r.text)
        return r.text
    except Exception as e:
        print(f"  WARN: could not fetch {name}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------- collectors

def fetch_paged(name: str, pages, want: int, min_len: int = 400) -> list[str]:
    """Accumulate documents from a paging generator, cached as JSON across runs.

    The cache is EXTENDED, not just returned: a later run with a bigger token target
    needs more documents than a smaller earlier run cached. Silently reusing the short
    cache is how a 1M-token corpus ends up with a 200K-token run's prose slice — the
    slice starves, the backfill over-weights whatever source was largest, and the
    resulting expert ranking is biased toward it.
    """
    dst = cache_dir() / name
    docs: list[str] = []
    if dst.exists() and dst.stat().st_size > 0:
        try:
            docs = json.loads(dst.read_text())
        except json.JSONDecodeError:
            docs = []
    if len(docs) >= want:
        print(f"  {name}: {len(docs)} docs (cached)", file=sys.stderr)
        return docs

    if docs:
        print(f"  {name}: cache has {len(docs)}, need {want} — fetching more",
              file=sys.stderr)
    try:
        for batch in pages:
            docs.extend(d for d in batch if len(d) >= min_len)
            print(f"  {name}: {len(docs)}/{want} docs", file=sys.stderr, end="\r")
            if len(docs) >= want:
                break
    except Exception as e:
        print(f"\n  WARN: {name} fetch stopped early: {e}", file=sys.stderr)
    print(f"  {name}: {len(docs)}/{want} docs", file=sys.stderr)
    if docs:
        dst.write_text(json.dumps(docs))
    return docs


def polite_get(url: str, headers: dict | None = None, delay: float = 1.0, tries: int = 5):
    """GET with backoff on 429. These are free public APIs — do not hammer them."""
    import time
    for attempt in range(tries):
        r = requests.get(url, headers=headers or {}, timeout=120)
        if r.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"  429, backing off {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(delay)
        return r
    raise RuntimeError("rate limited after retries")


def _wiki_pages(lang: str, target_docs: int):
    got = 0
    while got < target_docs:
        r = polite_get(WIKI_API.format(lang=lang), headers={"User-Agent": UA})
        pages = r.json().get("query", {}).get("pages", {})
        batch = [p.get("extract", "").strip() for p in pages.values()]
        batch = [b for b in batch if b]
        if not batch:
            return
        got += len(batch)
        yield batch


def _arxiv_pages(target_docs: int):
    import time
    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom"}
    got, start = 0, 0
    q = "+OR+".join(f"cat:{c}" for c in ARXIV_CATS)
    while got < target_docs:
        r = requests.get(ARXIV_API.format(q=q, start=start), timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        batch = []
        for e in root.findall("a:entry", ns):
            title = (e.findtext("a:title", "", ns) or "").strip()
            summ = (e.findtext("a:summary", "", ns) or "").strip()
            if summ:
                batch.append(f"{title}\n\n{summ}")
        if not batch:
            return
        got += len(batch)
        start += 200
        yield batch
        time.sleep(3)   # arXiv asks for 3s between requests


def collect_general(n_docs: int) -> list[str]:
    docs = fetch_paged("wikipedia_en.json", _wiki_pages("en", n_docs), n_docs, min_len=250)
    txt = fetch(DATAV3_URL, "calibration_datav3.txt")
    if txt:
        # datav3 is line-oriented — one passage per line, not blank-line separated.
        docs += [d.strip() for d in txt.splitlines() if len(d.strip()) > 120]
    return docs


def collect_technical(n_docs: int) -> list[str]:
    return fetch_paged("arxiv.json", _arxiv_pages(n_docs), n_docs, min_len=300)


def collect_multilingual(n_docs: int) -> list[str]:
    per = max(20, n_docs // len(WIKI_LANGS))
    docs = []
    for lang in WIKI_LANGS:
        docs += fetch_paged(f"wikipedia_{lang}.json", _wiki_pages(lang, per), per, min_len=250)
    return docs


def collect_math() -> list[str]:
    txt = fetch(GSM8K_URL, "gsm8k_train.jsonl")
    if not txt:
        return []
    docs = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        q, a = rec.get("question", ""), rec.get("answer", "")
        if q and a:
            docs.append(f"Problem: {q}\n\nSolution: {a}")
    return docs


def collect_code(roots: list[str]) -> list[str]:
    docs = []
    for root in roots:
        rp = Path(root).expanduser()
        if not rp.is_dir():
            print(f"  WARN: code root not a directory: {rp}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(rp):
            dirnames[:] = [d for d in dirnames if d not in CODE_SKIP_DIRS
                           and not d.startswith(".")]
            for fn in filenames:
                if Path(fn).suffix not in CODE_EXT:
                    continue
                fp = Path(dirpath) / fn
                try:
                    if fp.stat().st_size > CODE_MAX_BYTES or fp.stat().st_size < 200:
                        continue
                    docs.append(fp.read_text(errors="replace"))
                except OSError:
                    continue
    return docs


def collect_reasoning(path: str | None) -> list[str]:
    if not path:
        print("  WARN: no --reasoning-file; run gen_reasoning.py to produce one",
              file=sys.stderr)
        return []
    p = Path(path).expanduser()
    if not p.exists():
        print(f"  WARN: reasoning file not found: {p}", file=sys.stderr)
        return []
    # gen_reasoning.py separates traces with a form feed
    return [d.strip() for d in p.read_text(errors="replace").split("\f") if d.strip()]


# ---------------------------------------------------------------- tokenizing

def count_tokens(model: str, text: str) -> int:
    """Exact token count via llama-tokenize. Cheap: mmap means model weights are
    never paged in, only the vocab."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        out = subprocess.run(
            [str(TOKENIZER), "-m", model, "-f", tmp, "--ids"],
            capture_output=True, text=True, timeout=600,
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip()[:300])
        body = out.stdout[out.stdout.find("[") + 1: out.stdout.rfind("]")]
        return sum(1 for x in body.split(",") if x.strip())
    finally:
        os.unlink(tmp)


def slice_ratio(model: str, docs: list[str], sample_bytes: int = 400_000) -> float:
    """Bytes-per-token for this slice, measured on a sample.

    Tokenizing every document separately would mean thousands of subprocess spawns;
    one measured ratio per slice hits the target proportions within a percent or so,
    and the final corpus is tokenized exactly for the report.
    """
    sample, total = [], 0
    for d in docs:
        sample.append(d)
        total += len(d)
        if total >= sample_bytes:
            break
    text = DOC_SEP.join(sample)
    if not text:
        return 3.5
    n = count_tokens(model, text)
    return len(text) / n if n else 3.5


# ---------------------------------------------------------------- assembly

def take_tokens(pool: list[str], budget: int, bpt: float) -> tuple[list[str], list[str], int]:
    """Take documents off a (pre-shuffled) pool until the token budget is met.

    Returns (taken, remaining, tokens_taken). A slice that runs dry returns fewer
    tokens than asked for — the caller must not treat that as success, since silently
    under-filling a slice is exactly how a corpus ends up skewed toward whatever
    source happened to be largest.
    """
    out, got, i = [], 0, 0
    for i, d in enumerate(pool):
        est = int(len(d) / bpt)
        if est == 0:
            continue
        out.append(d)
        got += est
        if got >= budget:
            return out, pool[i + 1:], got
    return out, [], got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="GGUF used for exact tokenization")
    ap.add_argument("--tokenizer", default=None, help="path to llama-tokenize")
    ap.add_argument("--target-tokens", type=int, default=1_000_000)
    ap.add_argument("--code-root", action="append", default=[],
                    help="repeatable; directories to harvest code from")
    ap.add_argument("--reasoning-file", default=None)
    ap.add_argument("--mix", default=None,
                    help="override shares, e.g. code=0.4,reasoning=0.2,prose=0.3,math=0.1")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("-o", "--output", default="corpus.txt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global TOKENIZER
    TOKENIZER = Path(args.tokenizer) if args.tokenizer else (
        Path(__file__).resolve().parents[2] / "build" / "bin" / "llama-tokenize")
    if not TOKENIZER.exists():
        sys.exit(f"llama-tokenize not found at {TOKENIZER} (pass --tokenizer)")

    mix = dict(DEFAULT_MIX)
    if args.mix:
        for part in args.mix.split(","):
            k, v = part.split("=")
            mix[k.strip()] = float(v)

    rng = random.Random(args.seed)

    # Rough doc counts to request from the paged APIs, from each slice's token budget.
    def n_docs_for(slice_name: str, bytes_per_doc: int) -> int:
        return int(args.target_tokens * mix.get(slice_name, 0) * 4 / bytes_per_doc) + 40

    print("collecting sources", file=sys.stderr)
    pools = {
        "code":         collect_code(args.code_root),
        "reasoning":    collect_reasoning(args.reasoning_file),
        "general":      collect_general(n_docs_for("general", 500)),
        "technical":    collect_technical(n_docs_for("technical", 1200)),
        "math":         collect_math(),
        "multilingual": collect_multilingual(n_docs_for("multilingual", 500)),
    }

    # Redistribute the shares of any slice that came back empty.
    missing = [k for k, v in pools.items() if not v]
    if missing:
        lost = sum(mix[k] for k in missing)
        keep = {k: v for k, v in mix.items() if k not in missing}
        if not keep:
            sys.exit("every source failed — nothing to build")
        scale = 1.0 + lost / sum(keep.values())
        mix = {k: v * scale for k, v in keep.items()}
        print(f"WARN: empty slices {missing}; redistributing {lost:.0%}", file=sys.stderr)

    print("\nmeasuring and selecting", file=sys.stderr)
    selected, report, leftovers = [], [], {}
    for name, share in mix.items():
        docs = pools[name][:]
        rng.shuffle(docs)
        bpt = slice_ratio(args.model, docs)
        budget = int(args.target_tokens * share)
        picked, rest, got = take_tokens(docs, budget, bpt)
        selected.extend(picked)
        leftovers[name] = (rest, bpt)
        report.append([name, share, len(pools[name]), len(picked), got, bpt, budget])

    # Any slice that ran dry leaves a hole. Fill it from slices that still have
    # unused documents rather than shipping a corpus that is quietly short.
    shortfall = sum(max(0, r[6] - r[4]) for r in report)
    if shortfall > 0:
        short_names = [r[0] for r in report if r[6] - r[4] > 0]
        print(f"WARN: {shortfall:,} tokens short on {short_names} — "
              f"backfilling from slices with headroom", file=sys.stderr)
        donors = [r for r in report if leftovers[r[0]][0]]
        if donors:
            per = shortfall // len(donors)
            for r in donors:
                rest, bpt = leftovers[r[0]]
                extra, _, got = take_tokens(rest, per, bpt)
                selected.extend(extra)
                r[3] += len(extra)
                r[4] += got
        else:
            print("WARN: no slice has spare documents; corpus will be under target",
                  file=sys.stderr)

    # Interleave slices so context windows are not domain-homogeneous.
    rng.shuffle(selected)
    corpus = DOC_SEP.join(selected)

    total_est = sum(r[4] for r in report) or 1
    print(f"\n{'slice':>13} {'target':>7} {'actual':>7} {'avail':>7} {'used':>7} "
          f"{'est.tok':>10} {'B/tok':>6}", file=sys.stderr)
    for name, share, avail, used, got, bpt, _budget in report:
        print(f"{name:>13} {share:>6.0%} {got/total_est:>6.0%} {avail:>7} {used:>7} "
              f"{got:>10,} {bpt:>6.2f}", file=sys.stderr)

    actual = count_tokens(args.model, corpus)
    print(f"\ncorpus: {len(selected):,} documents, {len(corpus):,} bytes, "
          f"{actual:,} tokens (exact)", file=sys.stderr)

    # ~1.12 KB of trace per token for V4 Flash (43 layers, top-6, hc=4)
    print(f"expected trace size: {actual * 1122 / 1e9:.2f} GB", file=sys.stderr)

    if args.dry_run:
        print("dry run — nothing written", file=sys.stderr)
        return

    Path(args.output).write_text(corpus)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
