#!/usr/bin/env python3
"""prune_experts.py — drop the least-used experts from a DeepSeek-V4 GGUF.

Reads a routing-mass matrix produced by read_trace.py (--counts) and writes a new
GGUF keeping only the top-K experts per layer.

Why this is more than a tensor slice
------------------------------------
`n_expert` is a SINGLE GLOBAL hparam in llama.cpp — `ffn_gate_inp` and the expert
stacks are created with it for every layer, including the hash-routed ones. So the
first `hash_layer_count` layers cannot be exempted from the cut, even though their
"usage" is really just vocabulary frequency. They must be pruned to the same K, and
their token->expert tables REMAPPED: any entry still pointing at a dropped expert
would be an out-of-range lookup at inference time, not merely degraded quality.

Tensors rewritten per layer:
  ffn_gate_inp.weight    [n_embd, n_expert]        -> slice expert axis
  ffn_{gate,down,up}_exps.weight  [.., .., n_expert] -> slice expert axis
  ffn_exp_probs_b.bias   [n_expert]                -> slice          (learned layers)
  ffn_gate_tid2eid.weight[n_used, n_vocab]         -> remap VALUES   (hash layers)
Shared experts (*_shexp) are never touched — the shared expert always fires.

Usage
-----
    python3 prune_experts.py --input model-00001-of-00004.gguf \\
        --mass mass.npy --keep 234 --output pruned.gguf
"""

import argparse
import sys
from pathlib import Path

import numpy as np

def _find_gguf_py() -> None:
    """Locate llama.cpp's gguf-py regardless of where this script was copied to."""
    here = Path(__file__).resolve()
    cands = [p / "gguf-py" for p in here.parents]
    cands += [Path.cwd() / "gguf-py", Path("/root/llama.cpp/gguf-py"),
              Path.home() / "llama.cpp" / "gguf-py"]
    for c in cands:
        if (c / "gguf" / "__init__.py").exists():
            sys.path.insert(0, str(c))
            return
    sys.exit("could not locate gguf-py; run from inside a llama.cpp checkout")


_find_gguf_py()
import gguf                                          # noqa: E402
from gguf.gguf_reader import GGUFReader              # noqa: E402
from gguf.gguf_writer import GGUFWriter              # noqa: E402

# Names verified against an actual GGUF, not inferred from llama.cpp's tensor enum --
# the enum is LLM_TENSOR_FFN_EXP_PROBS_B but the file says "exp_probs_b.bias" with no
# ffn_ prefix, and getting that wrong copies the tensor through unsliced (the model
# then fails to load with a shape mismatch).
SLICE_ON_EXPERT_AXIS = (
    "ffn_gate_inp.weight",     # (n_embd, n_expert)
    "ffn_gate_exps.weight",    # (n_embd, n_ff_exp, n_expert)
    "ffn_down_exps.weight",    # (n_ff_exp, n_embd, n_expert)
    "ffn_up_exps.weight",      # (n_embd, n_ff_exp, n_expert)
    "exp_probs_b.bias",        # (n_expert,)
)
REMAP_VALUES = ("ffn_gate_tid2eid.weight",)   # hash layers: values ARE expert ids

# DO NOT touch these. Their 256 is 2*n_embd_indexer for the Lightning Indexer and has
# nothing to do with expert count -- matching tensors by "shape contains n_expert"
# would silently corrupt them.
NEVER_SLICE = ("indexer_compressor_gate.weight", "indexer_compressor_kv.weight",
               "indexer_compressor_ape.weight")


def layer_of(name: str):
    if not name.startswith("blk."):
        return None
    try:
        return int(name.split(".")[1])
    except (IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="first split of the source GGUF")
    ap.add_argument("--mass", required=True, help="mass.npy from read_trace.py --counts")
    ap.add_argument("--keep", type=int, required=True, help="experts to keep per layer")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mass = np.load(args.mass)                        # [n_layer, n_expert]
    n_layer, n_expert = mass.shape
    K = args.keep
    if not 1 <= K <= n_expert:
        sys.exit(f"--keep must be in 1..{n_expert}")

    src = Path(args.input)
    stem = src.name.split("-00001-of-")[0]
    parts = sorted(src.parent.glob(f"{stem}-*.gguf")) or [src]
    print(f"source: {len(parts)} split(s)", file=sys.stderr)

    readers = [GGUFReader(str(p)) for p in parts]

    # ---- architecture + hash-layer count, straight from the source metadata ----
    arch = ""
    hash_layers = 0
    f = readers[0].fields.get("general.architecture")
    if f is not None:
        arch = str(f.contents())
    f = readers[0].fields.get(f"{arch}.hash_layer_count")
    if f is not None:
        hash_layers = int(f.contents())
    print(f"arch={arch} n_layer={n_layer} n_expert={n_expert} -> keep {K} "
          f"({1 - K / n_expert:.1%} cut) hash_layers={hash_layers}", file=sys.stderr)

    # ---- choose survivors per layer, hottest first, then sort for stable order ----
    keep_idx, old2new, fallback = {}, {}, {}
    for il in range(n_layer):
        order = np.argsort(-mass[il])                # descending usage
        kept = np.sort(order[:K])
        keep_idx[il] = kept
        m = np.full(n_expert, -1, dtype=np.int64)
        for new_i, old_i in enumerate(kept):
            m[old_i] = new_i
        old2new[il] = m
        # Dropped experts are spread across survivors by their own usage rank rather
        # than all collapsing onto one, which would overload a single expert.
        dropped = [e for e in order[K:]]
        fb = {}
        for rank, e in enumerate(dropped):
            fb[int(e)] = int(rank % K)
        fallback[il] = fb

    if args.dry_run:
        for il in (0, hash_layers, n_layer - 1):
            print(f"  layer {il}: keeping {keep_idx[il][:6].tolist()}… "
                  f"dropping {n_expert - K}", file=sys.stderr)
        return

    writer = GGUFWriter(args.output, arch=arch, endianess=readers[0].endianess)

    # ---- copy KV, overriding expert_count; drop split.* (single output file) ----
    for field in readers[0].fields.values():
        name = field.name
        if name.startswith("split."):
            continue
        if name == f"{arch}.expert_count":
            writer.add_uint32(name, K)
            print(f"  {name}: {n_expert} -> {K}", file=sys.stderr)
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), vt, sub_type=sub)

    # ---- pass 1: plan METADATA only ------------------------------------------------
    # GGUF requires every tensor-info record before any tensor data, so shapes must be
    # known up front. They are derivable from the source shape without materialising
    # anything -- which is the whole point: an earlier version built the full output in
    # RAM (~96 GB) and would OOM any box smaller than the cloud one it was written on.
    plan, sliced = [], 0
    for ri, r in enumerate(readers):
        for ti, t in enumerate(r.tensors):
            il = layer_of(t.name)
            short = t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name
            action = "copy"
            shape, dtype = t.data.shape, t.data.dtype
            if il is not None and il < n_layer:
                if short in SLICE_ON_EXPERT_AXIS:
                    # numpy view is reversed vs ggml, so axis 0 IS the expert axis
                    if shape[0] != n_expert:
                        sys.exit(f"{t.name}: expected expert axis {n_expert}, "
                                 f"got {shape[0]} -- refusing to slice")
                    action, shape = "slice", (K,) + shape[1:]
                    sliced += 1
                elif short in REMAP_VALUES:
                    action, dtype = "remap", np.dtype(np.int32)
            nbytes = int(np.prod(shape)) * dtype.itemsize
            plan.append((t.name, short, ri, ti, il, action, shape, dtype,
                         nbytes, t.tensor_type))

    # A tensor still carrying the ORIGINAL expert count is one this script failed to
    # recognise. Catch it here, not as a load-time shape mismatch.
    for name, short, _, _, _, action, shape, _, _, _ in plan:
        if short in NEVER_SLICE or action != "copy":
            continue
        if len(shape) and shape[0] == n_expert:
            sys.exit(f"UNSLICED: {name} still has {n_expert} on its leading axis. "
                     f"Add it to SLICE_ON_EXPERT_AXIS or NEVER_SLICE.")
    expected = (n_layer - hash_layers) + n_layer * 4
    print(f"  slicing {sliced} tensors on the expert axis (expected ~{expected})",
          file=sys.stderr)

    for name, _, _, _, _, _, shape, dtype, nbytes, ttype in plan:
        writer.add_tensor_info(name, shape, dtype, nbytes, ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # ---- pass 2: materialise ONE tensor at a time and write it straight out ---------
    remapped_rows, total, done = 0, 0, 0
    for name, short, ri, ti, il, action, shape, dtype, nbytes, _ in plan:
        t = readers[ri].tensors[ti]
        if action == "slice":
            data = t.data[keep_idx[il]]
        elif action == "remap":
            tbl = np.array(t.data, dtype=np.int64)               # [n_vocab, n_used]
            out = old2new[il][tbl]
            bad = out < 0
            if bad.any():
                remapped_rows += int(bad.sum())
                out[bad] = [fallback[il][int(v)] for v in tbl[bad]]
            data = out.astype(np.int32)
        else:
            data = t.data

        if data.shape != tuple(shape) or data.nbytes != nbytes:
            sys.exit(f"{name}: planned {tuple(shape)}/{nbytes}B but produced "
                     f"{data.shape}/{data.nbytes}B")
        writer.write_tensor_data(data, tensor_endianess=readers[0].endianess)
        total += data.nbytes
        del data                     # release before the next tensor is materialised
        done += 1
        if done % 200 == 0:
            print(f"    {done}/{len(plan)} tensors, {total/1e9:.1f} GB written",
                  file=sys.stderr)
    writer.close()

    print(f"\nwrote {args.output}", file=sys.stderr)
    print(f"  {total/1e9:.1f} GB, {len(plan)} tensors", file=sys.stderr)
    if hash_layers:
        print(f"  remapped {remapped_rows:,} hash-table entries that pointed at "
              f"dropped experts", file=sys.stderr)


if __name__ == "__main__":
    main()
