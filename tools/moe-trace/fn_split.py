#!/usr/bin/env python3
"""fn_split.py — split qwen4exp expert tensors into hot/cold packs by routing mass.

Adapted from v4-tiered tier_split.py. Differences:
  * 2 tiers (hot / cold), NO pruning — all 512 experts survive
  * per-layer VARIABLE hot count from a global cold-byte budget (--cold-gib):
    all (layer, expert) slabs ranked by activation mass, coldest slabs go cold
  * cold-tier id table uses sentinel -1 for non-members (CPU mul_mat_id skips
    them; the cold pack must be placed on the CPU backend via -ot)
  * hot-tier non-members map to id 0 with mask 0 (Vulkan-safe dummy)
  * router (ffn_gate_inp) columns reordered to the new expert numbering

Input must be a MERGED single-file GGUF (llama-gguf-split --merge first).

Usage:
  fn_split.py --input merged.gguf --counts counts.npy --output split.gguf \
              [--cold-gib 18] [--dry-run]
"""
import argparse, os, struct, sys
import numpy as np

ALIGN = 32
CHUNK = 1 << 26  # 64 MiB

SLICE_SUFFIX = ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                "ffn_down_exps.weight")
REORDER_SUFFIX = ("ffn_gate_inp.weight",)

GGUF_MAGIC = 0x46554747
T_STR, T_ARR = 8, 9
GG_F32, GG_I32 = 0, 26


def rs(f):
    n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode()

def ws(s):
    b = s.encode(); return struct.pack('<Q', len(b)) + b

def skip_value(f, t):
    if t == T_STR: return rs(f)
    if t == T_ARR:
        et, n = struct.unpack('<IQ', f.read(12))
        if et in (0, 1): return f.read(n)
        for _ in range(n): skip_value(f, et)
        return None
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
    return f.read(sz)

def short(nm):
    return nm.split('.', 2)[-1] if nm.startswith('blk.') else nm

def layer_of(nm):
    try: return int(nm.split('.')[1]) if nm.startswith('blk.') else None
    except ValueError: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--counts', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--cold-gib', type=float, default=18.0)
    ap.add_argument('--min-hot', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    mass = np.load(args.counts).astype(np.float64)   # [n_layer, n_expert]
    n_layer, n_expert = mass.shape

    src = open(args.input, 'rb')
    fsize = os.fstat(src.fileno()).st_size
    magic, ver, n_tensors, n_kv = struct.unpack('<IIQQ', src.read(24))
    assert magic == GGUF_MAGIC, hex(magic)

    kvs = []
    for _ in range(n_kv):
        k = rs(src)
        start = src.tell() - (8 + len(k.encode()))
        t, = struct.unpack('<I', src.read(4))
        skip_value(src, t)
        kvs.append((k, start, src.tell()))

    tinfo = []
    for _ in range(n_tensors):
        nm = rs(src)
        nd, = struct.unpack('<I', src.read(4))
        dims = list(struct.unpack('<' + 'Q' * nd, src.read(8 * nd)))
        ty, off = struct.unpack('<IQ', src.read(12))
        tinfo.append([nm, dims, ty, off])
    hdr_end = src.tell()
    data_start = (hdr_end + ALIGN - 1) // ALIGN * ALIGN

    by_off = sorted(range(n_tensors), key=lambda i: tinfo[i][3])
    delta = {}
    for j, i in enumerate(by_off):
        end = tinfo[by_off[j+1]][3] if j + 1 < n_tensors else fsize - data_start
        delta[i] = end - tinfo[i][3]

    # sanity: every 3-D per-expert tensor must be sliced; note (but allow) 2-D
    # trailing-512 tensors that are not the router — n_embd_k_gqa == 512 makes
    # attn_k/attn_v collide with n_expert by coincidence on this arch
    for i, (nm, dims, ty, off) in enumerate(tinfo):
        if layer_of(nm) is None or layer_of(nm) >= n_layer or not dims:
            continue
        if len(dims) == 3 and dims[-1] == n_expert:
            assert short(nm) in SLICE_SUFFIX, f"unhandled 3-D expert tensor: {nm}"
        elif dims[-1] == n_expert and short(nm) not in REORDER_SUFFIX \
                and layer_of(nm) == 0 and 'attn' not in nm and 'index' not in nm:
            print(f"[fn_split] note: trailing-{n_expert} tensor left untouched: {nm}",
                  file=sys.stderr)

    # ---- bytes per expert-slab (uniform across layers by construction) ----
    slab_by_layer = np.zeros(n_layer)
    for i, (nm, dims, ty, off) in enumerate(tinfo):
        il = layer_of(nm)
        if il is not None and il < n_layer and short(nm) in SLICE_SUFFIX:
            assert delta[i] % n_expert == 0, (nm, delta[i])
            slab_by_layer[il] += delta[i] // n_expert
    assert (slab_by_layer > 0).all()
    # UD quants upcast some layers (Q8_0/Q5_K), so slab bytes vary per layer:
    # select cold slabs coldest-first by mass, accumulating actual bytes
    order_by_layer = [np.argsort(-mass[il], kind='stable') for il in range(n_layer)]
    lidx = np.repeat(np.arange(n_layer), n_expert)
    asc = np.argsort(mass.ravel(), kind='stable')      # coldest first
    budget = args.cold_gib * 2**30
    cold_ct = np.zeros(n_layer, np.int64)
    cold_bytes = cold_mass = 0.0
    for k in asc:
        il = int(lidx[k])
        if cold_ct[il] >= n_expert - args.min_hot:
            continue
        b = slab_by_layer[il]  # already per-expert bytes
        if cold_bytes + b > budget:
            break
        cold_bytes += b; cold_mass += mass.ravel()[k]; cold_ct[il] += 1
    hot_counts = n_expert - cold_ct
    print(f"[fn_split] slab {slab_by_layer.min()/2**20:.2f}-"
          f"{slab_by_layer.max()/2**20:.2f} MiB; cold "
          f"{int(cold_ct.sum())} slabs = {cold_bytes/2**30:.1f} GiB "
          f"({cold_bytes/(slab_by_layer*n_expert).sum()*100:.1f}% of expert bytes); "
          f"expected miss rate {cold_mass/mass.sum()*100:.2f}%; "
          f"hot/layer min {hot_counts.min()} med {int(np.median(hot_counts))} "
          f"max {hot_counts.max()}", file=sys.stderr)

    # ---- per-layer tables ----
    tables = {}
    for il in range(n_layer):
        H = hot_counts[il]
        ids_hot   = np.zeros(n_expert, np.int32)
        mask_hot  = np.zeros(n_expert, np.float32)
        ids_cold  = np.full(n_expert, -1, np.int32)    # sentinel: CPU skips
        mask_cold = np.zeros(n_expert, np.float32)
        j = np.arange(n_expert)
        ids_hot[:H] = j[:H];        mask_hot[:H] = 1.0
        # non-member (zero-weight dummy) rows MUST be spread across hot experts,
        # not concentrated on id 0: Vulkan's batched mul_mat_id mis-executes when
        # one id dominates the ids tensor (~85% duplicates -> wrong values at
        # ubatch >= 512; wiki chunks 14/22 went PPL 4.97 vs 3.41)
        ids_hot[H:] = (j[H:] % H).astype(np.int32)
        ids_cold[H:] = j[H:] - H;   mask_cold[H:] = 1.0
        tables[il] = (ids_hot, mask_hot, ids_cold, mask_cold)

    # ---- output plan ----
    plan = []
    n_sliced = 0
    for i, (nm, dims, ty, off) in enumerate(tinfo):
        il, sh = layer_of(nm), short(nm)
        if il is not None and il < n_layer and sh in SLICE_SUFFIX:
            stride = delta[i] // n_expert
            base = nm[:-len('.weight')]
            H = hot_counts[il]
            order = order_by_layer[il]
            for tn_, sel in (('hot', order[:H]), ('cold', order[H:])):
                nd = dims[:]; nd[-1] = len(sel)
                plan.append((f"{base}_{tn_}.weight", nd, ty, 'slice', (i, stride, sel)))
            n_sliced += 1
        elif il is not None and il < n_layer and sh in REORDER_SUFFIX:
            stride = delta[i] // n_expert
            plan.append((nm, dims, ty, 'reorder', (i, stride, order_by_layer[il])))
        else:
            plan.append((nm, dims, ty, 'copy', (i,)))
    for il in range(n_layer):
        ih, mh, ic, mc = tables[il]
        plan.append((f"blk.{il}.ffn_exp_tier_ids_hot",   [n_expert], GG_I32, 'blob', ih.tobytes()))
        plan.append((f"blk.{il}.ffn_exp_tier_mask_hot",  [n_expert], GG_F32, 'blob', mh.tobytes()))
        plan.append((f"blk.{il}.ffn_exp_tier_ids_cold",  [n_expert], GG_I32, 'blob', ic.tobytes()))
        plan.append((f"blk.{il}.ffn_exp_tier_mask_cold", [n_expert], GG_F32, 'blob', mc.tobytes()))

    print(f"[fn_split] {n_sliced} expert tensors -> hot/cold; "
          f"{len(plan)} output tensors", file=sys.stderr)
    if args.dry_run:
        return

    out = open(args.output, 'wb')
    out.write(struct.pack('<IIQQ', GGUF_MAGIC, ver, len(plan), n_kv))
    for k, s, e in kvs:
        src.seek(s); out.write(src.read(e - s))

    off_cur, infos = 0, []
    for nm, dims, ty, action, payload in plan:
        if action == 'copy':
            sz = delta[payload[0]]
        elif action in ('slice', 'reorder'):
            _, stride, sel = payload; sz = stride * len(sel)
        else:
            sz = len(payload)
        infos.append((nm, dims, ty, off_cur, sz))
        off_cur += (sz + ALIGN - 1) // ALIGN * ALIGN
    for nm, dims, ty, off, sz in infos:
        out.write(ws(nm) + struct.pack('<I', len(dims)))
        out.write(struct.pack('<' + 'Q' * len(dims), *dims))
        out.write(struct.pack('<IQ', ty, off))
    out.write(b'\0' * ((-out.tell()) % ALIGN))

    def copy_range(pos, n):
        src.seek(pos)
        while n:
            c = src.read(min(CHUNK, n)); out.write(c); n -= len(c)

    done = 0
    for (nm, dims, ty, action, payload), (_, _, _, _, sz) in zip(plan, infos):
        if action == 'copy':
            i = payload[0]
            copy_range(data_start + tinfo[i][3], delta[i])
        elif action in ('slice', 'reorder'):
            i, stride, sel = payload
            base = data_start + tinfo[i][3]
            for e in sel:
                copy_range(base + int(e) * stride, stride)
        else:
            out.write(payload)
        out.write(b'\0' * ((-out.tell()) % ALIGN))
        done += 1
        if done % 50 == 0:
            print(f"[fn_split] {done}/{len(plan)} tensors", file=sys.stderr, flush=True)

    out.close()
    print(f"[fn_split] wrote {args.output} "
          f"({os.path.getsize(args.output)/2**30:.1f} GiB)", file=sys.stderr)


if __name__ == '__main__':
    main()
