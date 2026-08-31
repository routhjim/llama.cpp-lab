#!/usr/bin/env python3
"""gguf_shards.py — read a sharded GGUF as one logical tensor map, no merge step.

fn_split.py requires a merged single-file GGUF. Merging the Q8_0 donor would
cost 176 GiB of intermediate writes and does not fit anywhere on this box
(/home has 167 GiB free, the internal partition 199 GiB but it must also hold
the output). This reads the shards in place instead.

Also usable standalone to compare two donors for tensor congruence:
  gguf_shards.py --compare 'A/*.gguf' 'B/*.gguf' [--counts counts.npy]
"""
import argparse, glob, os, struct, sys
from fn_split import (ALIGN, GGUF_MAGIC, T_STR, T_ARR, rs, skip_value,
                      short, layer_of, SLICE_SUFFIX)

SPLIT_KEYS = ('split.no', 'split.count', 'split.tensors.count')


class Shard:
    def __init__(self, path):
        self.path = path
        self.f = f = open(path, 'rb')
        self.fsize = os.fstat(f.fileno()).st_size
        magic, self.ver, n_tensors, n_kv = struct.unpack('<IIQQ', f.read(24))
        assert magic == GGUF_MAGIC, f"{path}: bad magic {hex(magic)}"
        self.kvs = []                      # (key, start, end) raw byte spans
        for _ in range(n_kv):
            k = rs(f)
            start = f.tell() - (8 + len(k.encode()))
            t, = struct.unpack('<I', f.read(4))
            skip_value(f, t)
            self.kvs.append((k, start, f.tell()))
        self.tinfo = []
        for _ in range(n_tensors):
            nm = rs(f)
            nd, = struct.unpack('<I', f.read(4))
            dims = list(struct.unpack('<' + 'Q' * nd, f.read(8 * nd)))
            ty, off = struct.unpack('<IQ', f.read(12))
            self.tinfo.append([nm, dims, ty, off])
        self.data_start = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
        # tensor byte length = gap to the next tensor by offset (same as fn_split)
        by_off = sorted(range(len(self.tinfo)), key=lambda i: self.tinfo[i][3])
        self.nbytes = {}
        for j, i in enumerate(by_off):
            end = (self.tinfo[by_off[j + 1]][3] if j + 1 < len(by_off)
                   else self.fsize - self.data_start)
            self.nbytes[i] = end - self.tinfo[i][3]


class Donor:
    """All shards of one quant, presented as a single name -> tensor map."""

    def __init__(self, pattern, label=""):
        paths = sorted(glob.glob(pattern)) if not isinstance(pattern, list) else sorted(pattern)
        if not paths:
            raise SystemExit(f"no shards matched: {pattern}")
        self.label = label or os.path.basename(os.path.dirname(paths[0]))
        self.shards = [Shard(p) for p in paths]
        self.t = {}                        # name -> (shard, dims, ty, abs_off, nbytes)
        for sh in self.shards:
            for i, (nm, dims, ty, off) in enumerate(sh.tinfo):
                if nm in self.t:
                    raise SystemExit(f"{self.label}: duplicate tensor {nm}")
                self.t[nm] = (sh, dims, ty, sh.data_start + off, sh.nbytes[i])
        self.ver = self.shards[0].ver
        # KVs from shard 0, minus the split bookkeeping
        self.kvs = [(k, s, e) for (k, s, e) in self.shards[0].kvs
                    if k not in SPLIT_KEYS]
        self.kv_src = self.shards[0].f

    def read(self, name, off, n):
        sh, _, _, base, _ = self.t[name]
        sh.f.seek(base + off)
        out = bytearray()
        while n:
            c = sh.f.read(n)
            if not c:
                raise EOFError(f"{self.label}:{name} short read")
            out += c; n -= len(c)
        return bytes(out)

    def copy_into(self, out, name, off, n, chunk=1 << 26):
        sh, _, _, base, _ = self.t[name]
        sh.f.seek(base + off)
        while n:
            c = sh.f.read(min(chunk, n))
            if not c:
                raise EOFError(f"{self.label}:{name} short read")
            out.write(c); n -= len(c)

    def total_bytes(self):
        return sum(v[4] for v in self.t.values())

    def expert_bytes(self):
        return sum(v[4] for nm, v in self.t.items() if short(nm) in SLICE_SUFFIX)


def summarize(d, n_layer=None, n_expert=None):
    tot, exp = d.total_bytes(), d.expert_bytes()
    print(f"  {d.label:<12} shards={len(d.shards):<2} tensors={len(d.t):<6} "
          f"total={tot/2**30:7.1f} GiB  experts={exp/2**30:7.1f} GiB "
          f"({exp/tot*100:.1f}%)  non-expert={ (tot-exp)/2**30:6.1f} GiB")
    if n_layer and n_expert:
        slab = {}
        for nm, v in d.t.items():
            il = layer_of(nm)
            if il is not None and il < n_layer and short(nm) in SLICE_SUFFIX:
                assert v[4] % n_expert == 0, (nm, v[4])
                slab[il] = slab.get(il, 0) + v[4] // n_expert
        s = sorted(slab.values())
        print(f"  {'':<12} per-expert slab: min {s[0]/2**20:.2f} "
              f"med {s[len(s)//2]/2**20:.2f} max {s[-1]/2**20:.2f} MiB "
              f"over {len(s)} layers")
    return tot, exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', nargs=2, metavar=('HOT', 'COLD'), required=True)
    ap.add_argument('--counts')
    args = ap.parse_args()

    n_layer = n_expert = None
    if args.counts:
        import numpy as np
        n_layer, n_expert = np.load(args.counts).shape
        print(f"counts.npy: {n_layer} layers x {n_expert} experts\n")

    a = Donor(args.compare[0], "HOT")
    b = Donor(args.compare[1], "COLD")
    print("Donors:")
    summarize(a, n_layer, n_expert)
    summarize(b, n_layer, n_expert)

    print("\nCongruence:")
    na, nb = set(a.t), set(b.t)
    only_a, only_b = na - nb, nb - na
    print(f"  common tensors      : {len(na & nb)}")
    print(f"  only in HOT         : {len(only_a)}  {sorted(only_a)[:4]}")
    print(f"  only in COLD        : {len(only_b)}  {sorted(only_b)[:4]}")
    dim_mismatch, ty_same = [], 0
    for nm in sorted(na & nb):
        if a.t[nm][1] != b.t[nm][1]:
            dim_mismatch.append((nm, a.t[nm][1], b.t[nm][1]))
        if a.t[nm][2] == b.t[nm][2]:
            ty_same += 1
    print(f"  dim mismatches      : {len(dim_mismatch)}  {dim_mismatch[:3]}")
    print(f"  same quant type     : {ty_same} / {len(na & nb)}")
    if dim_mismatch or only_a or only_b:
        print("\n  *** NOT CONGRUENT — dual-donor slicing is unsafe ***")
        return 1
    print("\n  OK: identical tensor names and shapes; safe to source slabs from either.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
