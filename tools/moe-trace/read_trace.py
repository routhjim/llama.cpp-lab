#!/usr/bin/env python3
"""Reader for moe-trace.bin — MoE routing traces for expert-pruning calibration.

Format
------
header:  b"MOETRACE" | u32 version | u32 n_layer | u32 n_expert
                     | u32 n_used  | u32 hc      | u32 hash_layer_count | u32 reserved

chunk (repeating):
    u32 0xC0A1E5CE | u32 n_tokens | i32 tokens[n_tokens]
    per layer:
        u8|u16 topk [n_used * n_tokens]   expert ids (u8 in v1, u16 in v2)
        f16  wnorm[n_used * n_tokens]     normalized gate weights
        f16  hcpre[hc     * n_tokens]     HC stream gates (omitted when hc == 0)

Usage
-----
    python3 read_trace.py moe-trace.bin --summary
    python3 read_trace.py moe-trace.bin --counts counts.npy
"""

import argparse
import struct
import sys

import numpy as np

CHUNK_MAGIC = 0xC0A1E5CE


class Trace:
    def __init__(self, path):
        self.f = open(path, "rb")
        magic = self.f.read(8)
        if magic != b"MOETRACE":
            raise ValueError(f"not a moe-trace file (magic={magic!r})")
        (self.version, self.n_layer, self.n_expert,
         self.n_used, self.hc, self.hash_layers, _) = struct.unpack("<7I", self.f.read(28))

    def chunks(self):
        """Yield (tokens, topk, wnorm, hcpre) per chunk.

        topk  : uint8/uint16 (v1/v2) [n_layer, n_tokens, n_used]
        wnorm : float32 [n_layer, n_tokens, n_used]
        hcpre : float32 [n_layer, n_tokens, hc]  or None
        """
        while True:
            hdr = self.f.read(8)
            if len(hdr) < 8:
                return
            magic, n_tokens = struct.unpack("<2I", hdr)
            if magic != CHUNK_MAGIC:
                raise ValueError(f"chunk desync at offset {self.f.tell() - 8}")

            tokens = np.frombuffer(self.f.read(4 * n_tokens), dtype=np.int32)

            n_sel = self.n_used * n_tokens
            id_dt = np.uint8 if self.version < 2 else np.uint16
            topk = np.empty((self.n_layer, n_tokens, self.n_used), dtype=id_dt)
            wnorm = np.empty((self.n_layer, n_tokens, self.n_used), dtype=np.float32)
            hcpre = (np.empty((self.n_layer, n_tokens, self.hc), dtype=np.float32)
                     if self.hc else None)

            for il in range(self.n_layer):
                topk[il] = np.frombuffer(self.f.read(n_sel * id_dt().itemsize), dtype=id_dt).reshape(n_tokens, self.n_used)
                w = np.frombuffer(self.f.read(2 * n_sel), dtype=np.float16)
                wnorm[il] = w.astype(np.float32).reshape(n_tokens, self.n_used)
                if self.hc:
                    h = np.frombuffer(self.f.read(2 * self.hc * n_tokens), dtype=np.float16)
                    hcpre[il] = h.astype(np.float32).reshape(n_tokens, self.hc)

            yield tokens, topk, wnorm, hcpre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--summary", action="store_true", help="print routing summary")
    ap.add_argument("--counts", metavar="OUT.npy",
                    help="write [n_layer, n_expert] weighted activation mass")
    args = ap.parse_args()

    tr = Trace(args.path)
    print(f"version={tr.version} n_layer={tr.n_layer} n_expert={tr.n_expert} "
          f"n_used={tr.n_used} hc={tr.hc} hash_layers={tr.hash_layers}", file=sys.stderr)

    counts = np.zeros((tr.n_layer, tr.n_expert), dtype=np.float64)  # raw hit counts
    mass   = np.zeros((tr.n_layer, tr.n_expert), dtype=np.float64)  # gate-weighted
    n_tok  = 0

    for tokens, topk, wnorm, _hcpre in tr.chunks():
        n_tok += len(tokens)
        for il in range(tr.n_layer):
            ids = topk[il].ravel()
            ids = ids[ids != 0xFFFF]   # 0xFFFF = no valid expert id (dense layer, or an
                                   # id outside the uint16 encoding); not a real expert
        np.add.at(counts[il], ids, 1.0)
            np.add.at(mass[il], ids, wnorm[il].ravel().astype(np.float64))

    print(f"traced {n_tok} tokens", file=sys.stderr)

    if args.counts:
        np.save(args.counts, mass)
        print(f"wrote {args.counts}", file=sys.stderr)

    if args.summary:
        print(f"{'layer':>5} {'routing':>7} {'dead':>5} {'p50':>10} {'p10/p90':>9} {'gini':>6}")
        for il in range(tr.n_layer):
            m = mass[il]
            kind = "hash" if il < tr.hash_layers else "learned"
            dead = int((counts[il] == 0).sum())
            p10, p50, p90 = np.percentile(m, [10, 50, 90])
            ratio = p10 / p90 if p90 > 0 else 0.0
            s = np.sort(m)
            n = len(s)
            gini = ((2 * np.arange(1, n + 1) - n - 1) * s).sum() / (n * s.sum()) if s.sum() > 0 else 0.0
            print(f"{il:>5} {kind:>7} {dead:>5} {p50:>10.1f} {ratio:>9.3f} {gini:>6.3f}")

        print("\nA healthy learned layer has few dead experts and a moderate gini.",
              file=sys.stderr)
        print("gini near 0 => uniform usage, little to prune safely.", file=sys.stderr)
        print("gini near 1 => strongly skewed, the cold tail is a real prune target.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
