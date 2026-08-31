#!/usr/bin/env python3
"""Pre-warm the cold expert pack: buffered-read the _cold tensor byte ranges
so decode misses hit page cache instead of faulting from NVMe."""
import struct, sys, time
p = sys.argv[1] if len(sys.argv) > 1 else '/home/jrouth/models/fn-split/xl-split.gguf'
f = open(p, 'rb')
magic, ver, n_tensors, n_kv = struct.unpack('<IIQQ', f.read(24))
def rs(f):
    n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode()
T_STR, T_ARR = 8, 9
def skip_value(f, t):
    if t == T_STR: return rs(f)
    if t == T_ARR:
        et, n = struct.unpack('<IQ', f.read(12))
        if et in (0, 1): return f.read(n)
        for _ in range(n): skip_value(f, et)
        return None
    f.read({0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t])
for _ in range(n_kv):
    rs(f); t, = struct.unpack('<I', f.read(4)); skip_value(f, t)
ranges = []
tinfo = []
for _ in range(n_tensors):
    nm = rs(f); nd, = struct.unpack('<I', f.read(4))
    f.read(8*nd); ty, off = struct.unpack('<IQ', f.read(12))
    tinfo.append((nm, off))
hdr_end = f.tell()
data_start = (hdr_end + 31) // 32 * 32
offs = sorted(o for _, o in tinfo)
import bisect, os
fsize = os.path.getsize(p)
for nm, off in tinfo:
    if '_exps_cold' in nm:
        j = bisect.bisect_right(offs, off)
        end = offs[j] if j < len(offs) else fsize - data_start
        ranges.append((data_start + off, end - off))
tot = sum(n for _, n in ranges)
t0 = time.monotonic()
done = 0
for pos, n in ranges:
    f.seek(pos)
    while n > 0:
        c = f.read(min(1 << 25, n))
        if not c: break
        done += len(c); n -= len(c)
el = time.monotonic() - t0
print(f"warmed {done/2**30:.1f} GiB of cold pack in {el:.1f}s ({done/2**30/el:.2f} GiB/s)")
