import numpy as np, struct
def read(path):
    """yield (tokens[n], seed[n,width] f16, topk_id[n,K], topk_logp[n,K] f16) per document"""
    with open(path, "rb") as f:
        assert f.read(4) == b"MTPD"; ver, width, K = struct.unpack("<3I", f.read(12))
        while True:
            h = f.read(4)
            if len(h) < 4: return
            n, = struct.unpack("<I", h)
            tok = np.frombuffer(f.read(4*n), np.int32)
            seed = np.frombuffer(f.read(2*n*width), np.float16).reshape(n, width)
            tid = np.frombuffer(f.read(4*n*K), np.int32).reshape(n, K)
            tlp = np.frombuffer(f.read(2*n*K), np.float16).reshape(n, K)
            yield tok, seed, tid, tlp
