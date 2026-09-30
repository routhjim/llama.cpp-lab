#!/usr/bin/env python3
"""eval_chain.py DUMP [GGUF] [DEPTH] -- per-depth draft quality of an MTP head on a dump, assistant spans only.
Depth k (1-based) predicts token t+k+1 from the head's own chained streams; scored against the target's top-K
at t+k. Chains are teacher-forced on the real tokens (= conditional on earlier drafts being accepted)."""
import os, sys, time, torch, numpy as np
from readdump import read
from mtp_torch import load_gguf, MTPHead
from eval_head import assistant_mask  # noqa  (module-level eval code guarded below)

def main():
    dump = sys.argv[1]
    gguf = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.environ.get("MTP_Q8_GGUF", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
    D = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    WIN = 2048
    w = load_gguf(gguf)
    pt = sys.argv[4] if len(sys.argv) > 4 else None
    if pt:   # trained tensors override the gguf ones, cast to the dtype the gguf load used
        for k, v in torch.load(pt).items():
            assert k in w and w[k].shape == v.shape, k
            w[k] = v.to(w[k].device, w[k].dtype)
    head = MTPHead(w).eval()
    gguf = gguf + (f" + {pt.split('/')[-1]}" if pt else "")
    top1 = np.zeros(D); acc = np.zeros(D); n = 0
    with torch.no_grad():
        for tok, seed, tid, tlp in read(dump):
            am = assistant_mask(tok)
            L = len(tok) - D - 1
            for s in range(0, L, WIN):
                e = min(s + WIN, L)
                # chain valid only if tokens t+1 .. t+D+1 are all generated (inside an assistant turn)
                valid = np.ones(e - s, bool)
                for k in range(1, D + 2): valid &= am[s + k:e + k]
                if not valid.any(): continue
                h = torch.from_numpy(seed[s:e].astype(np.float32)).cuda().to(torch.bfloat16)
                toks = torch.from_numpy(np.stack([tok[s + k:e + k] for k in range(1, D + 1)], 1).astype(np.int64)).cuda()
                pos = torch.arange(s, e, device="cuda")
                outs = head.chain(h, toks, pos, D)
                vm = torch.from_numpy(valid).cuda()
                for k, lg in enumerate(outs):
                    ids = torch.from_numpy(tid[s + k + 1:e + k + 1].astype(np.int64)).cuda()
                    pt = torch.from_numpy(tlp[s + k + 1:e + k + 1].astype(np.float32)).cuda().exp()
                    q = lg.float().softmax(-1)
                    top1[k] += (lg.argmax(-1) == ids[:, 0])[vm].float().sum().item()
                    acc[k] += torch.minimum(pt, q.gather(1, ids)).sum(1)[vm].sum().item()
                n += int(valid.sum())
    t1, a = top1 / n, acc / n
    exp_len = sum(np.prod(a[:k + 1]) for k in range(D))
    print(f"FINAL {gguf.split('/')[-1]} n={n} | " + " ".join(f"d{k+1}: top1 {t1[k]:.4f} acc {a[k]:.4f}" for k in range(D))
          + f" | expected accepted/round {exp_len:.3f}")

if __name__ == "__main__":
    main()
