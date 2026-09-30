#!/usr/bin/env python3
"""eval_head.py DUMP [GGUF] [MAX_POS] -- depth-1 agreement of an MTP head with the target on a dump.
At position t the head sees (h_t, token t+1) and predicts token t+2; the target's own prediction for t+2
is its top-K at position t+1. Reports top-1 agreement and expected acceptance
sum_x min(p_target(x), p_draft(x)) over the target's top-K (a lower bound; top-20 holds ~all the mass)."""
import os, sys, time, torch, numpy as np
from readdump import read
from mtp_torch import load_gguf, MTPHead

IM_START, ASSISTANT, IM_END = 248045, 74455, 248046

def assistant_mask(tok):
    # True where the target is GENERATING: inside <|im_start|>assistant ... <|im_end|>
    m = np.zeros(len(tok), bool); inside = False
    for i in range(len(tok)):
        if tok[i] == IM_START and i + 1 < len(tok) and tok[i + 1] == ASSISTANT: inside = True; continue
        if tok[i] == IM_END: inside = False
        m[i] = inside
    return m

if __name__ == "__main__":
    dump = sys.argv[1]
    gguf = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.environ.get("MTP_Q8_GGUF", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
    max_pos = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    WIN = 4096

    t0 = time.time(); head = MTPHead(load_gguf(gguf)).eval(); print(f"loaded {gguf} in {time.time()-t0:.0f}s", flush=True)
    agree = acc = n = 0
    agree_a = acc_a = n_a = 0
    with torch.no_grad():
        for tok, seed, tid, tlp in read(dump):
            am = assistant_mask(tok)
            for s in range(0, len(tok) - 2, WIN):
                e = min(s + WIN, len(tok) - 2)
                h = torch.from_numpy(seed[s:e].astype(np.float32)).cuda().to(torch.bfloat16)
                nxt = torch.from_numpy(tok[s + 1:e + 1].astype(np.int64)).cuda()
                pos = torch.arange(s, e, device="cuda")
                logits, _ = head(h, nxt, pos)
                q = logits.float().softmax(-1)
                ids = torch.from_numpy(tid[s + 1:e + 1].astype(np.int64)).cuda()          # target top-K for t+2
                pt = torch.from_numpy(tlp[s + 1:e + 1].astype(np.float32)).cuda().exp()
                hit = (logits.argmax(-1) == ids[:, 0]).float()
                ea = torch.minimum(pt, q.gather(1, ids)).sum(1)
                agree += hit.sum().item(); acc += ea.sum().item()
                # the drafted token t+2 must itself be generated, i.e. position t+2 inside an assistant turn
                ma = torch.from_numpy(am[s + 2:e + 2]).cuda()
                agree_a += hit[ma].sum().item(); acc_a += ea[ma].sum().item(); n_a += int(ma.sum().item())
                n += e - s
                if n >= max_pos: break
            print(f"  {n} positions: all top1 {agree/n:.4f} accept {acc/n:.4f} | assistant ({n_a}) top1 {agree_a/max(n_a,1):.4f} accept {acc_a/max(n_a,1):.4f}", flush=True)
            if n >= max_pos: break
    print(f"FINAL {gguf.split('/')[-1]}: all n={n} top1={agree/n:.4f} accept={acc/n:.4f} | assistant n={n_a} top1={agree_a/max(n_a,1):.4f} accept={acc_a/max(n_a,1):.4f}")
