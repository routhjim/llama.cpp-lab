#!/usr/bin/env python3
"""train.py TRAIN_DUMP OUT_PT [--lr 2e-5] [--epochs 1] [--depth 3] [--win 1024]
Re-fit the MTP head to the target it drafts for (C2T8): soft cross-entropy against the target's own top-K
distribution at draft depths 1..D, unrolled exactly like the draft KV cache (mtp_torch.MTPHead.chain),
counted only where the whole chain is generated text (assistant turns).
Trains attention, eh_proj, hyper-connection mixers, norms, router and shared expert; the 512 experts and the
token_embd / output tables stay frozen. Saves the trained tensors (fp32) to OUT_PT."""
import os, sys, time, math, argparse, torch, numpy as np
import torch.nn.functional as F
from readdump import read
from mtp_torch import load_gguf, MTPHead
from eval_head import assistant_mask

ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("out")
ap.add_argument("--gguf", default=os.environ.get("MTP_Q8_GGUF", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"))
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--depth", type=int, default=3); ap.add_argument("--win", type=int, default=1024)
ap.add_argument("--max-steps", type=int, default=0)
a = ap.parse_args()

FROZEN_SUB = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "indexer.")   # indexer: unused by the dense MTP path
FROZEN_EXACT = ("token_embd.weight", "output.weight")
w = load_gguf(a.gguf)
head = MTPHead(w)
train_keys = []
for key, p in head.p.items():
    name = key.replace("__", ".")
    if name in FROZEN_EXACT or any(f in name for f in FROZEN_SUB):
        p.requires_grad_(False)
    else:
        # fp32 master weights; MTPHead.g casts bf16 on use
        head.p[key] = torch.nn.Parameter(p.detach().float(), requires_grad=True)
        train_keys.append(key)
n_train = sum(head.p[k].numel() for k in train_keys)
print(f"trainable tensors {len(train_keys)}, params {n_train/1e6:.1f} M", flush=True)

# windows: (seed, toks, target ids/logp, valid mask) built lazily per document
def windows():
    D, W = a.depth, a.win
    for tok, seed, tid, tlp in read(a.dump):
        am = assistant_mask(tok)
        L = len(tok) - D - 1
        for s in range(0, L, W):
            e = min(s + W, L)
            valid = np.ones(e - s, bool)
            for k in range(1, D + 2): valid &= am[s + k:e + k]
            if valid.sum() < 16: continue
            yield s, e, tok, seed, tid, tlp, valid

n_win = sum(1 for _ in windows())
total = n_win * a.epochs if not a.max_steps else min(a.max_steps, n_win * a.epochs)
print(f"{n_win} windows/epoch, {total} steps", flush=True)
opt = torch.optim.AdamW([head.p[k] for k in train_keys], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))

step = 0; t0 = time.time(); run = []
for ep in range(a.epochs):
    for s, e, tok, seed, tid, tlp, valid in windows():
        h = torch.from_numpy(seed[s:e].astype(np.float32)).cuda().to(torch.bfloat16)
        toks = torch.from_numpy(np.stack([tok[s + k:e + k] for k in range(1, a.depth + 1)], 1).astype(np.int64)).cuda()
        pos = torch.arange(s, e, device="cuda")
        vm = torch.from_numpy(valid).cuda()
        outs = head.chain(h, toks, pos, a.depth)
        loss = 0.0; accs = []
        for k, lg in enumerate(outs):
            ids = torch.from_numpy(tid[s + k + 1:e + k + 1].astype(np.int64)).cuda()[vm]
            pt = torch.from_numpy(tlp[s + k + 1:e + k + 1].astype(np.float32)).cuda().exp()[vm]
            pt = pt / pt.sum(1, keepdim=True)
            lq = lg[vm].float().log_softmax(-1).gather(1, ids)
            loss = loss + -(pt * lq).sum(1).mean() / a.depth
            with torch.no_grad(): accs.append(torch.minimum(pt, lq.exp()).sum(1).mean().item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([head.p[k] for k in train_keys], 1.0)
        opt.step(); sched.step(); step += 1
        run.append([loss.item()] + accs)
        if step % 25 == 0 or step == 1:
            m = np.mean(run[-25:], 0)
            print(f"step {step}/{total} loss {m[0]:.4f} acc " + " ".join(f"{x:.3f}" for x in m[1:]) +
                  f" lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break

torch.save({k.replace("__", "."): head.p[k].detach().cpu() for k in train_keys}, a.out)
print(f"saved {len(train_keys)} tensors to {a.out}", flush=True)
