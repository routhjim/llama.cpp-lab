"""Torch port of the Flash-Next (qwen4exp) MTP draft head, following graph_mtp in
src/models/qwen4exp.cpp (graph_mtp) line for line.

Input per position t: h_t (the trunk's wide hyper-connection residual, hc*n_embd = 4*2560) and the
token at t+1. Output: logits for the token at t+2, plus the head's own output streams (fed back as
h for chained drafts). The block's attention is causal over the positions of one sequence, like the
draft context's KV cache.

Weights load from a draft GGUF (converter transforms already applied), so the math matches
llama.cpp exactly: ggml ne [in, out] -> torch [out, in].
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize

N_EMBD, HC, N_HEAD, N_KV, HEAD = 2560, 4, 24, 2, 256
N_ROT, THETA, EPS = 64, 1e7, 1e-6
N_EXP, TOPK = 512, 10
L = "blk.48."


def load_gguf(path, dtype=torch.bfloat16, device="cuda"):
    r = GGUFReader(path)
    w = {}
    for t in r.tensors:
        a = dequantize(t.data, t.tensor_type)
        a = np.asarray(a, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
        keep_f32 = t.tensor_type.name == "F32"      # norms, routers: keep full precision
        if t.name.endswith("ffn_gate_inp_shexp.weight"):
            a = a.reshape(1, -1)                     # some files store it as [n_embd], others as [n_embd, 1]
        w[t.name] = torch.from_numpy(a).to(device, torch.float32 if keep_f32 else dtype)
    return w


def rms(x, w=None, dim_size=None):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def rope_neox(x, pos):
    # x [T, H, D]; rotate the first N_ROT dims NeoX-style (pairs i, i + N_ROT/2)
    half = N_ROT // 2
    inv = THETA ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = pos.float()[:, None] * inv[None, :]                      # [T, half]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    xr = x[..., :N_ROT].float()
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    return torch.cat([rot.to(x.dtype), x[..., N_ROT:]], -1)


def hc_mix(x, w_norm, w_down, w_up, w_inject):
    # x [T, HC, E] -> mixed [T, E], inject [T, HC] or None
    T = x.shape[0]
    xn = (rms(x).float() * w_norm.float().view(HC, N_EMBD)).to(x.dtype)   # per-stream RMS, [hc_dim] gamma
    xn = xn.reshape(T, HC * N_EMBD) / HC
    lo = F.silu(xn @ w_down.T)
    gate = torch.sigmoid(lo @ w_up.T)
    mixed = (xn * gate).view(T, HC, N_EMBD).sum(1)
    inject = (xn @ w_inject.T) if w_inject is not None else None
    return mixed, inject


def hc_combine(residual, block_out, inject):
    wgt = 2.0 * torch.sigmoid(inject.float()).to(residual.dtype)   # [T, HC]
    return residual + block_out[:, None, :] * wgt[:, :, None]


class MTPHead(torch.nn.Module):
    def __init__(self, w, trainable=()):
        super().__init__()
        self.p = torch.nn.ParameterDict()
        for k, v in w.items():
            key = k.replace(".", "__")
            self.p[key] = torch.nn.Parameter(v, requires_grad=any(s in k for s in trainable))

    def g(self, name):
        p = self.p[name.replace(".", "__")]
        # trainable weights are fp32 masters; matmul weights run in bf16 like the frozen ones, while
        # norm / router / gate vectors (F32 in the gguf) are consumed with .float() anyway
        if p.dtype == torch.float32 and p.dim() >= 2 and not name.endswith(("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")):
            return p.to(torch.bfloat16)
        return p

    def attn_qkv(self, x, pos):
        T = x.shape[0]
        q_full = (x @ self.g(L + "attn_q.weight").T).view(T, N_HEAD, 2 * HEAD)
        q, gate = q_full[..., :HEAD], q_full[..., HEAD:].reshape(T, N_HEAD * HEAD)
        q = rms(q, self.g(L + "attn_q_norm.weight"))
        k = rms((x @ self.g(L + "attn_k.weight").T).view(T, N_KV, HEAD), self.g(L + "attn_k_norm.weight"))
        v = (x @ self.g(L + "attn_v.weight").T).view(T, N_KV, HEAD)
        return rope_neox(q, pos), rope_neox(k, pos), v, gate

    def attn_chain(self, x, pos, kv1, kv_prev):
        """Draft step k>=2 at base position p: attends the step-1 (true-seed) entries 0..p and the entries
        this chain wrote at p in earlier steps, then itself -- the draft KV cache during one draft round."""
        T = x.shape[0]
        q, k, v, gate = self.attn_qkv(x, pos)
        K = torch.cat([kv1[0]] + [a for a, _ in kv_prev] + [k], 0)       # [(1+m)T, N_KV, D]
        V = torch.cat([kv1[1]] + [b for _, b in kv_prev] + [v], 0)
        m = len(kv_prev) + 1
        i = torch.arange(T, device=x.device)
        causal = i[None, :] <= i[:, None]                                 # step-1 block: j <= p
        diag = torch.eye(T, dtype=torch.bool, device=x.device)
        mask = torch.cat([causal] + [diag] * m, 1)                        # [T, (1+m)T]
        qh = q.transpose(0, 1)
        kh = K.transpose(0, 1).repeat_interleave(N_HEAD // N_KV, 0)
        vh = V.transpose(0, 1).repeat_interleave(N_HEAD // N_KV, 0)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None], scale=1.0 / math.sqrt(HEAD))
        o = o.transpose(0, 1).reshape(T, N_HEAD * HEAD) * torch.sigmoid(gate)
        return o @ self.g(L + "attn_output.weight").T, (k, v)

    def attn(self, x, pos, kv_extra=None):
        T = x.shape[0]
        q_full = (x @ self.g(L + "attn_q.weight").T).view(T, N_HEAD, 2 * HEAD)
        q, gate = q_full[..., :HEAD], q_full[..., HEAD:].reshape(T, N_HEAD * HEAD)
        q = rms(q, self.g(L + "attn_q_norm.weight"))
        k = rms((x @ self.g(L + "attn_k.weight").T).view(T, N_KV, HEAD), self.g(L + "attn_k_norm.weight"))
        v = (x @ self.g(L + "attn_v.weight").T).view(T, N_KV, HEAD)
        q, k = rope_neox(q, pos), rope_neox(k, pos)
        # GQA: [H, T, D]
        qh = q.transpose(0, 1)
        kh = k.transpose(0, 1).repeat_interleave(N_HEAD // N_KV, 0)
        vh = v.transpose(0, 1).repeat_interleave(N_HEAD // N_KV, 0)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=1.0 / math.sqrt(HEAD))
        o = o.transpose(0, 1).reshape(T, N_HEAD * HEAD) * torch.sigmoid(gate)
        return o @ self.g(L + "attn_output.weight").T

    def ffn(self, x):
        T = x.shape[0]
        logits = x.float() @ self.g(L + "ffn_gate_inp.weight").T.float()
        probs = logits.softmax(-1)
        wts, idx = probs.topk(TOPK, -1)
        wts = (wts / wts.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(x)
        G, U, D = self.g(L + "ffn_gate_exps.weight"), self.g(L + "ffn_up_exps.weight"), self.g(L + "ffn_down_exps.weight")
        flat = idx.reshape(-1)
        order = flat.argsort()
        counts = torch.bincount(flat, minlength=N_EXP).tolist()
        tok_of = (order // TOPK)
        start = 0
        for e, c in enumerate(counts):
            if c == 0: continue
            sel = order[start:start + c]; t = tok_of[start:start + c]; start += c
            xe = x[t]
            he = F.silu(xe @ G[e].T) * (xe @ U[e].T)
            ye = he @ D[e].T
            out.index_add_(0, t, ye * wts.reshape(-1)[sel][:, None])
        sh = F.silu(x @ self.g(L + "ffn_gate_shexp.weight").T) * (x @ self.g(L + "ffn_up_shexp.weight").T)
        sh = sh @ self.g(L + "ffn_down_shexp.weight").T
        sg = torch.sigmoid(x.float() @ self.g(L + "ffn_gate_inp_shexp.weight").T.float()).to(x.dtype)
        return out + sh * sg

    def step(self, h, next_tok, pos, kv1=None, kv_prev=()):
        """One draft step. kv1=None: step 1, causal over the window, returns its own (k, v) as kv1.
        Otherwise a chained step (see attn_chain). Returns logits, h_out, (k, v) of this step."""
        T = h.shape[0]
        h_norm = rms(h, self.g(L + "nextn.hnorm.weight")).view(T, HC, N_EMBD)
        e = F.embedding(next_tok, self.g("token_embd.weight"))
        e_norm = rms(e, self.g(L + "nextn.enorm.weight"))[:, None, :].expand(T, HC, N_EMBD)
        inpL = torch.cat([e_norm, h_norm], -1) @ self.g(L + "nextn.eh_proj.weight").T
        cur, inj = hc_mix(inpL, self.g(L + "hc_attn_norm.weight"), self.g(L + "hc_attn_down.weight"),
                          self.g(L + "hc_attn_up.weight"), self.g(L + "hc_attn_inject.weight"))
        if kv1 is None:
            q, k, v, gate = self.attn_qkv(cur, pos)
            qh = q.transpose(0, 1)
            kh = k.transpose(0, 1).repeat_interleave(N_HEAD // N_KV, 0)
            vh = v.transpose(0, 1).repeat_interleave(N_HEAD // N_KV, 0)
            o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=1.0 / math.sqrt(HEAD))
            cur = (o.transpose(0, 1).reshape(T, N_HEAD * HEAD) * torch.sigmoid(gate)) @ self.g(L + "attn_output.weight").T
            kv = (k, v)
        else:
            cur, kv = self.attn_chain(cur, pos, kv1, kv_prev)
        inpL = hc_combine(inpL, cur, inj)
        cur, inj = hc_mix(inpL, self.g(L + "hc_ffn_norm.weight"), self.g(L + "hc_ffn_down.weight"),
                          self.g(L + "hc_ffn_up.weight"), self.g(L + "hc_ffn_inject.weight"))
        cur = self.ffn(cur)
        inpL = hc_combine(inpL, cur, inj)
        head, _ = hc_mix(inpL, self.g("output_hc_norm.weight"), self.g("output_hc_down.weight"),
                         self.g("output_hc_up.weight"), None)
        return head @ self.g("output.weight").T, inpL.reshape(T, HC * N_EMBD), kv

    def chain(self, h, toks, pos, depth):
        """Unrolled draft: h [T, HC*E] true seeds, toks [T, depth] = tokens t+1..t+depth (teacher-forced
        on the target's real continuation, i.e. assuming earlier drafts were accepted).
        Returns a list of logits per depth; depth k predicts token t+k+1."""
        out = []
        logits, h_k, kv1 = self.step(h, toks[:, 0], pos)
        out.append(logits); kv_prev = []
        for k in range(1, depth):
            logits, h_k, kv = self.step(h_k, toks[:, k], pos + k, kv1=kv1, kv_prev=tuple(kv_prev))
            out.append(logits); kv_prev.append(kv)
        return out

    def forward(self, h, next_tok, pos):
        """h [T, HC*E], next_tok [T] (token at t+1), pos [T] -> logits [T, V], h_out [T, HC*E]"""
        T = h.shape[0]
        h_norm = rms(h, self.g(L + "nextn.hnorm.weight")).view(T, HC, N_EMBD)
        e = F.embedding(next_tok, self.g("token_embd.weight"))
        e_norm = rms(e, self.g(L + "nextn.enorm.weight"))[:, None, :].expand(T, HC, N_EMBD)
        inpL = torch.cat([e_norm, h_norm], -1) @ self.g(L + "nextn.eh_proj.weight").T          # [T, HC, E]
        cur, inj = hc_mix(inpL, self.g(L + "hc_attn_norm.weight"), self.g(L + "hc_attn_down.weight"),
                          self.g(L + "hc_attn_up.weight"), self.g(L + "hc_attn_inject.weight"))
        cur = self.attn(cur, pos)
        inpL = hc_combine(inpL, cur, inj)
        cur, inj = hc_mix(inpL, self.g(L + "hc_ffn_norm.weight"), self.g(L + "hc_ffn_down.weight"),
                          self.g(L + "hc_ffn_up.weight"), self.g(L + "hc_ffn_inject.weight"))
        cur = self.ffn(cur)
        inpL = hc_combine(inpL, cur, inj)
        head, _ = hc_mix(inpL, self.g("output_hc_norm.weight"), self.g("output_hc_down.weight"),
                         self.g("output_hc_up.weight"), None)
        logits = head @ self.g("output.weight").T
        return logits, inpL.reshape(T, HC * N_EMBD)
