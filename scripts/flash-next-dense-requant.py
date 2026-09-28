#!/usr/bin/env python3
"""Build a Flash-Next dense-path requant variant from UD-Q4_K_XL (see docs/flash-next-dense-requant.md).

Usage: flash-next-dense-requant.py VARIANT SRC_FIRST_SHARD IMATRIX OUT_PREFIX [--quantize PATH]
  VARIANT          C1, C2, T8, C3, C2T8 or C23T8
  SRC_FIRST_SHARD  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-0000N.gguf
  IMATRIX          unsloth's imatrix for the same model
  OUT_PREFIX       output path without the -0000N-of-0000N suffix (splits are kept)
Tensors not overridden keep their source type (q8_0 for the dense path in UD-Q4_K_XL).
"""
import argparse, subprocess, sys

BIG = ["attn_qkv", "attn_q", "attn_gate", "ssm_out"]  # the dense tensors that dominate a decode step
N_BLOCKS = 48
C1 = {g: "q4_K" for g in BIG}
C3 = {g: "q5_K" for g in BIG}
TAIL = set(range(40, 48))  # "T8": the last 8 blocks keep every big dense tensor at q8_0

# variant -> (type per tensor group, None = keep source type; blocks exempted from the overrides)
VARIANTS = {
    "C1":    (C1, set()),
    "C2":    ({**C1, "ssm_out": None}, set()),
    "T8":    (C1, TAIL),
    "C3":    (C3, set()),
    "C2T8":  ({**C1, "ssm_out": None}, TAIL),
    "C23T8": ({**C3, "ssm_out": None}, TAIL),
}

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=VARIANTS)
ap.add_argument("src"); ap.add_argument("imatrix"); ap.add_argument("out_prefix")
ap.add_argument("--quantize", default="llama-quantize")
a = ap.parse_args()

spec, exempt = VARIANTS[a.variant]
args = []
for il in range(N_BLOCKS):
    if il in exempt:
        continue
    for g in BIG:
        if spec.get(g):
            args += ["--tensor-type", rf"^blk\.{il}\.{g}\.weight={spec[g]}"]
args += ["--tensor-type", r"^output\.weight=q6_K"]

cmd = [a.quantize, "--allow-requantize", "--keep-split", "--imatrix", a.imatrix, *args,
       a.src, f"{a.out_prefix}.gguf", "COPY", "8"]
print(f"{a.variant}: {len(args) // 2} tensor overrides", flush=True)
sys.exit(subprocess.call(cmd))
