#!/usr/bin/env python3
"""export_gguf.py TRAINED_PT OUT_GGUF [BASE_GGUF] -- write a draft GGUF: BASE's metadata and tensors, with the
trained tensors replaced and re-quantized to the type BASE stores them in. BASE (default $MTP_BASE_GGUF) is the
drafter you serve; its experts/embeddings stay untouched because they were frozen in training.
Adapted from gguf-py/gguf/scripts/gguf_new_metadata.py (copy_with_new_metadata)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf
from gguf.quants import quantize

trained_pt, out = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MTP_BASE_GGUF", "mtp-Qwen3.8-Flash-Next-Q4DRAFT.gguf")

trained = {k: v.float().numpy() for k, v in torch.load(trained_pt).items()}
reader = gguf.GGUFReader(base)
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = gguf.GGUFWriter(out, arch=arch, endianess=reader.endianess)

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), vt, sub_type=st)

data = {}
replaced = []
for t in reader.tensors:
    if t.name in trained:
        a = trained[t.name]
        want = tuple(int(x) for x in reversed(t.shape))
        assert a.size == int(np.prod(want)), (t.name, a.shape, want)
        a = a.reshape(want)
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            d = a.astype(np.float32)
        else:
            d = quantize(a.astype(np.float32), t.tensor_type)
        assert d.nbytes == t.data.nbytes, (t.name, d.nbytes, t.data.nbytes)
        replaced.append(t.name)
    else:
        d = t.data
    data[t.name] = d
    writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in reader.tensors:
    writer.write_tensor_data(data[t.name], tensor_endianess=reader.endianess)
writer.close()
missing = sorted(set(trained) - set(replaced))
print(f"wrote {out}: replaced {len(replaced)} tensors" + (f"; NOT in base: {missing}" if missing else ""))
