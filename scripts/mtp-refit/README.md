# mtp-refit

Re-fit the Qwen3.8-Flash-Next MTP draft head to the target it serves. Write-up and results:
[docs/flash-next-mtp-refit.md](../../docs/flash-next-mtp-refit.md).

| file | role |
|---|---|
| `build_corpus.py` | render the model's own agent traffic through its chat template into a corpus |
| `dump.sh` | run `llama-mtp-dump` (tools/mtp-dump) over a corpus: MTP seed + target top-20 per position |
| `readdump.py` | reader for the dump format |
| `mtp_torch.py` | torch port of `graph_mtp` (src/models/qwen4exp.cpp), incl. chained draft steps |
| `eval_head.py` | depth-1 agreement of a head with the target |
| `eval_chain.py` | per-depth expected acceptance, assistant spans only (matches live server acceptance) |
| `train.py` | soft cross-entropy re-fit at depths 1..3 |
| `export_gguf.py` | write the trained tensors back into the served draft gguf |
