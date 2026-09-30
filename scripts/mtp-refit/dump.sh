#!/usr/bin/env bash
# dump.sh CORPUS OUT -- llama-mtp-dump over a corpus with the TARGET model (the one the drafter serves).
# MODEL = target gguf (first shard), DEV = its Vulkan device, BIN = build/bin. KV/load flags as served.
set -eu
B=${BIN:-$(dirname "$(realpath "$0")")/../../build/bin}
exec "$B/llama-mtp-dump" -m "${MODEL:?set MODEL to the target gguf}" -dev "${DEV:-Vulkan0}" -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 -lm mmap -lzm auto -c 16384 -b 1024 -ub 1024 -t 8 \
  -f "$(realpath "$1")" -o "$(realpath -m "$2")"
