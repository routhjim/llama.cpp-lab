#!/usr/bin/env bash
# run-qwen38-2x.sh: Qwen3.8-27B layer-split over two 7900 XTX as 2 pipelined llama-server instances (ports PORT, PORT+1)
# on ONE copy of the weights (LLAMA_SERVER_INSTANCES, #56), every slot reserving its own KV (no -kvu), MTP drafters on
# card 1 (card 2 holds the output head, so card 1 gets fewer layers), static draft depth 2, plus pipe-router.py on
# ROUTER_PORT (one endpoint, each conversation pinned to one instance).
# Needs #54 (cross-device copy race) and wants #55 (host<->device rings: 115 -> 185 t/s at 2 x np4).
# A 7900 XTX without ReBAR (hot-plugged dock) needs GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1, set below.
# --no-spec-draft-backend-sampling: draft GPU sampling returned token id == n_vocab with drafters on card 1 (open bug).
# Required: MODEL=<target gguf> DRAFT=<mtp gguf> TMPL=<chat template>. Check the dock PCIe links first: see
# docs/egpu-dock-pcie-gen1.md (Gen1 links cost another 30%).
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
B=${BIN:-$HERE/build/bin/llama-server}
: "${MODEL:?set MODEL}" "${DRAFT:?set DRAFT}" "${TMPL:?set TMPL}"
export LD_LIBRARY_PATH=$(dirname "$B")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 LLAMA_SERVER_INSTANCES=${INSTANCES:-2}
DEVS=$("$B" --list-devices 2>/dev/null)
LIST=$(echo "$DEVS" | grep -E '^\s+Vulkan[0-9]+:.*7900 XTX' | sed -E 's/^\s+(Vulkan[0-9]+):.*/\1/' | paste -sd,)
[ "$(echo "$LIST" | tr ',' '\n' | grep -c .)" -ge 2 ] || { echo "run-qwen38-2x: need two 7900 XTX, found '$LIST'" >&2; exit 1; }
DD=${DRAFT_DEV:-${LIST%%,*}}
PORT=${PORT:-8080}; RPORT=${ROUTER_PORT:-8090}
BACKS=$(seq -s, $PORT $((PORT + LLAMA_SERVER_INSTANCES - 1)))
echo "run-qwen38-2x: devs $LIST split ${TS:-48,52} drafters on $DD instances $BACKS x np${NP:-4} ctx ${CTX:-524288}/instance router :$RPORT" >&2
python3 "$HERE/scripts/two-xtx/pipe-router.py" --port "$RPORT" --backends "$BACKS" &
ROUTER=$!
trap 'kill $ROUTER 2>/dev/null' EXIT
"$B" -m "$MODEL" -dev "$LIST" -sm layer -ts ${TS:-48,52} -ngl 99 -fa on \
  -ctk ${CTK:-q4_0} -ctv ${CTV:-q4_0} -ctkd q4_0 -ctvd q4_0 -np ${NP:-4} -c ${CTX:-524288} -ub ${UB:-256} -b ${BATCH:-2048} -fit off \
  -md "$DRAFT" -ngld 99 -devd "$DD" --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2} --spec-coupled --no-spec-draft-backend-sampling \
  --jinja --chat-template-file "$TMPL" --metrics --host 127.0.0.1 --port "$PORT" "$@"
