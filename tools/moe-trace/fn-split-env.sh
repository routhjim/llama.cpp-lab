#!/usr/bin/env bash
# fn-split-env.sh -- optional convenience environment for split models from fn_split.py.
#
#   source tools/moe-trace/fn-split-env.sh
#
# NOTHING HERE IS REQUIRED. An earlier version of this file declared
# GGML_VK_OPT_LOOKAHEAD=3 mandatory and said a run without it was void. That was true
# against a Vulkan graph-optimizer bug which is now FIXED UPSTREAM by b387ddfd8, "vulkan:
# fix missing view-alias dependencies in ggml_vk_graph_optimize" (#27812). Measured on
# 25 wiki chunks after that fix:
#
#     loader forced the cap        PPL = 4.0255 +/- 0.03839
#     GGML_VK_OPT_LOOKAHEAD=20     PPL = 3.9939 +/- 0.03771
#
# No difference beyond the noise floor, so the cap and the loader's setenv that enforced it
# were both removed. Kept here only as a note, because "silently voids the run" is exactly
# the kind of rule that outlives its cause.
#
# WHAT IS ACTUALLY REQUIRED, and is a launch flag rather than an env var:
#
#     --no-op-offload      ggml_backend_sched offloads expert ops and inspects the ids
#                          tensor; the sentinel in the cold table trips it. Without this
#                          flag a split model aborts, or computes garbage if the abort is
#                          patched out.
#     -ot 'exps_cold=CPU'  the cold pack's id table uses GGML_MMID_SENTINEL, which only
#                          the CPU backend honours.

# Bounded LRU of mlock'd cold expert slabs the kernel cannot reclaim. 0/unset = off.
# Only matters under memory pressure; inert when the pack stays resident.
export GGML_EXPERT_CACHE_GIB=${GGML_EXPERT_CACHE_GIB:-0}

# Admit the top-K learned successors in addition to demand. Measured a wash on this
# workload (~44% precision, misses 38,947 with vs 39,348 without), so default off.
export GGML_EXPERT_CACHE_TRACE=${GGML_EXPERT_CACHE_TRACE:-0}
