#!/usr/bin/env bash
# fn-split-env.sh -- REQUIRED environment for the split-XL model. Source this before
# launching ANY llama.cpp binary against xl-split.gguf (or any fn_split.py output).
#
#   source ~/bin/fn-split-env.sh
#
# WHY THIS EXISTS (2026-08-30):
# The hot/cold split routing subgraph in models/qwen4exp.cpp is MIS-EXECUTED by a
# Vulkan graph optimization at ubatch >= ~512. It is silent: the model still emits
# fluent text, it just computes the prompt wrong. Measured at the campaign's own
# settings (-b 4096 -ub 2048, 25 wiki chunks):
#
#     no workaround (as the TB2.1 campaign ran):  PPL ~19-20
#     validated config (GGML_VK_OPT_LOOKAHEAD=3): PPL   3.997  == unsplit XL
#
# ~5x worse. The whole tb21best campaign (16 tasks, 8 passed) ran WITHOUT this and is
# void; "split-XL loses to IQ4_XS" was an artifact of a missing environment variable,
# not a property of the model.
#
# HOW IT WAS MISSED: the PPL harness (ppl_split3.sh) set GGML_VK_OPT_LOOKAHEAD=3, so
# every validation run was clean. The ANSWERER never set it. Validated config and
# production config were never the same. An audit on 2026-08-30 found only 3 of 21
# scripts that launch the split model had any workaround at all -- all 3 were PPL
# harnesses. Hence this single sourced file: one place to fix, impossible to half-apply.
#
# IMPACT IS BATCH-DEPENDENT. Decode runs at batch 1-3 and is clean, which is why greedy
# output matched byte-for-byte and nothing looked wrong. Only PREFILL is corrupted, so
# runs with tiny prompts (the spec sweeps, the cache A/Bs -- all <100-token prompts) are
# NOT affected. Runs with real contexts (any campaign) ARE.
#
# TWO MITIGATIONS (see BUGS.md):
#   GGML_VK_DISABLE_GRAPH_OPTIMIZE=1  turns the optimizer off entirely. Safest.
#   GGML_VK_OPT_LOOKAHEAD=3           caps the optimizer's reorder window (K<=3 clean,
#                                     K>=4 broken). Lighter touch, keeps most of the
#                                     optimizer. This is what the validated PPL used.
export GGML_VK_OPT_LOOKAHEAD=${GGML_VK_OPT_LOOKAHEAD:-3}

# madvise readahead for the mmap'd cold pack. Harmless when the pack is resident.
export GGML_MMID_MADVISE=${GGML_MMID_MADVISE:-1}

# Managed expert cache: bounded LRU set of mlock'd cold slabs the kernel cannot reclaim.
# 0/unset = off. Only matters under memory pressure; inert when the pack stays resident.
export GGML_EXPERT_CACHE_GIB=${GGML_EXPERT_CACHE_GIB:-0}
export GGML_EXPERT_CACHE_TRACE=${GGML_EXPERT_CACHE_TRACE:-0}

# ---------------------------------------------------------------------------
# CACHE / DRAFT-DEPTH COUPLING RULE  (see fn-expert-swap/FINDINGS.md section 6)
#
# A speculative verify batch reads the UNION of its tokens' experts. Measured on
# Flash-Next (tools/expert_union.py), per verify PASS:
#
#     depth  3 ->  2.8 GiB      depth 24 -> 11.3 GiB
#     depth  8 ->  5.5 GiB      depth 48 -> 16.4 GiB
#     depth 16 ->  8.8 GiB      depth 64 -> 18.9 GiB
#
# If ONE verify pass touches more than the cache holds, every pass flushes the cache and
# the hit rate collapses toward zero. So --spec-ngram-mod-n-max and
# GGML_EXPERT_CACHE_GIB are NOT independent knobs. Safe depth by cache size:
#
#     6 GiB -> 8      12 GiB -> 24      18 GiB -> 48      24 GiB -> 64
#
# This is exact and cheap to enforce: union(B) is a measured monotone function of batch
# LENGTH, which is known before committing (unlike a predictive cap -- expert routing is
# not known until the target runs).
#
# TODAY this is harmless: only ~6% of experts are cold, so little of the union reaches
# the cache. It becomes FATAL in an all-cold design, where a 64-token ngram draft touches
# 18.9 GiB and would flush a 12 GiB cache entirely -- and ~32% of ngram firings die at
# position 2, so a third of those flushes buy nothing.
#
# Exported for launchers to use as --spec-ngram-mod-n-max. NOT applied automatically:
# changing it mid-campaign would break comparability with runs already in flight.
# NOTE: the union table above was measured at XL (2.92 MiB/slab). It scales with
# SLAB SIZE, so it is quant-dependent -- at Q8 (4.98 MiB/slab) every union is 1.705x
# larger and the old hard-coded thresholds under-report the cache needed by that much.
# Pass the slab size, or set FN_SPLIT_SLAB_MIB.
#   slab MiB = <expert bytes> / (n_layer * n_expert);  XL 2.92, Q8 4.98
# The 0.95 factor reproduces the documented XL table exactly (6->8, 12->24, 18->48,
# 24->64). Worked example, Q8 @ 50 GiB: budget 47.5 GiB, depth-64 union 18.9*1.705
# = 32.2 GiB -> 64 is safe, and the minimum cache for depth 64 at Q8 is ~36 GiB.
#
# CAVEAT (2026-08-30, all-cold Q8): this rule bounds ONE verify pass. In an all-cold
# design the binding constraint is instead the steady-state working set ACROSS passes
# -- 119.5 GiB of experts against any cache we can build -- so eviction is normal and
# is NOT evidence of union thrash. Measured 21034 evictions at a 1.7% miss rate: a
# healthy LRU. Do not shrink ngram depth on eviction count alone; check the MISS rate.
fn_split_safe_depth() {
    local gib=${1:-${GGML_EXPERT_CACHE_GIB:-0}}
    local slab=${2:-${FN_SPLIT_SLAB_MIB:-2.92}}
    gib=${gib%.*}
    # 0 = managed cache DISABLED. The constraint does not apply: the kernel page cache
    # is the cache, and it is far larger than any verify pass's union.
    [ "${gib:-0}" -le 0 ] 2>/dev/null && { echo 64; return; }
    awk -v gib="$gib" -v slab="$slab" 'BEGIN{
        n=split("64:18.9 48:16.4 24:11.3 8:5.5 3:2.8", a, " ");
        budget = 0.95 * gib; scale = slab / 2.92;
        for (i=1; i<=n; i++) { split(a[i], p, ":");
            if (p[2]*scale <= budget) { print p[1]; exit } }
        print 1 }'
}
export FN_SPLIT_SAFE_DRAFT_DEPTH=$(fn_split_safe_depth)

# Reading the acceptance number: llama.cpp reports ONE blended accepted/drafted figure
# across all drafters in --spec-type. With ngram-mod + draft-mtp that number is close to
# useless -- ngram is <1% of events but ~23% of drafted tokens, so the headline mostly
# tracks HOW OFTEN NGRAM FIRED, not how well the drafter is doing. Measured live:
# MTP 76.7%, ngram 18.4%, blended 63.2%. Use tools/spec_stats.py for the real split.
