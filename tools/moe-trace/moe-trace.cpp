// moe-trace — capture MoE routing traces for expert-pruning calibration.
//
// Runs a corpus through the model (prefill only) and records, per token per layer:
//   * the full top-k expert coalition   (ffn_moe_topk-N)
//   * the normalized gate weights       (ffn_moe_weights_norm-N)
//   * the hyper-connection stream gates (hc_pre-N)  [deepseek4 only]
//
// Coalitions are captured rather than per-expert counters because coalition-aware
// attribution (Shapley-style over observed top-k sets) cannot be reconstructed from
// counts after the fact. Same for the HC stream gates: on deepseek4 the router input
// is a dynamic mixture of `hc` residual streams, so expert usage may be conditioned
// on stream regime. Pooling would hide it and there is no way to recover it later.
//
// This uses only the public cb_eval hook — no changes to model or graph code.
//
// GOTCHA (deepseek4): build_hc_pre() is called twice per layer, for attention and
// for the FFN, and both name their gate vector "hc_pre-N". We want the FFN one.
// It is always the LAST hc_pre-N to execute before ffn_moe_topk-N (the MoE depends
// on it), so we keep the most recent and bind it when topk arrives.
//
// NOTE (deepseek4): the first `hash_layer_count` layers do not use learned routing.
// They select experts by a fixed token-id -> expert table (ffn_gate_tid2eid), so
// their "frequency" is just the vocab distribution of the corpus. They are captured
// and flagged in the header; exclude them from frequency-based pruning.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static const char * MOETRACE_MAGIC = "MOETRACE";
static const uint32_t MOETRACE_VERSION    = 2;   // v2: expert ids are uint16
static const uint32_t MOETRACE_CHUNK_MAGIC = 0xC0A1E5CEu;

// ---------------------------------------------------------------------------

struct layer_stage {
    std::vector<int32_t> topk;   // [n_used * n_tokens]
    std::vector<float>   wnorm;  // [n_used * n_tokens]
    std::vector<float>   hcpre;  // [hc     * n_tokens]
    bool have_topk = false;
    bool have_w    = false;
    bool have_hc   = false;

    void clear() {
        topk.clear(); wnorm.clear(); hcpre.clear();
        have_topk = have_w = have_hc = false;
    }
};

struct trace_ctx {
    FILE * fout = nullptr;

    int n_layer  = 0;
    int n_used   = 0;   // n_expert_used
    int hc       = 0;   // hyper-connection stream count (0 if not deepseek4)

    std::vector<layer_stage>        stage;
    std::vector<std::vector<float>> hc_pending;  // most recent hc_pre-N, per layer

    std::vector<llama_token> cur_tokens;         // tokens of the window in flight
    size_t  tok_offset    = 0;                   // how far into the window we've flushed
    int64_t n_tok_written = 0;
    int64_t n_chunks      = 0;

    bool saw_any = false;   // did any target tensor fire at all?

    void reset_stage() {
        for (auto & s : stage) s.clear();
    }
};

// Parse "prefix-<il>". Returns il, or -1 if no match.
static int match_layer(const char * name, const char * prefix) {
    const size_t plen = strlen(prefix);
    if (strncmp(name, prefix, plen) != 0) return -1;
    if (name[plen] != '-') return -1;
    const char * p = name + plen + 1;
    if (*p < '0' || *p > '9') return -1;
    int il = 0;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9') return -1;
        il = il*10 + (*p - '0');
    }
    return il;
}

// These tensors are strided VIEWS, not contiguous buffers — hc_pre is a [hc, nt]
// window into the [(2+hc)*hc, nt] mixes tensor, and ffn_moe_topk is a view of the
// leading k columns of the argsort result. So ggml_nbytes() spans the parent's
// stride range and is much larger than nelements*type_size: the staging buffer must
// be sized by nbytes, and the logical elements gathered back out via nb[].
template <typename T>
static bool copy_strided(ggml_tensor * t, std::vector<T> & dst, ggml_type want) {
    if (t->type != want)            return false;
    if (t->ne[2] != 1 || t->ne[3] != 1) return false;

    static std::vector<uint8_t> raw;
    const size_t nb = ggml_nbytes(t);
    raw.resize(nb);
    ggml_backend_tensor_get(t, raw.data(), 0, nb);

    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    dst.resize((size_t) ne0 * ne1);

    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        const uint8_t * row = raw.data() + (size_t) i1 * t->nb[1];
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            memcpy(&dst[(size_t) i1*ne0 + i0], row + (size_t) i0 * t->nb[0], sizeof(T));
        }
    }
    return true;
}

static void flush_chunk(trace_ctx & tc);

static bool moe_trace_cb(ggml_tensor * t, bool ask, void * user_data) {
    trace_ctx & tc = *(trace_ctx *) user_data;
    const char * name = t->name;

    const int il_topk = match_layer(name, "ffn_moe_topk");
    const int il_w    = match_layer(name, "ffn_moe_weights_norm");
    const int il_hc   = match_layer(name, "hc_pre");

    if (ask) {
        return il_topk >= 0 || il_w >= 0 || il_hc >= 0;
    }

    tc.saw_any = true;

    // hc_pre fires twice per layer (attn, then ffn). Keep the latest; it gets bound
    // when this layer's topk arrives, which is strictly after the FFN one.
    if (il_hc >= 0 && il_hc < tc.n_layer) {
        copy_strided(t, tc.hc_pending[il_hc], GGML_TYPE_F32);
        return true;
    }

    if (il_w >= 0 && il_w < tc.n_layer) {
        tc.stage[il_w].have_w = copy_strided(t, tc.stage[il_w].wnorm, GGML_TYPE_F32);
        return true;
    }

    if (il_topk >= 0 && il_topk < tc.n_layer) {
        // Seeing a layer twice means a new ubatch began — flush the previous one.
        if (tc.stage[il_topk].have_topk) {
            flush_chunk(tc);
        }
        if (!copy_strided(t, tc.stage[il_topk].topk, GGML_TYPE_I32)) {
            return true;
        }
        tc.stage[il_topk].have_topk = true;

        if (!tc.hc_pending[il_topk].empty()) {
            tc.stage[il_topk].hcpre  = tc.hc_pending[il_topk];
            tc.stage[il_topk].have_hc = true;
        }
        return true;
    }

    return true;
}

// ---------------------------------------------------------------------------

static void write_u32(FILE * f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }

static void write_fp16(FILE * f, const std::vector<float> & src, size_t n) {
    static std::vector<ggml_fp16_t> buf;
    buf.resize(n);
    const size_t have = std::min(n, src.size());
    ggml_fp32_to_fp16_row(src.data(), buf.data(), (int64_t) have);
    for (size_t i = have; i < n; ++i) buf[i] = 0;
    fwrite(buf.data(), sizeof(ggml_fp16_t), n, f);
}

static void flush_chunk(trace_ctx & tc) {
    // token count implied by the staged tensors
    int n_tokens = 0;
    for (const auto & s : tc.stage) {
        if (s.have_topk) { n_tokens = (int) (s.topk.size() / tc.n_used); break; }
    }
    if (n_tokens == 0) { tc.reset_stage(); return; }

    write_u32(tc.fout, MOETRACE_CHUNK_MAGIC);
    write_u32(tc.fout, (uint32_t) n_tokens);

    // Token ids for this chunk. A decode may split into several ubatches, so read from
    // the running offset rather than the head of the window; -1 marks any token we
    // could not attribute (should not happen, but must not silently misalign).
    std::vector<int32_t> toks(n_tokens, -1);
    for (int i = 0; i < n_tokens; ++i) {
        const size_t idx = tc.tok_offset + i;
        if (idx < tc.cur_tokens.size()) toks[i] = tc.cur_tokens[idx];
    }
    tc.tok_offset += n_tokens;
    fwrite(toks.data(), sizeof(int32_t), n_tokens, tc.fout);

    std::vector<uint16_t> ids;
    for (int il = 0; il < tc.n_layer; ++il) {
        const auto & s = tc.stage[il];
        const size_t n_sel = (size_t) tc.n_used * n_tokens;

        // expert ids as uint16 (v2) — 512-expert models exceed the old uint8 encoding
        ids.assign(n_sel, 0);
        for (size_t i = 0; i < n_sel && i < s.topk.size(); ++i) {
            ids[i] = (uint16_t) s.topk[i];
        }
        fwrite(ids.data(), sizeof(uint16_t), n_sel, tc.fout);

        write_fp16(tc.fout, s.wnorm, n_sel);
        if (tc.hc > 0) {
            write_fp16(tc.fout, s.hcpre, (size_t) tc.hc * n_tokens);
        }
    }

    tc.n_tok_written += n_tokens;
    tc.n_chunks++;
    tc.reset_stage();
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    common_params params;

    // IMATRIX scope: same shape of tool (corpus in via -f, artifact out via -o).
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }
    common_init();

    if (params.prompt.empty()) {
        LOG_ERR("%s: no corpus — pass one with -f <corpus.txt>\n", __func__);
        return 1;
    }

    // A whole context window is submitted as one logical batch, so n_batch must be at
    // least n_ctx or llama_decode asserts (n_tokens_all <= cparams.n_batch). The
    // physical ubatch is left alone: the callback tracks a token offset, so a decode
    // that splits into several ubatches is handled correctly and activation memory
    // stays bounded.
    if (params.n_batch < (uint32_t) params.n_ctx) {
        params.n_batch = params.n_ctx;
    }
    params.warmup = false;

    trace_ctx tc;

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &tc;

    auto init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();

    // There are no public accessors for expert counts, so read everything
    // architecture-specific straight out of the GGUF metadata.
    std::string arch;
    {
        char buf[128];
        if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) >= 0) {
            arch = buf;
        }
    }
    auto meta_int = [&](const char * suffix, int fallback) {
        char key[192], val[64];
        snprintf(key, sizeof(key), "%s.%s", arch.c_str(), suffix);
        if (llama_model_meta_val_str(model, key, val, sizeof(val)) >= 0) {
            return atoi(val);
        }
        return fallback;
    };

    const int n_layer  = llama_model_n_layer(model);
    const int n_expert = meta_int("expert_count", 0);
    const int n_used   = meta_int("expert_used_count", 0);

    if (n_expert == 0 || n_used == 0) {
        LOG_ERR("%s: not an MoE model (n_expert=%d)\n", __func__, n_expert);
        return 1;
    }
    if (n_expert > 65536) {
        LOG_ERR("%s: n_expert=%d exceeds the uint16 id encoding in this format\n", __func__, n_expert);
        return 1;
    }

    // deepseek4-specific; absent on other MoE architectures, which is fine.
    const int hc          = meta_int("hyper_connection.count", 0);
    const int hash_layers = meta_int("hash_layer_count", 0);

    tc.n_layer = n_layer;
    tc.n_used  = n_used;
    tc.hc      = hc;
    tc.stage.resize(n_layer);
    tc.hc_pending.resize(n_layer);

    const std::string out_path = params.out_file.empty() ? std::string("moe-trace.bin") : params.out_file;
    tc.fout = fopen(out_path.c_str(), "wb");
    if (!tc.fout) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, out_path.c_str());
        return 1;
    }

    fwrite(MOETRACE_MAGIC, 1, 8, tc.fout);
    write_u32(tc.fout, MOETRACE_VERSION);
    write_u32(tc.fout, (uint32_t) n_layer);
    write_u32(tc.fout, (uint32_t) n_expert);
    write_u32(tc.fout, (uint32_t) n_used);
    write_u32(tc.fout, (uint32_t) hc);
    write_u32(tc.fout, (uint32_t) hash_layers);
    write_u32(tc.fout, 0);  // reserved

    LOG_INF("%s: n_layer=%d n_expert=%d n_used=%d hc=%d hash_layers=%d -> %s\n",
            __func__, n_layer, n_expert, n_used, hc, hash_layers, out_path.c_str());
    if (hash_layers > 0) {
        LOG_WRN("%s: layers 0..%d use hash routing (token-id table, not a learned "
                "router) — exclude them from frequency-based pruning\n", __func__, hash_layers - 1);
    }

    // -----------------------------------------------------------------------

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    std::vector<llama_token> corpus = common_tokenize(ctx, params.prompt, add_bos, true);

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_chunk = (int) (corpus.size() / n_ctx);

    LOG_INF("%s: corpus = %zu tokens, %d windows of %d\n", __func__, corpus.size(), n_chunk, n_ctx);
    if (n_chunk == 0) {
        LOG_ERR("%s: corpus shorter than one context window (%d)\n", __func__, n_ctx);
        return 1;
    }

    for (int i = 0; i < n_chunk; ++i) {
        std::vector<llama_token> window(corpus.begin() + (size_t) i*n_ctx,
                                        corpus.begin() + (size_t) (i+1)*n_ctx);
        if (add_bos) window[0] = llama_vocab_bos(llama_model_get_vocab(model));

        tc.cur_tokens = window;
        tc.tok_offset = 0;
        llama_memory_clear(llama_get_memory(ctx), true);

        if (llama_decode(ctx, llama_batch_get_one(window.data(), window.size()))) {
            LOG_ERR("%s: decode failed on window %d\n", __func__, i);
            return 1;
        }
        flush_chunk(tc);   // trailing ubatch

        if ((i+1) % 10 == 0 || i+1 == n_chunk) {
            LOG_INF("%s: %d/%d windows, %lld tokens traced\n",
                    __func__, i+1, n_chunk, (long long) tc.n_tok_written);
        }
    }

    fclose(tc.fout);

    if (!tc.saw_any) {
        LOG_ERR("%s: no routing tensors were observed — names may have changed, or graph "
                "fusion elided them. Check with llama-eval-callback first.\n", __func__);
        return 1;
    }

    LOG_INF("%s: done — %lld tokens in %lld chunks -> %s\n",
            __func__, (long long) tc.n_tok_written, (long long) tc.n_chunks, out_path.c_str());

    llama_backend_free();
    return 0;
}
