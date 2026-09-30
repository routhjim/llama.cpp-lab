// mtp-dump: record, for every token of a corpus, the input the Flash-Next (qwen4exp) MTP head
// seeds from and the target model's own next-token distribution. This is the training data for
// re-fitting the MTP head to a requantized target.
//
// Per position t it writes:
//   - token[t]
//   - the wide hyper-connection residual after the last trunk layer ("l_last-<n_layer-1>",
//     n_embd x hc floats, stored fp16) -- what graph_mtp consumes as h_nextn
//   - the top-K token ids and log-probs of the target's logits at t (its prediction of t+1)
//
// Uses only the public cb_eval hook, like moe-trace; no model or graph changes.
//
// Corpus: one text file, documents separated by a line containing exactly MTP_DOC_SEP. Each
// document is tokenized as-is (render chat templates beforehand), truncated to -c tokens, and fed
// in -b sized steps with the KV cache carried over, so the recorded states see the whole
// preceding document. The KV cache is cleared between documents.
//
// Output (-o, default mtp-dump.bin): a header, then one record per document:
//   header : magic "MTPD", u32 version=1, u32 width (n_embd*hc), u32 top_k
//   record : u32 n_tokens, i32 tokens[n], f16 seed[n][width], i32 topk_id[n][top_k], f16 topk_logp[n][top_k]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const char * MTP_DOC_SEP = "<|mtp-doc-sep|>";
static const int    TOP_K       = 20;

struct dump_ctx {
    std::string        seed_name;   // "l_last-47"
    std::vector<float> seed;        // accumulated over the ubatches of one llama_decode, [n][width]
    int64_t            width = 0;
};

static bool dump_cb(ggml_tensor * t, bool ask, void * user_data) {
    dump_ctx & dc = *(dump_ctx *) user_data;
    const bool match = dc.seed_name == t->name;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }
    // [n_embd, hc, n_tokens]; the last layer is uncropped because every position requests a logit
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const int64_t width = t->ne[0] * t->ne[1];
    const int64_t n     = t->ne[2];
    if (dc.width == 0) {
        dc.width = width;
    }
    GGML_ASSERT(dc.width == width);
    const size_t off = dc.seed.size();
    dc.seed.resize(off + (size_t) (width * n));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, dc.seed.data() + off, 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        for (int64_t k = 0; k < n; ++k) {
            for (int64_t j = 0; j < t->ne[1]; ++j) {
                for (int64_t i = 0; i < t->ne[0]; ++i) {
                    dc.seed[off + k*width + j*t->ne[0] + i] =
                        *(const float *) (raw.data() + i*t->nb[0] + j*t->nb[1] + k*t->nb[2]);
                }
            }
        }
    }
    return true;
}

static void write_u32(FILE * f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }

static void write_f16(FILE * f, const float * src, size_t n) {
    std::vector<ggml_fp16_t> buf(n);
    ggml_fp32_to_fp16_row(src, buf.data(), (int64_t) n);
    fwrite(buf.data(), sizeof(ggml_fp16_t), n, f);
}

static std::vector<std::string> split_docs(const std::string & text) {
    std::vector<std::string> docs;
    std::istringstream in(text);
    std::string line, cur;
    while (std::getline(in, line)) {
        if (line == MTP_DOC_SEP) {
            if (!cur.empty()) docs.push_back(cur);
            cur.clear();
        } else {
            cur += line;
            cur += '\n';
        }
    }
    if (!cur.empty()) docs.push_back(cur);
    return docs;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx   = 32768;
    params.n_batch = 1024;
    params.n_ubatch = 1024;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }
    common_init();

    if (params.prompt_file.empty() && params.prompt.empty()) {
        LOG_ERR("%s: pass the corpus with -f\n", __func__);
        return 1;
    }
    params.warmup = false;
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    dump_ctx dc;
    params.cb_eval           = dump_cb;
    params.cb_eval_user_data = &dc;

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const int           n_layer = llama_model_n_layer(model);
    dc.seed_name = "l_last-" + std::to_string(n_layer - 1);

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = std::min<int>(params.n_batch, n_ctx);

    const std::vector<std::string> docs = split_docs(params.prompt);
    LOG_INF("%s: %zu documents, n_ctx %d, step %d, capturing %s\n", __func__, docs.size(), n_ctx, n_batch, dc.seed_name.c_str());

    const std::string out_path = params.out_file.empty() ? std::string("mtp-dump.bin") : params.out_file;
    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) {
        LOG_ERR("%s: cannot open %s\n", __func__, out_path.c_str());
        return 1;
    }
    fwrite("MTPD", 1, 4, f);
    write_u32(f, 1);
    const long width_pos = ftell(f);
    write_u32(f, 0);                  // width, patched once the first capture tells us
    write_u32(f, TOP_K);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    std::vector<int32_t> top_id;
    std::vector<float>   top_lp;
    std::vector<int32_t> order(n_vocab);
    size_t total = 0;

    for (size_t d = 0; d < docs.size(); ++d) {
        std::vector<llama_token> toks = common_tokenize(ctx, docs[d], add_bos, true);
        if ((int) toks.size() > n_ctx) toks.resize(n_ctx);
        if (toks.size() < 2) continue;
        const int n = (int) toks.size();

        llama_memory_clear(llama_get_memory(ctx), true);
        dc.seed.clear();
        top_id.assign((size_t) n * TOP_K, 0);
        top_lp.assign((size_t) n * TOP_K, 0.0f);

        bool ok = true;
        for (int s = 0; s < n && ok; s += n_batch) {
            const int m = std::min(n_batch, n - s);
            common_batch_clear(batch);
            for (int i = 0; i < m; ++i) {
                common_batch_add(batch, toks[s + i], s + i, { 0 }, true);
            }
            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: decode failed in document %zu at %d\n", __func__, d, s);
                ok = false;
                break;
            }
            for (int i = 0; i < m; ++i) {
                const float * lg = llama_get_logits_ith(ctx, i);
                float mx = -INFINITY;
                for (int v = 0; v < n_vocab; ++v) mx = std::max(mx, lg[v]);
                double z = 0.0;
                for (int v = 0; v < n_vocab; ++v) z += std::exp((double) (lg[v] - mx));
                const float lz = mx + (float) std::log(z);
                for (int v = 0; v < n_vocab; ++v) order[v] = v;
                std::partial_sort(order.begin(), order.begin() + TOP_K, order.end(),
                                  [&](int a, int b) { return lg[a] > lg[b]; });
                for (int k = 0; k < TOP_K; ++k) {
                    top_id[(size_t) (s + i) * TOP_K + k] = order[k];
                    top_lp[(size_t) (s + i) * TOP_K + k] = lg[order[k]] - lz;
                }
            }
        }
        if (!ok) break;
        if (dc.seed.size() != (size_t) n * (size_t) dc.width) {
            LOG_ERR("%s: document %zu: captured %zu seed floats, expected %d x %lld -- is %s in the graph?\n",
                    __func__, d, dc.seed.size(), n, (long long) dc.width, dc.seed_name.c_str());
            break;
        }
        if (total == 0) {
            const long here = ftell(f);
            fseek(f, width_pos, SEEK_SET);
            write_u32(f, (uint32_t) dc.width);
            fseek(f, here, SEEK_SET);
        }
        write_u32(f, (uint32_t) n);
        fwrite(toks.data(), sizeof(int32_t), (size_t) n, f);
        write_f16(f, dc.seed.data(), dc.seed.size());
        fwrite(top_id.data(), sizeof(int32_t), top_id.size(), f);
        write_f16(f, top_lp.data(), top_lp.size());
        fflush(f);
        total += (size_t) n;
        LOG_INF("%s: document %zu/%zu: %d tokens (total %zu)\n", __func__, d + 1, docs.size(), n, total);
    }

    fclose(f);
    llama_batch_free(batch);
    llama_backend_free();
    LOG_INF("%s: wrote %zu positions to %s\n", __func__, total, out_path.c_str());
    return 0;
}
