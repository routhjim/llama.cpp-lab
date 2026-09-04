// moe-trace-spec: record MoE expert routing from BOTH the target model and the MTP draft
// head, on the same tokens, during real speculative generation.
//
// This is examples/speculative-simple with a recorder bolted on -- the generation loop is
// copied verbatim rather than reimplemented, because the draft/verify/checkpoint dance is
// easy to get subtly wrong and a wrong loop would still produce a plausible-looking trace.
//
// WHY: MTP drafts n tokens verified in ONE forward pass whose cost is driven by the UNION of
// experts across those positions. Measured: the union grows 10, 17.2, 23.7, 29.6 for depth
// 1..4 -- sublinear, but the marginal never nears zero (p50 = 5 new experts for the 4th
// token, p25 = 3, p90 = 9). A dynamic-depth policy wants to extend only when it is small.
//
// The obstacle is timing: the target's routing for draft position k is computed INSIDE the
// verify pass you are deciding about. The one signal available early is the DRAFT head's own
// routing, since it runs first. This tool measures whether that predicts the target's.
//
// HOW: common_base_params_to_speculative copies common_params wholesale and
// common_context_params_to_llama forwards cb_eval, so one callback reaches BOTH contexts
// with no library change. They are told apart by layer index: the target emits
// ffn_moe_topk-0..n_layer-1; the MTP block is at n_layer.
//
// OUTPUT records, appended as produced:
//   u32 'MTSP' | u32 il | u32 n_tokens | u32 n_used | i32 ids[n_used*n_tokens]

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>


// ---------------------------------------------------------------- routing recorder
static const uint32_t MTSP_MAGIC = 0x5053544d; // 'MTSP'

struct moe_rec {
    FILE * fout        = nullptr;
    int    n_layer     = 0;   // il == n_layer is the MTP draft block
    size_t n_rec       = 0;
    size_t n_draft_rec = 0;
    std::vector<int32_t> buf;
};
static moe_rec g_rec;

static bool moe_read_i32(ggml_tensor * t, std::vector<int32_t> & dst) {
    if (t->type != GGML_TYPE_I32) return false;
    const size_t n = ggml_nelements(t);
    dst.resize(n);
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(int32_t));
        return true;
    }
    std::vector<uint8_t> tmp(ggml_nbytes(t));
    ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
    size_t k = 0;
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
        const uint8_t * row = (const uint8_t *) tmp.data() + i2*t->nb[2] + i1*t->nb[1];
        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) dst[k++] = *(const int32_t *)(row + i0*t->nb[0]);
    }
    return true;
}

static bool moe_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    static const char * PREFIX = "ffn_moe_topk-";
    if (strncmp(t->name, PREFIX, strlen(PREFIX)) != 0) return false;
    if (ask) return true;
    if (!g_rec.fout) return true;
    const int il = atoi(t->name + strlen(PREFIX));
    if (!moe_read_i32(t, g_rec.buf)) return true;
    const uint32_t n_used   = (uint32_t) t->ne[0];
    const uint32_t n_tokens = n_used ? (uint32_t)(g_rec.buf.size() / n_used) : 0;
    const uint32_t ilu      = (uint32_t) il;
    fwrite(&MTSP_MAGIC, 4, 1, g_rec.fout);
    fwrite(&ilu,        4, 1, g_rec.fout);
    fwrite(&n_tokens,   4, 1, g_rec.fout);
    fwrite(&n_used,     4, 1, g_rec.fout);
    fwrite(g_rec.buf.data(), sizeof(int32_t), g_rec.buf.size(), g_rec.fout);
    g_rec.n_rec++;
    if (il >= g_rec.n_layer) g_rec.n_draft_rec++;
    return true;
}
// -------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    // install the recorder BEFORE any context exists; common_base_params_to_speculative
    // copies params wholesale so this reaches the draft context too
    params.cb_eval           = moe_cb;
    params.cb_eval_user_data = nullptr;

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    g_rec.n_layer = llama_model_n_layer(model_tgt);
    // -o is not in the SPECULATIVE arg scope, so take the path from the environment
    const char * trace_path = getenv("MOE_SPEC_TRACE");
    if (!trace_path) trace_path = "moe-spec-trace.bin";
    g_rec.fout    = fopen(trace_path, "wb");
    if (!g_rec.fout) { LOG_ERR("%s: cannot open trace output '%s'\n", __func__, trace_path); return 1; }
    LOG_INF("%s: target n_layer=%d; the MTP block records as il=%d\n", __func__, g_rec.n_layer, g_rec.n_layer);


    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        llama_batch batch_prompt = llama_batch_init(inp.size(), 0, 1);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            common_batch_add(batch_prompt, inp[i], i, { seq_id }, false);
        }

        llama_decode(ctx_tgt, batch_prompt);

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    llama_tokens draft;

    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            break;
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    if (g_rec.fout) {
        fflush(g_rec.fout);
        fclose(g_rec.fout);
        g_rec.fout = nullptr;
        LOG_INF("%s: wrote %zu routing records, %zu of them from the draft head\n",
                __func__, g_rec.n_rec, g_rec.n_draft_rec);
        if (g_rec.n_draft_rec == 0) {
            LOG_ERR("%s: NO draft-head records -- the correlation cannot be computed from "
                    "this trace. Check that the MTP block emits ffn_moe_topk\n", __func__);
        }
    }

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
