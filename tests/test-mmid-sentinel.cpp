// test-mmid-sentinel.cpp: GGML_MMID_SENTINEL in GGML_OP_MUL_MAT_ID.
//
// A model whose experts are split across several weight tensors dispatches MUL_MAT_ID once
// per pack with the same id vector, marking ids owned by another pack with the sentinel.
// The CPU backend must leave those rows out of the multiply and write exact zeros.
//
// Zeroing is the part worth testing. Skipping the multiply is easy to get right; leaving the
// row untouched instead of zeroing it looks correct in any test that starts from a zeroed
// buffer, and then propagates whatever was there in a real graph where the buffer is reused.
// So the destination is deliberately poisoned with NaN before the compute: a row that is
// merely skipped stays NaN, and masking its weight to zero downstream would NOT rescue it,
// because 0 * NaN = NaN. Only real zeros pass.
//
// This is a CPU-only contract and so cannot live in test-backend-ops, which compares every
// backend against the CPU reference: other backends assert or read out of bounds on a
// sentinel id, and a case there would be a knowingly-failing test rather than a check.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int64_t K       = 32;  // reduction length
static const int64_t M       = 16;  // output width per expert
static const int64_t N_MATS  = 8;   // experts in this pack
static const int64_t N_USED  = 4;   // experts selected per token
static const int64_t N_TOK   = 6;

// Deterministic LCG mapped to [-1, 1]
static uint64_t g_rng = 0x9e3779b97f4a7c15ULL;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float) ((int32_t) (g_rng >> 33)) / (float) (1u << 30);
}

int main(void) {
    // Every (token, slot) is either a valid expert or the sentinel. Slot 0 of token 0 and
    // the whole of token 3 are sentinels, so the test covers a partially and a fully
    // masked token -- a fully masked token exercises the case where an expert pack
    // contributes nothing at all, which is what happens when a token routes entirely to
    // the other pack.
    std::vector<int32_t> ids(N_USED * N_TOK);
    for (int64_t t = 0; t < N_TOK; t++) {
        for (int64_t u = 0; u < N_USED; u++) {
            const bool sentinel = (t == 0 && u == 0) || (t == 3) || (t == 5 && u == N_USED - 1);
            ids[u + t*N_USED] = sentinel ? GGML_MMID_SENTINEL
                                         : (int32_t) ((t*N_USED + u) % N_MATS);
        }
    }

    const size_t mem_size = 64u*1024*1024;
    ggml_init_params ip = { mem_size, NULL, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, N_MATS);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N_USED, N_TOK);
    ggml_tensor * tid = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, N_TOK);

    float * as_d = (float *) as->data;
    float * b_d  = (float *) b->data;
    for (int64_t i = 0; i < ggml_nelements(as); i++) as_d[i] = frand();
    for (int64_t i = 0; i < ggml_nelements(b);  i++) b_d[i]  = frand();
    memcpy(tid->data, ids.data(), ids.size()*sizeof(int32_t));

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, tid);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // poison: a skipped-but-not-zeroed row stays NaN and fails below
    float * out_d = (float *) out->data;
    const int64_t n_out = ggml_nelements(out);
    for (int64_t i = 0; i < n_out; i++) out_d[i] = NAN;

    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    int n_sent = 0, n_real = 0, n_bad = 0;
    for (int64_t t = 0; t < N_TOK; t++) {
        for (int64_t u = 0; u < N_USED; u++) {
            const int32_t id = ids[u + t*N_USED];
            for (int64_t j = 0; j < M; j++) {
                const float got = out_d[j + u*M + t*M*N_USED];
                if (id == GGML_MMID_SENTINEL) {
                    n_sent++;
                    // exact zero, not "small" and not NaN
                    if (!(got == 0.0f)) {
                        if (n_bad < 8) {
                            fprintf(stderr, "FAIL sentinel t=%lld u=%lld j=%lld: %f%s\n",
                                    (long long) t, (long long) u, (long long) j, got,
                                    std::isnan(got) ? "  (row was skipped but never zeroed)" : "");
                        }
                        n_bad++;
                    }
                } else {
                    n_real++;
                    double ref = 0.0;
                    for (int64_t i = 0; i < K; i++) {
                        ref += (double) as_d[i + j*K + (int64_t) id*K*M] * (double) b_d[i + u*K + t*K*N_USED];
                    }
                    if (!(std::fabs(got - (float) ref) <= 1e-4f * (1.0f + std::fabs((float) ref)))) {
                        if (n_bad < 8) {
                            fprintf(stderr, "FAIL value t=%lld u=%lld j=%lld: got %f want %f\n",
                                    (long long) t, (long long) u, (long long) j, got, (float) ref);
                        }
                        n_bad++;
                    }
                }
            }
        }
    }

    ggml_free(ctx);

    if (n_bad) {
        printf("test-mmid-sentinel: FAILED (%d bad of %d)\n", n_bad, n_sent + n_real);
        return 1;
    }
    printf("test-mmid-sentinel: OK (%d sentinel zeros, %d computed values)\n", n_sent, n_real);
    return 0;
}
