// Learner / prefetcher for the sparse expert residency manager. Included at the end of
// ggml-vk-sparse.inc; not a standalone translation unit.
//
// The compute thread only pushes events: (token, layer, selected experts, miss mask). This
// thread learns from them and prefetches experts for the next few layers of the SAME token
// while the GPU is still busy with the current ones, so that the exact stall-and-bind at
// each layer finds them resident (or in flight) instead of reading them on the critical path.
//
// Signals, all learned online from the running workload:
//   cross    P(expert y at layer b | expert x at layer a)   for a < b <= a + depth
//   tt       P(expert y at layer L for token t+1 | expert x at layer L for token t)
//   ema      per-layer trailing frequency of each expert (the live union profile)
//   library  snapshots of ema taken when the profile changed: a regime is matched by how
//            well this token's known layers agree with it, and if the match is strong its
//            frequencies are an extra source (opportunistic: only used when it fits now)
// Candidate score = noisy-OR of the source probabilities. The spend decision is not a fixed
// threshold: per (layer, score bucket) the learner tracks how often its prefetches were used
// and only spends when that measured precision is above p_min (with a little exploration).
//
// Prefetched slabs go into a PROBATION list (evicted first); a real hit promotes them to the
// LRU. Slabs that die unused are charged back to the bucket that issued them.
//
// Env: GGML_VK_SPARSE_PREFETCH=1 enables. GGML_VK_SPARSE_PF_DEPTH (6), _PF_PMIN (0.5),
//      _PF_PROB_GIB (4), _PF_BATCH_MIB (128), _PF_READERS (6), _PF_NMAX (12).

struct vk_sparse_learner {
    vk_sparse_mgr * M;
    std::thread th;
    std::atomic<bool> stop {false};
    std::mutex qmu;
    std::condition_variable qcv;
    std::deque<vk_sparse_ev> q;

    int    depth   = 8;
    int    next_tok_layers = 8;   // near the end of a token, prefetch the next token's first layers
    double p_min   = 0.5;
    size_t prob_cap = 4ull << 30;
    size_t batch_max = 32ull << 20;    // per layer event
    size_t tok_max   = 512ull << 20;   // per token
    size_t tok_bytes = 0;
    int    n_max   = 12;
    int    explore_n = 8;
    int    K       = 48;

    struct row {
        std::vector<std::pair<uint16_t, uint32_t>> v;
        uint32_t total = 0;
        void add(uint16_t y, int K) {
            total++;
            for (auto & e : v) if (e.first == y) { e.second++; return; }
            if ((int) v.size() < K) { v.push_back({ y, 1 }); return; }
            size_t w = 0;
            for (size_t i = 1; i < v.size(); i++) if (v[i].second < v[w].second) w = i;
            v[w] = { y, 1 };
        }
    };
    std::unordered_map<uint64_t, row> cross, tt;
    static uint64_t kc(int a, int x, int b) { return ((uint64_t) a << 40) | ((uint64_t) x << 20) | (uint64_t) b; }
    static uint64_t kt(int L, int x)        { return ((uint64_t) L << 20) | (uint64_t) x; }

    uint32_t cur_tok = UINT32_MAX;
    int n_layer = 0, n_expert = 0;
    std::vector<std::vector<uint16_t>> cur, prev;   // [layer] selected experts
    std::vector<std::vector<float>> ema;            // [layer][expert]
    float alpha = 1.0f / 32;
    struct profile { std::vector<std::vector<float>> f; uint64_t seen_tok; };
    std::vector<profile> lib;
    uint32_t tok_since_snap = 0;

    struct cal { uint32_t issued = 0, used = 0, dead = 0; };
    std::vector<std::array<cal, 10>> calib;         // [layer][bucket]

    std::map<int, std::vector<uint32_t>> by_layer;  // layer -> tensor indices (gate/up/down)
    size_t by_layer_n = 0;
    vk::Fence fence;

    uint64_t n_ev = 0, n_pf = 0, n_pf_used = 0, n_pf_dead = 0, pf_bytes = 0, n_batches = 0, n_lib_hits = 0, n_stale = 0;

    // batches in flight: the learner submits reads and moves on; the binder completes them in order
    struct pending { uint32_t ti, e; int layer, bucket; };
    struct batch {
        std::vector<vk_sparse_bind> binds;
        std::vector<pending> pend;
        std::atomic<uint32_t> remaining {0};
    };
    std::deque<std::unique_ptr<batch>> bq;
    std::mutex bmu;
    std::condition_variable bcv;
    std::thread binder;
    size_t inflight_bytes = 0;
    size_t inflight_max = 256ull << 20;

    explicit vk_sparse_learner(vk_sparse_mgr * m) : M(m) {
        auto env_d = [](const char * n, double def) { const char * v = getenv(n); return v ? atof(v) : def; };
        depth     = (int) env_d("GGML_VK_SPARSE_PF_DEPTH", 8);
        next_tok_layers = (int) env_d("GGML_VK_SPARSE_PF_NEXT", 8);
        p_min     = env_d("GGML_VK_SPARSE_PF_PMIN", 0.5);
        prob_cap  = (size_t) (env_d("GGML_VK_SPARSE_PF_PROB_GIB", 4.0) * 1073741824.0);
        batch_max = (size_t) (env_d("GGML_VK_SPARSE_PF_BATCH_MIB", 32) * 1048576.0);
        tok_max   = (size_t) (env_d("GGML_VK_SPARSE_PF_TOKEN_MIB", 512) * 1048576.0);
        explore_n = (int) env_d("GGML_VK_SPARSE_PF_EXPLORE", 8);
        M->prob_cap = prob_cap;
        n_max     = (int) env_d("GGML_VK_SPARSE_PF_NMAX", 12);
        fence     = M->dev->device.createFence({});
        inflight_max = (size_t) (env_d("GGML_VK_SPARSE_PF_INFLIGHT_MIB", 256) * 1048576.0);
        M->pf.start((int) env_d("GGML_VK_SPARSE_PF_READERS", 6));
        th = std::thread([this] { main(); });
        binder = std::thread([this] { binder_main(); });
    }

    ~vk_sparse_learner() {
        stop = true;
        qcv.notify_all();
        bcv.notify_all();
        if (th.joinable()) th.join();
        M->pf.stopall();   // wakes the binder out of wait_ctr
        if (binder.joinable()) binder.join();
        M->dev->device.destroyFence(fence);
    }

    void binder_main() {
        for (;;) {
            std::unique_ptr<batch> b;
            {
                std::unique_lock<std::mutex> lk(bmu);
                bcv.wait(lk, [this] { return stop || !bq.empty(); });
                if (bq.empty()) return;
                b = std::move(bq.front()); bq.pop_front();
            }
            M->pf.wait_ctr(b->remaining);
            if (stop) return;
            M->bind_apply(b->binds, fence);
            std::lock_guard<std::recursive_mutex> g(M->mu);
            size_t bytes = 0;
            for (auto & p : b->pend) {
                auto & T = *M->tensors[p.ti];
                auto & sl = T.slots[p.e];
                bytes += T.nb02;
                if (sl.state != VKS_LOADING) continue;
                sl.state = VKS_PROB;
                M->list_push(VK_SPARSE_KEY(p.ti, p.e), M->pr_head, M->pr_tail);
                M->pr_bytes += T.nb02;
                n_pf++; pf_bytes += T.nb02;
            }
            { std::lock_guard<std::mutex> lk(bmu); inflight_bytes -= std::min(inflight_bytes, bytes); }
            n_batches++;
            M->load_cv.notify_all();
        }
    }

    void push(const vk_sparse_ev & ev) {
        std::lock_guard<std::mutex> g(qmu);
        if (q.size() > 4096) q.pop_front();
        q.push_back(ev);
        qcv.notify_one();
    }

    // called by the manager under M->mu
    void credit_used(const vk_sparse_slot & s, int layer) {
        n_pf_used++;
        if (layer >= 0 && layer < (int) calib.size() && s.pf_bucket >= 0) calib[layer][s.pf_bucket].used++;
    }
    void credit_dead(const vk_sparse_slot & s, int layer) {
        n_pf_dead++;
        if (layer >= 0 && layer < (int) calib.size() && s.pf_bucket >= 0) calib[layer][s.pf_bucket].dead++;
    }

    void refresh_layout() {
        std::lock_guard<std::recursive_mutex> g(M->mu);
        if (M->tensors.size() == by_layer_n) return;
        by_layer.clear();
        for (uint32_t ti = 0; ti < M->tensors.size(); ti++) {
            auto & T = *M->tensors[ti];
            if (T.bufp && T.layer >= 0 && T.fd >= 0) by_layer[T.layer].push_back(ti);
            n_expert = std::max(n_expert, (int) T.n_expert);
            n_layer  = std::max(n_layer, T.layer + 1);
        }
        by_layer_n = M->tensors.size();
        cur.resize(n_layer); prev.resize(n_layer);
        ema.resize(n_layer, std::vector<float>(n_expert, 0.f));
        for (auto & e : ema) e.resize(n_expert, 0.f);
        calib.resize(n_layer);
    }

    void main() {
        while (!stop) {
            vk_sparse_ev ev;
            {
                std::unique_lock<std::mutex> lk(qmu);
                qcv.wait(lk, [this] { return stop || !q.empty(); });
                if (stop) return;
                ev = q.front(); q.pop_front();
            }
            refresh_layout();
            if (ev.layer < 0 || ev.layer >= n_layer) continue;
            handle(ev);
        }
    }

    void handle(const vk_sparse_ev & ev) {
        n_ev++;
        const int L = ev.layer;
        if (ev.tok != cur_tok) {
            prev.swap(cur);
            for (auto & c : cur) c.clear();
            cur_tok = ev.tok;
            tok_bytes = 0;
            if (++tok_since_snap >= 64) { tok_since_snap = 0; snapshot(); }
        }
        cur[L].assign(ev.ids, ev.ids + ev.n);
        // learn
        for (int a = std::max(0, L - depth); a < L; a++) {
            for (uint16_t x : cur[a]) {
                auto & r = cross[kc(a, x, L)];
                for (uint16_t y : cur[L]) r.add(y, K);
            }
        }
        for (uint16_t x : prev[L]) {
            auto & r = tt[kt(L, x)];
            for (uint16_t y : cur[L]) r.add(y, K);
        }
        auto & e = ema[L];
        for (auto & v : e) v *= (1.f - alpha);
        for (uint16_t y : cur[L]) e[y] += alpha;
        // predict the next layers of this token
        predict(L);
    }

    // keep a small library of distinct regimes (snapshots of the live profile)
    void snapshot() {
        float best = 0.f;
        for (auto & p : lib) best = std::max(best, similarity(p.f, ema));
        if (best < 0.8f) {
            if (lib.size() >= 16) lib.erase(lib.begin());
            lib.push_back({ ema, cur_tok });
        }
    }
    static float similarity(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
        double dot = 0, na = 0, nb = 0;
        for (size_t L = 0; L < a.size() && L < b.size(); L++)
            for (size_t i = 0; i < a[L].size() && i < b[L].size(); i++) { dot += a[L][i] * b[L][i]; na += a[L][i] * a[L][i]; nb += b[L][i] * b[L][i]; }
        return (na > 0 && nb > 0) ? (float) (dot / std::sqrt(na * nb)) : 0.f;
    }
    // how well does this token's known route fit a profile? expected fraction of its picks covered
    float match(const std::vector<std::vector<float>> & f, int upto) {
        double s = 0; int n = 0;
        for (int a = 0; a <= upto && a < (int) cur.size(); a++) {
            if (cur[a].empty()) continue;
            for (uint16_t x : cur[a]) s += f[a][x];
            n += (int) cur[a].size();
        }
        return n ? (float) (s / n) : 0.f;
    }

    // score candidates for target layer b given what is known; returns true if any source spoke
    bool score_layer(int b, bool next_token, const std::vector<std::vector<float>> * reg, std::vector<float> & q) {
        std::fill(q.begin(), q.end(), 1.f);
        bool any = false;
        if (!next_token) {
            for (int a = std::max(0, b - depth); a < (int) cur.size() && a < b; a++) {
                for (uint16_t x : cur[a]) {
                    auto r = cross.find(kc(a, x, b));
                    if (r == cross.end() || r->second.total < 8) continue;
                    for (auto & yc : r->second.v) { q[yc.first] *= 1.f - (float) yc.second / r->second.total; any = true; }
                }
            }
        }
        // token-to-token: for this token's later layers the previous token speaks; for the next
        // token's early layers this token does
        const auto & src = next_token ? cur[b] : prev[b];
        for (uint16_t x : src) {
            auto r = tt.find(kt(b, x));
            if (r == tt.end() || r->second.total < 8) continue;
            for (auto & yc : r->second.v) { q[yc.first] *= 1.f - (float) yc.second / r->second.total; any = true; }
        }
        for (int y = 0; y < n_expert; y++) {
            float p = ema[b][y];
            if (reg) p = std::max(p, (*reg)[b][y]);
            if (p > 0.02f) { q[y] *= 1.f - std::min(p, 0.95f); any = true; }
        }
        return any;
    }

    void predict(int L) {
        const std::vector<std::vector<float>> * reg = nullptr;
        {
            float live = match(ema, L), best = 0.f;
            for (auto & p : lib) { float m = match(p.f, L); if (m > best) { best = m; reg = &p.f; } }
            if (!(best >= 0.4f && best > live + 0.1f)) reg = nullptr; else n_lib_hits++;
        }
        { std::lock_guard<std::mutex> lk(bmu); if (inflight_bytes >= inflight_max) return; }
        std::vector<float> q(n_expert);
        std::vector<std::pair<float, uint16_t>> cand;
        auto b_ = std::make_unique<batch>();
        size_t batch_bytes = 0;

        // targets: this token's next layers, then (near the end) the next token's first layers
        std::vector<std::pair<int, bool>> targets;
        for (int b = L + 1; b <= std::min(L + depth, n_layer - 1); b++) targets.push_back({ b, false });
        if (L >= n_layer - 1 - next_tok_layers) {
            for (int b = 0; b < std::min(next_tok_layers, n_layer); b++) targets.push_back({ b, true });
        }

        for (auto & tg : targets) {
            const int b = tg.first;
            auto it = by_layer.find(b);
            if (it == by_layer.end()) continue;
            // stale: the compute thread already passed this layer of the current token
            if (!tg.second && M->cur_layer_seen.load() >= b) { n_stale++; continue; }
            if (!score_layer(b, tg.second, reg, q)) continue;
            cand.clear();
            for (int y = 0; y < n_expert; y++) { float sc = 1.f - q[y]; if (sc >= 0.15f) cand.push_back({ sc, (uint16_t) y }); }
            std::sort(cand.begin(), cand.end(), [](auto & u, auto & v) { return u.first > v.first; });
            int taken = 0;
            std::lock_guard<std::recursive_mutex> g(M->mu);
            for (auto & c : cand) {
                if (taken >= n_max || batch_bytes >= batch_max || tok_bytes >= tok_max) break;
                const int bucket = std::min(9, (int) (c.first * 10));
                auto & cb = calib[b][bucket];
                const bool explore = (int) cb.issued < explore_n;
                const uint32_t settled = cb.used + cb.dead;
                const double prec = settled ? (double) cb.used / settled : 0.0;
                if (!(explore ? c.first >= 0.4f : (settled >= (uint32_t) explore_n && prec >= p_min))) continue;
                bool absent = false;
                for (uint32_t ti : it->second) if (M->tensors[ti]->slots[c.second].state == VKS_ABSENT) absent = true;
                if (!absent) continue;
                for (uint32_t ti : it->second) {
                    auto & T = *M->tensors[ti];
                    auto & sl = T.slots[c.second];
                    if (sl.state != VKS_ABSENT) continue;
                    if (!M->make_room_prob(T.nb02, T.off + (size_t) c.second * T.nb02, sl.runs, b_->binds)) continue;
                    sl.state = VKS_LOADING; sl.pf_bucket = (int8_t) bucket; sl.used = 0;
                    M->read_slab(T, c.second, sl.runs, M->pf, &b_->remaining);
                    M->binds_for(T.vkbuf, T.off + (size_t) c.second * T.nb02, T.nb02, sl.runs, b_->binds, false);
                    b_->pend.push_back({ ti, c.second, b, bucket });
                    batch_bytes += T.nb02; tok_bytes += T.nb02;
                    cb.issued++;
                }
                taken++;
            }
        }
        if (b_->pend.empty() && b_->binds.empty()) return;
        std::lock_guard<std::mutex> lk(bmu);
        inflight_bytes += batch_bytes;
        bq.push_back(std::move(b_));
        bcv.notify_one();
    }
};

// ---------------------------------------------------------------- manager glue for the learner

void vk_sparse_mgr::learn_start() {
    if (!getenv("GGML_VK_SPARSE_PREFETCH")) return;
    learn = new vk_sparse_learner(this);
    fprintf(stderr, "ggml_vulkan: sparse prefetcher enabled (depth %d, p_min %.2f, probation %.1f GiB)\n",
            learn->depth, learn->p_min, learn->prob_cap / 1073741824.0);
}

void vk_sparse_mgr::learn_stop() {
    delete learn;
    learn = nullptr;
}

void vk_sparse_mgr::learn_event(uint32_t tok, int layer, const std::vector<uint32_t> & sel, uint8_t miss) {
    if (!learn) return;
    vk_sparse_ev ev;
    ev.tok = tok; ev.layer = (int16_t) layer; ev.n = (uint8_t) std::min<size_t>(sel.size(), 8); ev.miss = miss;
    for (int i = 0; i < ev.n; i++) ev.ids[i] = (uint16_t) sel[i];
    learn->push(ev);
}

void vk_sparse_mgr::learn_credit_used(const vk_sparse_slot & s, int layer) { if (learn) learn->credit_used(s, layer); }
void vk_sparse_mgr::learn_credit_dead(const vk_sparse_slot & s, int layer) { if (learn) learn->credit_dead(s, layer); }

void vk_sparse_mgr::learn_stats(char * buf, size_t n) {
    if (!learn) { buf[0] = 0; return; }
    snprintf(buf, n, " | pf issued %llu used %llu dead %llu (%.1f%% precision) %.2f GiB batches %llu stale %llu lib %llu probation %.1f GiB",
             (unsigned long long) learn->n_pf, (unsigned long long) learn->n_pf_used, (unsigned long long) learn->n_pf_dead,
             100.0 * learn->n_pf_used / (double) (learn->n_pf_used + learn->n_pf_dead + 1), learn->pf_bytes / 1073741824.0,
             (unsigned long long) learn->n_batches, (unsigned long long) learn->n_stale, (unsigned long long) learn->n_lib_hits, pr_bytes / 1073741824.0);
}
