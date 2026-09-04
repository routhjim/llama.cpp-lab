// GPU-resident MoE experts via Vulkan sparse binding.
//
// Included from ggml-vulkan.cpp (it needs vk_device / vk_buffer and the static buffer
// helpers); not a standalone translation unit.
//
// Problem: a 186 GiB expert pack on a 124 GiB box. The experts cannot all be resident, the
// GPU cannot read the file mapping directly (amdgpu refuses file-backed host-pointer import),
// and Vulkan has no fault-and-retry. What it does have is sparse residency: a buffer whose
// virtual range is fully addressable while physical memory is bound only where we say.
//
// Design:
//  * Expert tensors live in "<dev>_Sparse" buffers: a sparse VkBuffer of the full tensor size
//    with NO memory bound at allocation. Expert e sits at its natural offset e * nb02, so the
//    shaders, the graph and the loader see an ordinary Vulkan buffer. No id remapping.
//  * Physical memory is a pool of 1 GiB DEVICE_LOCAL|HOST_VISIBLE chunks carved into 64 KiB
//    granules. On this hardware the pool is device memory the CPU can write, so a cold expert
//    is pread() straight from the GGUF into the bytes the GPU will read -- no staging copy.
//  * Residency is CPU-managed: before a MUL_MAT_ID over a sparse tensor is recorded, the
//    graph is flushed, the selected expert ids are read back, every missing expert is read
//    and bound (one vkQueueBindSparse for the whole set), and only then is the matmul
//    recorded. residencyNonResidentStrict makes an unbound page read as ZERO, silently, so
//    binding must be proactive; the stall is what makes it correct by construction.
//  * Policy mirrors the CPU expert cache that measured best: recency (LRU) for decode, and a
//    separate transient ring for prefill, which touches essentially every expert and would
//    otherwise flush the hot set every pass (freeze-during-prefill semantics).
//
// Env:
//   GGML_VK_SPARSE_GIB=N            physical pool budget (default 48)
//   GGML_VK_SPARSE_TRANSIENT_GIB=N  part of the pool reserved for prefill scans (default 6)
//   GGML_VK_SPARSE_SCAN_TOKENS=N    a call over more than N tokens is prompt processing (default 4):
//                                   its misses stream through the transient ring, the LRU is frozen
//   GGML_VK_SPARSE_READERS=N        pread worker threads (default 16)
//   GGML_VK_SPARSE_MIRROR=<dir>     second copy of the model files (same names); misses are read
//                                   from both drives, split by expert parity
//   GGML_VK_SPARSE_WARM=<file>      "layer expert touches" ranking to admit after load
//   GGML_VK_SPARSE_WARM_N=N         only the first N lines of it
//   GGML_VK_SPARSE_DUMP=<file>      write touch counts in that format at exit
//   GGML_VK_SPARSE_STATS=1          print counters every 10 s
//   GGML_VK_SPARSE_STREAM=<file>    one line per decode layer: layer e0 e1 ... (routing trace)
//   GGML_VK_SPARSE_VALIDATE=1       assert residency after every ensure and spot-check bytes

#include <atomic>
#include <array>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_map>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <signal.h>
#include <climits>

#define VK_SPARSE_PAGE      4096ull
#define VK_SPARSE_GRANULE   (64ull * 1024)
#define VK_SPARSE_CHUNK     (1024ull * 1024 * 1024)
#define VK_SPARSE_KEY(t, e) ((uint32_t) (t) << 16 | (uint32_t) (e))

enum vk_sparse_state : uint8_t { VKS_ABSENT = 0, VKS_LRU = 1, VKS_TRANSIENT = 2, VKS_FIXED = 3, VKS_LOADING = 4, VKS_PROB = 5 };

struct vk_sparse_run { uint32_t chunk, gran, n; };   // n granules starting at gran in chunk

struct vk_sparse_slot {
    std::vector<vk_sparse_run> runs;
    int32_t prev = -1, next = -1;   // list links (LRU or transient, by state)
    uint8_t state  = VKS_ABSENT;
    uint8_t pinned = 0;             // selected since the last GPU drain: not evictable
    uint8_t used   = 0;
    int8_t  pf_bucket = -1;         // prefetched: calibration bucket that issued it
};

struct vk_sparse_tensor {
    vk_buffer_struct * bufp = nullptr;   // not owning: the buffer's destructor detaches us
    vk::Buffer  vkbuf;
    size_t      off      = 0;       // tensor offset in buf
    size_t      nb02     = 0;       // bytes per expert
    uint32_t    n_expert = 0;
    int         fd       = -1;      // buffered, for small checks; -1 = no file source (set_tensor path only)
    int         fd_direct = -1;     // O_DIRECT, for slab reads (falls back to fd if the fs refuses)
    int         fd_mirror = -1;     // O_DIRECT on the mirror copy, or -1
    off_t       file_off = -1;
    int         layer    = -1;
    std::string name;
    std::vector<vk_sparse_slot> slots;
};

struct vk_sparse_chunk {
    vk::DeviceMemory mem;
    uint8_t * map = nullptr;
    uint32_t n_gran = 0, n_free = 0;
    std::vector<uint64_t> bits;     // 1 = free
};

// Per sparse VkBuffer: pages bound outside the expert cache (set_tensor / memset / clear on
// arbitrary ranges), released when the buffer dies.
struct vk_sparse_buf {
    std::vector<std::pair<size_t, size_t>> fixed;   // [off, off+size)
    std::vector<vk_sparse_run> fixed_runs;
};

struct vk_sparse_bind { vk::Buffer buf; size_t res_off, size; vk::DeviceMemory mem; size_t mem_off; };
struct vk_sparse_job  { int fd; bool direct; off_t off; uint8_t * dst; size_t len; std::atomic<uint32_t> * ctr; };
struct vk_sparse_ev   { uint32_t tok; int16_t layer; uint8_t n, miss; uint16_t ids[8]; };

// a pool of pread workers; the demand path and the prefetcher each own one
struct vk_sparse_rpool {
    std::vector<std::thread> th;
    std::deque<vk_sparse_job> jobs;
    std::mutex mu;
    std::condition_variable cv, dcv, any;   // any: notified on every completion (batch counters)
    size_t inflight = 0;
    bool   stop = false;
    std::atomic<int> err {0};

    void start(int n) { for (int i = 0; i < n; i++) th.emplace_back([this] { main(); }); }
    void stopall() {
        { std::lock_guard<std::mutex> g(mu); stop = true; }
        cv.notify_all(); any.notify_all();
        for (auto & t : th) t.join();
        th.clear();
    }
    ~vk_sparse_rpool() { if (!th.empty()) stopall(); }
    void submit(int fd, bool direct, off_t off, uint8_t * dst, size_t len, std::atomic<uint32_t> * ctr = nullptr) {
        std::lock_guard<std::mutex> lk(mu);
        jobs.push_back({ fd, direct, off, dst, len, ctr });
        inflight++;
        cv.notify_one();
    }
    // wait until *ctr reaches zero (a batch of jobs completed)
    void wait_ctr(std::atomic<uint32_t> & ctr) {
        std::unique_lock<std::mutex> lk(mu);
        any.wait(lk, [&] { return ctr.load() == 0 || stop; });
    }
    void drain(const char * what) {
        std::unique_lock<std::mutex> lk(mu);
        dcv.wait(lk, [this] { return inflight == 0; });
        if (err.exchange(0)) {
            GGML_ABORT("ggml_vulkan: sparse expert read failed (%s); refusing to compute on zeros", what);
        }
    }
    void main();
};

struct vk_sparse_mgr {
    vk_device_struct * dev;
    std::recursive_mutex mu;

    uint32_t memtype = UINT32_MAX;
    size_t   cap = 0, transient_cap = 0, lru_cap = 0;
    size_t   chunk_bytes = 0;
    uint32_t scan_tokens = 4;
    bool     stats = false, validate = false;
    const char * dump_path = nullptr;
    FILE * stream = nullptr;

    std::vector<vk_sparse_chunk> chunks;
    std::vector<std::unique_ptr<vk_sparse_tensor>> tensors;
    std::unordered_map<const ggml_tensor *, uint32_t> by_tensor;
    struct vk_sparse_fds { int buffered, direct, mirror; };
    std::map<std::pair<dev_t, ino_t>, vk_sparse_fds> fds;   // per file
    const char * mirror_dir = nullptr;

    int32_t lru_head = -1, lru_tail = -1, tr_head = -1, tr_tail = -1;
    size_t  lru_bytes = 0, tr_bytes = 0;
    std::vector<uint32_t> pinned;

    vk::Fence bind_fence;

    // readers: demand path and prefetcher
    vk_sparse_rpool rd, pf;

    // probation list for prefetched slabs (evicted before the LRU), counted inside lru_cap
    int32_t pr_head = -1, pr_tail = -1;
    size_t  pr_bytes = 0;
    size_t  prob_cap = 0;                  // reserved for prefetches inside lru_cap (set by the learner)
    std::condition_variable_any load_cv;   // a LOADING slab became resident
    uint32_t tok_seq = 0;
    int      last_layer_ev = -1;
    struct vk_sparse_learner * learn = nullptr;
    std::atomic<int> cur_layer_seen {-1};
    void learn_start();
    void learn_stop();
    void learn_event(uint32_t tok, int layer, const std::vector<uint32_t> & sel, uint8_t miss);
    void learn_credit_used(const vk_sparse_slot & s, int layer);
    void learn_credit_dead(const vk_sparse_slot & s, int layer);
    void learn_stats(char * buf, size_t n);

    // kernel-log watcher: amdgpu can fail a page-table update for a sparse bind (BO_VA -12) and
    // Vulkan never learns of it; the range then reads as zeros. Re-bind everything when it happens.
    std::thread kmsg_thread;
    std::atomic<int> kmsg_errors {0};
    int kmsg_seen = 0;
    uint64_t n_rebind = 0;
    int   kmsg_fd  = -1;
    pid_t kmsg_pid = -1;

    // warm set
    std::thread warm_thread;
    std::vector<std::pair<int, int>> warm;   // (layer, expert), hottest first
    std::atomic<bool> warm_started {false};

    // stats
    uint64_t n_hit = 0, n_miss = 0, n_scan_hit = 0, n_scan_miss = 0, n_evict = 0, n_stall = 0, n_scan_calls = 0, n_calls = 0;
    uint64_t bytes_read = 0, t_bind_us = 0, t_read_us = 0, t_ensure_us = 0;
    double   t_last = 0;
    std::unordered_map<uint64_t, uint32_t> ref;   // (layer<<32|expert) -> touches

    explicit vk_sparse_mgr(vk_device_struct * d) : dev(d) {
        auto env_d = [](const char * n, double def) { const char * v = getenv(n); return v ? atof(v) : def; };
        cap           = (size_t) (env_d("GGML_VK_SPARSE_GIB", 48.0) * 1073741824.0);
        transient_cap = (size_t) (env_d("GGML_VK_SPARSE_TRANSIENT_GIB", 6.0) * 1073741824.0);
        transient_cap = std::min(transient_cap, cap / 2);
        lru_cap       = cap - transient_cap;
        scan_tokens   = (uint32_t) env_d("GGML_VK_SPARSE_SCAN_TOKENS", 4);
        stats         = getenv("GGML_VK_SPARSE_STATS") != nullptr;
        validate      = getenv("GGML_VK_SPARSE_VALIDATE") != nullptr;
        dump_path     = getenv("GGML_VK_SPARSE_DUMP");
        if (const char * sp = getenv("GGML_VK_SPARSE_STREAM")) { stream = fopen(sp, "w"); if (stream) setvbuf(stream, nullptr, _IOFBF, 1 << 20); }
        bind_fence    = dev->device.createFence({});
        mirror_dir = getenv("GGML_VK_SPARSE_MIRROR");
        rd.start(std::max(1, (int) env_d("GGML_VK_SPARSE_READERS", 16)));
        if (const char * w = getenv("GGML_VK_SPARSE_WARM")) {
            load_warm_file(w);
        }
        if (!getenv("GGML_VK_SPARSE_NO_KMSG_WATCH")) {
            kmsg_thread = std::thread([this] { kmsg_main(); });
        }
        learn_start();
    }

    ~vk_sparse_mgr() {
        learn_stop();
        if (warm_thread.joinable()) {
            warm_thread.join();
        }
        if (kmsg_thread.joinable()) {
            if (kmsg_pid > 0) kill(kmsg_pid, SIGKILL);   // journalctl -f never exits on its own; EOF unblocks the reader
            kmsg_thread.join();
            if (kmsg_pid > 0) waitpid(kmsg_pid, nullptr, 0);
        }
        rd.stopall();
        pf.stopall();
        if (dump_path) {
            dump_ref();
        }
        if (stream) fclose(stream);
        for (auto & c : chunks) {
            dev->device.unmapMemory(c.mem);
            dev->device.freeMemory(c.mem);
        }
        for (auto & kv : fds) {
            close(kv.second.buffered);
            if (kv.second.direct >= 0) close(kv.second.direct);
            if (kv.second.mirror >= 0) close(kv.second.mirror);
        }
        dev->device.destroyFence(bind_fence);
    }

    // ---------------------------------------------------------------- physical pool

    bool grow_chunk() {
        const size_t sz = std::min<size_t>(VK_SPARSE_CHUNK, cap > chunk_bytes ? cap - chunk_bytes : 0) & ~(VK_SPARSE_GRANULE - 1);
        if (sz == 0 || memtype == UINT32_MAX) {
            return false;
        }
        vk_sparse_chunk c;
        try {
            c.mem = dev->device.allocateMemory({ sz, memtype });
        } catch (const vk::SystemError & e) {
            GGML_LOG_WARN("ggml_vulkan: sparse pool: chunk allocation failed at %.1f GiB (%s)\n",
                          chunk_bytes / 1073741824.0, e.what());
            cap = chunk_bytes;   // stop asking
            lru_cap = cap > transient_cap ? cap - transient_cap : 0;
            return false;
        }
        c.map    = (uint8_t *) dev->device.mapMemory(c.mem, 0, VK_WHOLE_SIZE);
        c.n_gran = (uint32_t) (sz / VK_SPARSE_GRANULE);
        c.n_free = c.n_gran;
        c.bits.assign((c.n_gran + 63) / 64, ~0ull);
        chunks.push_back(std::move(c));
        chunk_bytes += sz;
        return true;
    }

    size_t free_bytes() const {
        size_t n = 0;
        for (const auto & c : chunks) {
            n += (size_t) c.n_free * VK_SPARSE_GRANULE;
        }
        return n + (cap > chunk_bytes ? cap - chunk_bytes : 0);
    }

    static bool bit(const std::vector<uint64_t> & b, uint32_t i) { return (b[i >> 6] >> (i & 63)) & 1; }
    static void clr(std::vector<uint64_t> & b, uint32_t i, uint32_t n) { for (uint32_t k = i; k < i + n; k++) b[k >> 6] &= ~(1ull << (k & 63)); }
    static void set(std::vector<uint64_t> & b, uint32_t i, uint32_t n) { for (uint32_t k = i; k < i + n; k++) b[k >> 6] |=  (1ull << (k & 63)); }

    // longest free run in chunk c starting the scan at *pos; returns its start, length in *len
    static uint32_t scan_run(const vk_sparse_chunk & c, uint32_t pos, uint32_t * len) {
        while (pos < c.n_gran && !bit(c.bits, pos)) pos++;
        uint32_t s = pos;
        while (pos < c.n_gran && bit(c.bits, pos)) pos++;
        *len = pos - s;
        return s;
    }

    // Allocate `bytes` of physical memory for the virtual range at `va`. Prefers one contiguous
    // run whose 2 MiB phase matches the VA: amdgpu keeps GPU page tables in the 512 MiB VRAM
    // carve-out, and a fully covered, phase-matched 2 MiB block needs no leaf page table.
    // Falls back to any contiguous run, then to pieces, so fragmentation never refuses a slab.
    bool phys_alloc(size_t bytes, size_t va, std::vector<vk_sparse_run> & out) {
        const uint32_t G2M   = (uint32_t) ((2u << 20) / VK_SPARSE_GRANULE);   // granules per 2 MiB
        const uint32_t phase = (uint32_t) ((va % (2u << 20)) / VK_SPARSE_GRANULE);
        uint32_t need = (uint32_t) ((bytes + VK_SPARSE_GRANULE - 1) / VK_SPARSE_GRANULE);
        out.clear();
        for (;;) {
            // pass 1: contiguous, phase-matched; pass 1b: contiguous anywhere
            for (int matched = 1; matched >= 0; matched--) {
                for (uint32_t ci = 0; ci < chunks.size(); ci++) {
                    auto & c = chunks[ci];
                    if (c.n_free < need) continue;
                    uint32_t pos = 0, len = 0;
                    while (pos < c.n_gran) {
                        uint32_t s = scan_run(c, pos, &len);
                        uint32_t s2 = s;
                        if (matched) {
                            s2 = s + ((phase + G2M - (s % G2M)) % G2M);
                        }
                        if (s2 + need <= s + len) {
                            clr(c.bits, s2, need); c.n_free -= need;
                            out.push_back({ ci, s2, need });
                            return true;
                        }
                        pos = s + len;
                    }
                }
            }
            // pass 2: pieces
            size_t total_free = 0;
            for (auto & c : chunks) total_free += c.n_free;
            if (total_free >= need) {
                uint32_t left = need;
                for (uint32_t ci = 0; ci < chunks.size() && left; ci++) {
                    auto & c = chunks[ci];
                    uint32_t pos = 0, len = 0;
                    while (pos < c.n_gran && left) {
                        uint32_t s = scan_run(c, pos, &len);
                        if (len) {
                            uint32_t take = std::min(len, left);
                            clr(c.bits, s, take); c.n_free -= take;
                            out.push_back({ ci, s, take });
                            left -= take;
                        }
                        pos = s + len;
                    }
                }
                return true;
            }
            if (!grow_chunk()) {
                return false;
            }
        }
    }

    void phys_free(std::vector<vk_sparse_run> & runs) {
        for (auto & r : runs) {
            set(chunks[r.chunk].bits, r.gran, r.n);
            chunks[r.chunk].n_free += r.n;
        }
        runs.clear();
    }

    uint8_t * run_ptr(const vk_sparse_run & r) const { return chunks[r.chunk].map + (size_t) r.gran * VK_SPARSE_GRANULE; }

    // ---------------------------------------------------------------- binding

    void binds_for(vk::Buffer buf, size_t res_off, size_t bytes, const std::vector<vk_sparse_run> & runs,
                   std::vector<vk_sparse_bind> & out, bool unbind) {
        size_t done = 0;
        for (const auto & r : runs) {
            size_t sz = std::min<size_t>((size_t) r.n * VK_SPARSE_GRANULE, bytes - done);
            out.push_back({ buf, res_off + done, sz,
                            unbind ? vk::DeviceMemory() : chunks[r.chunk].mem,
                            (size_t) r.gran * VK_SPARSE_GRANULE });
            done += sz;
            if (done >= bytes) break;
        }
    }

    // One vkQueueBindSparse for everything, fenced. The bind is a queue operation; the
    // fence is what orders it against the command buffers recorded afterwards.
    void bind_apply(std::vector<vk_sparse_bind> & binds) { bind_apply(binds, bind_fence); }
    void bind_apply(std::vector<vk_sparse_bind> & binds, vk::Fence fence) {
        if (binds.empty()) return;
        auto t0 = std::chrono::steady_clock::now();
        std::sort(binds.begin(), binds.end(), [](const vk_sparse_bind & a, const vk_sparse_bind & b) {
            return a.buf < b.buf || (a.buf == b.buf && a.res_off < b.res_off);
        });
        std::vector<vk::SparseMemoryBind> smb(binds.size());
        for (size_t i = 0; i < binds.size(); i++) {
            smb[i].resourceOffset = binds[i].res_off;
            smb[i].size           = binds[i].size;
            smb[i].memory         = binds[i].mem;
            smb[i].memoryOffset   = binds[i].mem_off;
        }
        std::vector<vk::SparseBufferMemoryBindInfo> infos;
        for (size_t i = 0; i < binds.size();) {
            size_t j = i;
            while (j < binds.size() && binds[j].buf == binds[i].buf) j++;
            infos.push_back({ binds[i].buf, (uint32_t) (j - i), &smb[i] });
            i = j;
        }
        vk::BindSparseInfo bsi;
        bsi.bufferBindCount = (uint32_t) infos.size();
        bsi.pBufferBinds    = infos.data();
        auto & qh = *dev->compute_queue->handle;
        qh.lock();
        vk::Result res;
        try {
            res = qh.queue.bindSparse(1, &bsi, fence);
        } catch (...) {
            qh.unlock();
            throw;
        }
        qh.unlock();
        if (res != vk::Result::eSuccess) {
            GGML_ABORT("ggml_vulkan: vkQueueBindSparse failed: %s", vk::to_string(res).c_str());
        }
        res = dev->device.waitForFences({ fence }, true, UINT64_MAX);
        if (res != vk::Result::eSuccess) {
            GGML_ABORT("ggml_vulkan: sparse bind fence wait failed: %s", vk::to_string(res).c_str());
        }
        dev->device.resetFences({ fence });
        t_bind_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }

    // ---------------------------------------------------------------- readers

    void read_slab(const vk_sparse_tensor & T, uint32_t e, const std::vector<vk_sparse_run> & runs) { read_slab(T, e, runs, rd); }
    void read_slab(const vk_sparse_tensor & T, uint32_t e, const std::vector<vk_sparse_run> & runs, vk_sparse_rpool & pool, std::atomic<uint32_t> * ctr = nullptr) {
        off_t  off  = T.file_off + (off_t) e * (off_t) T.nb02;
        size_t left = T.nb02;
        for (const auto & r : runs) {
            size_t sz = std::min<size_t>((size_t) r.n * VK_SPARSE_GRANULE, left);
            // split misses across the two drives by expert parity
            const int fdd = (T.fd_mirror >= 0 && (e & 1)) ? T.fd_mirror : T.fd_direct;
            if (ctr) ctr->fetch_add(1);
            if (fdd >= 0) pool.submit(fdd, true, off, run_ptr(r), sz, ctr);
            else          pool.submit(T.fd, false, off, run_ptr(r), sz, ctr);
            off += (off_t) sz; left -= sz;
            if (!left) break;
        }
        bytes_read += T.nb02;
    }

    void read_drain() { rd.drain("demand"); }

    // ---------------------------------------------------------------- lists

    vk_sparse_slot & slot(uint32_t k) { return tensors[k >> 16]->slots[k & 0xFFFF]; }
    size_t slab_bytes(uint32_t k) const { return tensors[k >> 16]->nb02; }

    void list_unlink(uint32_t k, int32_t & head, int32_t & tail) {
        auto & s = slot(k);
        if (s.prev >= 0) slot(s.prev).next = s.next; else head = s.next;
        if (s.next >= 0) slot(s.next).prev = s.prev; else tail = s.prev;
        s.prev = s.next = -1;
    }
    void list_push(uint32_t k, int32_t & head, int32_t & tail) {   // MRU at head
        auto & s = slot(k);
        s.prev = -1; s.next = head;
        if (head >= 0) slot(head).prev = (int32_t) k;
        head = (int32_t) k;
        if (tail < 0) tail = (int32_t) k;
    }

    void unlink_any(uint32_t k) {
        auto & s = slot(k);
        if (s.state == VKS_LRU)       { list_unlink(k, lru_head, lru_tail); lru_bytes -= slab_bytes(k); }
        if (s.state == VKS_TRANSIENT) { list_unlink(k, tr_head,  tr_tail);  tr_bytes  -= slab_bytes(k); }
        if (s.state == VKS_PROB)      { list_unlink(k, pr_head,  pr_tail);  pr_bytes  -= slab_bytes(k); }
    }

    // prefetched-but-unused slabs go first
    bool evict_probation(std::vector<vk_sparse_bind> & unbinds) {
        int32_t k = pr_tail;
        while (k >= 0 && slot(k).pinned) k = slot(k).prev;
        if (k < 0) return false;
        auto & s = slot(k);
        auto & T = *tensors[k >> 16];
        if (!s.used) learn_credit_dead(s, T.layer);
        binds_for(T.vkbuf, T.off + (size_t) (k & 0xFFFF) * T.nb02, T.nb02, s.runs, unbinds, true);
        unlink_any((uint32_t) k);
        phys_free(s.runs);
        s.state = VKS_ABSENT; s.used = 0; s.pf_bucket = -1;
        n_evict++;
        return true;
    }

    // evict the coldest unpinned entry of one list; false if there is none
    bool evict_one(bool transient, std::vector<vk_sparse_bind> & unbinds) {
        int32_t k = transient ? tr_tail : lru_tail;
        while (k >= 0 && slot(k).pinned) k = slot(k).prev;
        if (k < 0) return false;
        auto & s = slot(k);
        auto & T = *tensors[k >> 16];
        binds_for(T.vkbuf, T.off + (size_t) (k & 0xFFFF) * T.nb02, T.nb02, s.runs, unbinds, true);
        unlink_any((uint32_t) k);
        phys_free(s.runs);
        s.state = VKS_ABSENT; s.used = 0;
        n_evict++;
        return true;
    }

    // room for a prefetched slab: only probation is ever evicted for it
    bool make_room_prob(size_t need, size_t va, std::vector<vk_sparse_run> & runs, std::vector<vk_sparse_bind> & unbinds) {
        while (pr_bytes + need > prob_cap && evict_probation(unbinds)) {}
        if (pr_bytes + need > prob_cap) return false;
        while (!phys_alloc(need, va, runs)) {
            if (!evict_probation(unbinds)) return false;
        }
        return true;
    }

    // make room for `need` bytes in the given list and take physical memory for it
    bool make_room(bool transient, size_t need, size_t va, std::vector<vk_sparse_run> & runs, std::vector<vk_sparse_bind> & unbinds) {
        size_t & cur = transient ? tr_bytes : lru_bytes;
        size_t   lim = transient ? transient_cap : (lru_cap > prob_cap ? lru_cap - prob_cap : 0);
        while (cur + need > lim && evict_one(transient, unbinds)) {}
        if (cur + need > lim) {
            // own list exhausted (everything pinned, or cap smaller than one slab): borrow
            while (!phys_alloc(need, va, runs) && evict_one(!transient, unbinds)) {}
            if (runs.empty() && !phys_alloc(need, va, runs)) return false;
            return true;
        }
        while (!phys_alloc(need, va, runs)) {
            // fragmentation or fixed pages ate the budget: evict more, own list first
            if (!evict_one(transient, unbinds) && !evict_one(!transient, unbinds)) return false;
        }
        return true;
    }

    // ---------------------------------------------------------------- registration

    static int parse_layer(const char * name) {
        int l = -1;
        return (sscanf(name, "blk.%d.", &l) == 1) ? l : -1;
    }

    uint32_t register_tensor(const ggml_tensor * t) {
        auto it = by_tensor.find(t);
        if (it != by_tensor.end()) return it->second;
        GGML_ASSERT(t->view_src == nullptr && "sparse expert tensors must not be views");
        ggml_backend_vk_buffer_context * bctx = (ggml_backend_vk_buffer_context *) t->buffer->context;
        GGML_ASSERT(bctx->dev_buffer->sparse && "tensor is not in a sparse buffer");
        auto T = std::make_unique<vk_sparse_tensor>();
        T->bufp     = bctx->dev_buffer.get();
        T->vkbuf    = bctx->dev_buffer->buffer;
        T->off      = vk_tensor_offset(t);
        T->nb02     = t->nb[2];
        T->n_expert = (uint32_t) t->ne[2];
        T->layer    = parse_layer(t->name);
        T->name     = t->name;
        GGML_ASSERT(T->n_expert < 65536 && tensors.size() < 65535);
        GGML_ASSERT(ggml_is_contiguous(t));
        if ((T->off % VK_SPARSE_PAGE) || (T->nb02 % VK_SPARSE_PAGE)) {
            GGML_ABORT("ggml_vulkan: sparse tensor %s: offset %zu / expert stride %zu not %llu-aligned",
                       t->name, T->off, T->nb02, (unsigned long long) VK_SPARSE_PAGE);
        }
        T->slots.resize(T->n_expert);
        // pages already bound by set_tensor (test path) count as permanently resident
        for (auto & fr : T->bufp->sparse->fixed) {
            for (uint32_t e = 0; e < T->n_expert; e++) {
                size_t a = T->off + (size_t) e * T->nb02;
                if (a >= fr.first && a + T->nb02 <= fr.first + fr.second) T->slots[e].state = VKS_FIXED;
            }
        }
        tensors.push_back(std::move(T));
        by_tensor[t] = (uint32_t) tensors.size() - 1;
        return (uint32_t) tensors.size() - 1;
    }

    bool set_source(const ggml_tensor * t, int fd, size_t file_off) {
        std::lock_guard<std::recursive_mutex> g(mu);
        uint32_t ti = register_tensor(t);
        struct stat st;
        if (fstat(fd, &st) != 0) return false;
        auto key = std::make_pair(st.st_dev, st.st_ino);
        auto it = fds.find(key);
        if (it == fds.end()) {
            // own open file description: the loader's carries FADV_SEQUENTIAL readahead that
            // would spill page cache past every slab we read
            char link[64];
            snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
            int d = open(link, O_RDONLY | O_CLOEXEC);
            if (d < 0) d = dup(fd);
            if (d < 0) return false;
            posix_fadvise(d, 0, 0, POSIX_FADV_RANDOM);
            // O_DIRECT: one 5 MiB read is one in-flight I/O; buffered reads with readahead off
            // go out as synchronous 256 KiB chunks and cap the drive at ~2.5 GiB/s
            int dd = open(link, O_RDONLY | O_CLOEXEC | O_DIRECT);
            if (dd < 0) GGML_LOG_WARN("ggml_vulkan: sparse: O_DIRECT unavailable for %s, using buffered reads\n", link);
            int dm = -1;
            if (mirror_dir && dd >= 0) {
                // same file name in the mirror directory; must be byte-identical
                char real[PATH_MAX];
                ssize_t n = readlink(link, real, sizeof real - 1);
                if (n > 0) {
                    real[n] = 0;
                    const char * base = strrchr(real, '/');
                    std::string mp = std::string(mirror_dir) + "/" + (base ? base + 1 : real);
                    struct stat ms;
                    if (stat(mp.c_str(), &ms) == 0 && ms.st_size == st.st_size) {
                        dm = open(mp.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
                    }
                    if (dm < 0) GGML_LOG_WARN("ggml_vulkan: sparse: no usable mirror for %s at %s\n", real, mp.c_str());
                }
            }
            it = fds.emplace(key, vk_sparse_fds{ d, dd, dm }).first;
        }
        tensors[ti]->fd        = it->second.buffered;
        tensors[ti]->fd_direct = it->second.direct;
        tensors[ti]->fd_mirror = it->second.mirror;
        tensors[ti]->file_off = (off_t) file_off;
        return true;
    }

    // ---------------------------------------------------------------- the hot path

    // Pins protect the experts of the layer being recorded from eviction by the next layer's
    // ensure(). Every stall drains the GPU first, so at a stall all older pins can go.
    void release_pins() {
        std::lock_guard<std::recursive_mutex> g(mu);
        for (uint32_t k : pinned) slot(k).pinned = 0;
        pinned.clear();
    }

    void begin_graph() {
        release_pins();
    }

    void ensure(const ggml_tensor * t, const int32_t * ids, size_t n_ids, size_t n_tokens) {
        ensure_many(&t, 1, ids, n_ids, n_tokens);
    }

    // One I/O wave for every tensor that shares this ids set (gate/up/down of a layer): all
    // misses are read in parallel and bound with one queue op, instead of three serial rounds.
    void ensure_many(const ggml_tensor * const * ts, size_t n_t, const int32_t * ids, size_t n_ids, size_t n_tokens) {
        auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::recursive_mutex> g(mu);
        // prompt processing touches most of the pack; keep it out of the decode LRU (freeze)
        const bool scan = n_tokens > scan_tokens;
        std::vector<vk_sparse_bind> binds;
        std::vector<std::vector<uint32_t>> sels(n_t);
        size_t n_missing = 0;
        auto tr0 = std::chrono::steady_clock::now();

        for (size_t x = 0; x < n_t; x++) {
            uint32_t ti = register_tensor(ts[x]);
            auto & T = *tensors[ti];
            auto & sel = sels[x];
            {
                static thread_local std::vector<uint8_t> seen;
                seen.assign(T.n_expert, 0);
                for (size_t i = 0; i < n_ids; i++) {
                    int32_t e = ids[i];
                    if (e < 0 || (uint32_t) e >= T.n_expert) {
                        GGML_ABORT("ggml_vulkan: sparse: expert id %d out of range for %s (%u experts)", e, T.name.c_str(), T.n_expert);
                    }
                    if (!seen[e]) { seen[e] = 1; sel.push_back((uint32_t) e); }
                }
            }
            n_calls++; if (scan) n_scan_calls++;
            if (stream && !scan && x == 0 && n_tokens == 1 && T.layer >= 0) {
                fprintf(stream, "%d", T.layer);
                for (uint32_t e : sel) fprintf(stream, " %u", e);
                fputc('\n', stream);
            }

            std::vector<uint32_t> misses;
            uint8_t miss_mask = 0;
            uint64_t & hits_ctr = scan ? n_scan_hit : n_hit;
            for (size_t si = 0; si < sel.size(); si++) {
                uint32_t e = sel[si];
                uint32_t k = VK_SPARSE_KEY(ti, e);
                auto & s = T.slots[e];
                if (!s.pinned) { s.pinned = 1; pinned.push_back(k); }
                if (T.layer >= 0) ref[(uint64_t) T.layer << 32 | e]++;
                if (s.state == VKS_LOADING) {
                    // a prefetch for exactly this slab is in flight: wait for it instead of reading twice
                    load_cv.wait(g, [&] { return s.state != VKS_LOADING; });
                }
                if (s.state == VKS_PROB) {
                    hits_ctr++;
                    if (!s.used) { s.used = 1; learn_credit_used(s, T.layer); }
                    if (!scan) {   // confirmed use: promote out of probation into the LRU budget
                        list_unlink(k, pr_head, pr_tail); pr_bytes -= T.nb02;
                        list_push(k, lru_head, lru_tail); lru_bytes += T.nb02;
                        s.state = VKS_LRU;
                    }
                    continue;
                }
                switch (s.state) {
                    case VKS_ABSENT:
                        misses.push_back(e);
                        if (si < 8) miss_mask |= (uint8_t) (1u << si);
                        break;
                    case VKS_LRU:
                        hits_ctr++;
                        if (!scan) { list_unlink(k, lru_head, lru_tail); list_push(k, lru_head, lru_tail); }
                        break;
                    case VKS_TRANSIENT:
                        hits_ctr++;
                        if (!scan) {   // decode wants it: promote out of the ring
                            list_unlink(k, tr_head, tr_tail); tr_bytes -= T.nb02;
                            list_push(k, lru_head, lru_tail); lru_bytes += T.nb02;
                            s.state = VKS_LRU;
                        }
                        break;
                    case VKS_FIXED:
                        hits_ctr++;
                        break;
                }
            }
            (scan ? n_scan_miss : n_miss) += misses.size();
            if (x == 0 && !scan && T.layer >= 0) {
                if (T.layer <= last_layer_ev) tok_seq++;
                last_layer_ev = T.layer;
                cur_layer_seen.store(T.layer);
                learn_event(tok_seq, T.layer, sel, miss_mask);
            }
            if (misses.empty()) continue;
            if (T.fd < 0) {
                GGML_ABORT("ggml_vulkan: sparse: %s expert %u is not resident and has no file source", T.name.c_str(), misses[0]);
            }
            for (uint32_t e : misses) {
                uint32_t k = VK_SPARSE_KEY(ti, e);
                auto & s = T.slots[e];
                if (!make_room(scan, T.nb02, T.off + (size_t) e * T.nb02, s.runs, binds)) {
                    GGML_ABORT("ggml_vulkan: sparse: cannot make room for %s expert %u (pool %.1f GiB, lru %.1f, transient %.1f, %zu pinned)",
                               T.name.c_str(), e, cap / 1073741824.0, lru_bytes / 1073741824.0, tr_bytes / 1073741824.0, pinned.size());
                }
                read_slab(T, e, s.runs);
                binds_for(T.vkbuf, T.off + (size_t) e * T.nb02, T.nb02, s.runs, binds, false);
                if (scan) { s.state = VKS_TRANSIENT; list_push(k, tr_head, tr_tail); tr_bytes += T.nb02; }
                else      { s.state = VKS_LRU;       list_push(k, lru_head, lru_tail); lru_bytes += T.nb02; }
                n_missing++;
            }
        }

        if (n_missing) {
            bind_apply(binds);   // binds + the evictions' unbinds, one queue op
            read_drain();
            t_read_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tr0).count();
        }
        if (kmsg_errors.load() != kmsg_seen) {
            rebind_all();
            if (kmsg_errors.load() != kmsg_seen) {
                GGML_ABORT("ggml_vulkan: sparse: page-table updates keep failing (BO_VA -12); refusing to compute on unmapped experts");
            }
        }
        if (validate) {
            for (size_t x = 0; x < n_t; x++) check(*tensors[by_tensor[ts[x]]], sels[x]);
        }
        t_ensure_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        maybe_print();
    }

    // Debug guard against silent zeros: every selected expert must be bound, and one of them
    // must hold the file's bytes.
    void check(const vk_sparse_tensor & T, const std::vector<uint32_t> & sel) {
        for (uint32_t e : sel) {
            const auto & s = T.slots[e];
            if (s.state == VKS_ABSENT || s.state == VKS_LOADING || (s.state != VKS_FIXED && s.runs.empty())) {
                GGML_ABORT("ggml_vulkan: sparse VALIDATE: %s expert %u selected but not resident", T.name.c_str(), e);
            }
        }
        if (T.fd < 0 || sel.empty()) return;
        uint32_t e = sel[rand() % sel.size()];
        const auto & s = T.slots[e];
        if (s.runs.empty()) return;
        uint8_t want[4096];
        ssize_t n = pread(T.fd, want, sizeof want, T.file_off + (off_t) e * (off_t) T.nb02);
        if (n != (ssize_t) sizeof want || memcmp(want, run_ptr(s.runs[0]), sizeof want) != 0) {
            GGML_ABORT("ggml_vulkan: sparse VALIDATE: %s expert %u: GPU bytes differ from file", T.name.c_str(), e);
        }
    }

    // ---------------------------------------------------------------- set_tensor path

    // Bind (and zero-fill) every page of [off, off+size) in buf that is not bound yet. Used by
    // set_tensor / memset / clear so the buffer behaves like a normal one for tests and for
    // non-file sources. Such pages are never evicted.
    void pin_range(vk_buffer & buf, size_t off, size_t size) {
        std::lock_guard<std::recursive_mutex> g(mu);
        size_t a = off & ~(VK_SPARSE_PAGE - 1);
        size_t b = (off + size + VK_SPARSE_PAGE - 1) & ~(VK_SPARSE_PAGE - 1);
        b = std::min(b, buf->size);
        // subtract already-fixed ranges (kept simple: ranges are few and rarely overlap)
        std::vector<std::pair<size_t, size_t>> todo { { a, b } };
        for (auto & fr : buf->sparse->fixed) {
            std::vector<std::pair<size_t, size_t>> next;
            for (auto & r : todo) {
                size_t fa = fr.first, fb = fr.first + fr.second;
                if (fb <= r.first || fa >= r.second) { next.push_back(r); continue; }
                if (r.first < fa) next.push_back({ r.first, fa });
                if (fb < r.second) next.push_back({ fb, r.second });
            }
            todo.swap(next);
        }
        // expert slots already resident inside the range also need no new pages
        for (auto & Tp : tensors) {
            if (Tp->bufp != buf.get()) continue;
            for (uint32_t e = 0; e < Tp->n_expert; e++) {
                if (Tp->slots[e].state == VKS_ABSENT || Tp->slots[e].state == VKS_FIXED) continue;
                size_t sa = Tp->off + (size_t) e * Tp->nb02, sb = sa + Tp->nb02;
                std::vector<std::pair<size_t, size_t>> next;
                for (auto & r : todo) {
                    if (sb <= r.first || sa >= r.second) { next.push_back(r); continue; }
                    if (r.first < sa) next.push_back({ r.first, sa });
                    if (sb < r.second) next.push_back({ sb, r.second });
                }
                todo.swap(next);
            }
        }
        std::vector<vk_sparse_bind> binds;
        for (auto & r : todo) {
            size_t sz = r.second - r.first;
            if (!sz) continue;
            std::vector<vk_sparse_run> runs;
            if (!phys_alloc(sz, r.first, runs)) {
                GGML_ABORT("ggml_vulkan: sparse: pool exhausted pinning %zu bytes (budget %.1f GiB)", sz, cap / 1073741824.0);
            }
            for (auto & pr : runs) memset(run_ptr(pr), 0, (size_t) pr.n * VK_SPARSE_GRANULE);
            binds_for(buf->buffer, r.first, sz, runs, binds, false);
            buf->sparse->fixed.push_back({ r.first, sz });
            buf->sparse->fixed_runs.insert(buf->sparse->fixed_runs.end(), runs.begin(), runs.end());
        }
        bind_apply(binds);
    }

    void buffer_destroyed(vk_buffer_struct * b) {
        std::lock_guard<std::recursive_mutex> g(mu);
        // drop every tensor that lived in it
        for (size_t ti = 0; ti < tensors.size(); ti++) {
            auto & T = *tensors[ti];
            if (T.bufp != b) continue;
            for (uint32_t e = 0; e < T.n_expert; e++) {
                uint32_t k = VK_SPARSE_KEY((uint32_t) ti, e);
                if (T.slots[e].state == VKS_LRU || T.slots[e].state == VKS_TRANSIENT || T.slots[e].state == VKS_PROB || T.slots[e].state == VKS_LOADING) {
                    unlink_any(k);
                    phys_free(T.slots[e].runs);
                }
                T.slots[e].state = VKS_ABSENT;
            }
            T.bufp = nullptr;   // keep the entry so keys stay stable; it is inert now
            T.fd   = -1;
        }
        for (auto it = by_tensor.begin(); it != by_tensor.end();) {
            if (tensors[it->second]->bufp == nullptr) it = by_tensor.erase(it); else ++it;
        }
        phys_free(b->sparse->fixed_runs);
        b->sparse->fixed.clear();
    }

    // ---------------------------------------------------------------- kernel-log watcher

    void kmsg_main() {
        int pfd[2];
        if (pipe(pfd) != 0) return;
        pid_t pid = fork();
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGKILL);   // die with the parent: nothing else stops journalctl -f
            dup2(pfd[1], 1); close(pfd[0]); close(pfd[1]);
            int nul = open("/dev/null", O_WRONLY); if (nul >= 0) { dup2(nul, 2); close(nul); }
            execlp("journalctl", "journalctl", "-k", "-f", "-n", "0", "-o", "cat", (char *) nullptr);
            _exit(127);
        }
        close(pfd[1]);
        if (pid < 0) {
            close(pfd[0]);
            GGML_LOG_WARN("ggml_vulkan: sparse: cannot watch the kernel log; bind failures would go unnoticed\n");
            return;
        }
        kmsg_pid = pid; kmsg_fd = pfd[0];
        // raw read(), not stdio: a FILE blocked in fgets() holds its lock, and exit()'s
        // _IO_flush_all would then deadlock on it since this thread is never joined at exit
        std::string acc;
        char buf[512];
        for (;;) {
            ssize_t n = read(pfd[0], buf, sizeof buf);
            if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
            acc.append(buf, (size_t) n);
            size_t nl;
            while ((nl = acc.find('\n')) != std::string::npos) {
                std::string line = acc.substr(0, nl);
                acc.erase(0, nl + 1);
                if (line.find("Couldn't update BO_VA") != std::string::npos || line.find("Not enough memory for command submission") != std::string::npos) {
                    kmsg_errors++;
                }
            }
        }
        close(pfd[0]);
    }

    // Re-issue the binding of every resident slab. Called at a stall (GPU idle) after the
    // kernel reported a failed page-table update, whose victim we cannot identify.
    void rebind_all() {
        std::vector<vk_sparse_bind> binds;
        size_t n = 0;
        for (auto & Tp : tensors) {
            if (!Tp->bufp) continue;
            for (uint32_t e = 0; e < Tp->n_expert; e++) {
                auto & sl = Tp->slots[e];
                if (sl.state == VKS_ABSENT || sl.state == VKS_FIXED || sl.state == VKS_LOADING || sl.runs.empty()) continue;
                binds_for(Tp->vkbuf, Tp->off + (size_t) e * Tp->nb02, Tp->nb02, sl.runs, binds, false);
                n++;
            }
        }
        // bind in slices so one vkQueueBindSparse does not carry ~20k entries
        const size_t SLICE = 4096;
        for (size_t i = 0; i < binds.size(); i += SLICE) {
            std::vector<vk_sparse_bind> part(binds.begin() + i, binds.begin() + std::min(binds.size(), i + SLICE));
            bind_apply(part);
        }
        n_rebind++;
        GGML_LOG_WARN("ggml_vulkan: sparse: kernel reported %d failed page-table update(s); re-bound %zu resident slabs\n", kmsg_errors.load() - kmsg_seen, n);
        kmsg_seen = kmsg_errors.load();
    }

    // ---------------------------------------------------------------- warm set

    void load_warm_file(const char * path) {
        FILE * f = fopen(path, "r");
        if (!f) { GGML_LOG_WARN("ggml_vulkan: sparse warm: cannot open %s\n", path); return; }
        const char * ns = getenv("GGML_VK_SPARSE_WARM_N");
        size_t limit = ns ? (size_t) atoll(ns) : SIZE_MAX;
        char line[128];
        while (warm.size() < limit && fgets(line, sizeof line, f)) {
            int L, E;
            if (line[0] == '#') continue;
            if (sscanf(line, "%d %d", &L, &E) == 2) warm.push_back({ L, E });
        }
        fclose(f);
        fprintf(stderr, "ggml_vulkan: sparse warm: %zu (layer, expert) keys queued from %s\n", warm.size(), path);
    }

    // Called once the loader has registered every tensor. Admits the warm set COLDEST-FIRST on
    // a background thread so the cap trims the coldest entries, exactly like the CPU cache.
    void load_done() {
        if (warm.empty() || warm_started.exchange(true)) return;
        warm_thread = std::thread([this] { warm_main(); });
    }

    void warm_main() {
        std::map<int, std::vector<uint32_t>> by_layer;
        {
            std::lock_guard<std::recursive_mutex> g(mu);
            for (uint32_t ti = 0; ti < tensors.size(); ti++) {
                if (tensors[ti]->bufp && tensors[ti]->layer >= 0 && tensors[ti]->fd >= 0) by_layer[tensors[ti]->layer].push_back(ti);
            }
        }
        // keep only the hottest prefix that fits, then admit it coldest-first
        size_t n_fit = 0, bytes = 0;
        for (; n_fit < warm.size(); n_fit++) {
            auto it = by_layer.find(warm[n_fit].first);
            if (it == by_layer.end()) continue;
            size_t b = 0;
            for (uint32_t ti : it->second) b += tensors[ti]->nb02;
            if (bytes + b > lru_cap) break;
            bytes += b;
        }
        warm.resize(n_fit);
        size_t admitted = 0, skipped = 0;
        const size_t BATCH = 8;   // a bind burst can lose the page-table eviction race in VRAM
        for (size_t i = warm.size(); i > 0;) {
            std::lock_guard<std::recursive_mutex> g(mu);
            std::vector<vk_sparse_bind> binds;
            size_t n = 0;
            while (i > 0 && n < BATCH) {
                --i;
                auto it = by_layer.find(warm[i].first);
                if (it == by_layer.end()) { skipped++; continue; }
                for (uint32_t ti : it->second) {
                    auto & T = *tensors[ti];
                    uint32_t e = (uint32_t) warm[i].second;
                    if (!T.bufp || e >= T.n_expert || T.slots[e].state != VKS_ABSENT) continue;
                    auto & s = T.slots[e];
                    if (!make_room(false, T.nb02, T.off + (size_t) e * T.nb02, s.runs, binds)) { i = 0; break; }
                    read_slab(T, e, s.runs);
                    binds_for(T.vkbuf, T.off + (size_t) e * T.nb02, T.nb02, s.runs, binds, false);
                    s.state = VKS_LRU; list_push(VK_SPARSE_KEY(ti, e), lru_head, lru_tail); lru_bytes += T.nb02;
                    admitted++; n++;
                }
            }
            bind_apply(binds);
            read_drain();
        }
        fprintf(stderr, "ggml_vulkan: sparse warm: admitted %zu slabs (%zu keys without a tensor), resident %.1f/%.1f GiB\n",
                      admitted, skipped, lru_bytes / 1073741824.0, lru_cap / 1073741824.0);
        {
            std::lock_guard<std::recursive_mutex> g(mu);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let the journal catch up
            if (kmsg_errors.load() != kmsg_seen) rebind_all();
        }
    }

    void dump_ref() {
        FILE * f = fopen(dump_path, "w");
        if (!f) return;
        std::vector<std::pair<uint32_t, uint64_t>> v;
        for (auto & kv : ref) v.push_back({ kv.second, kv.first });
        std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.first > b.first; });
        fprintf(f, "# layer expert touches | GGML_VK_SPARSE_DUMP -> replay with GGML_VK_SPARSE_WARM\n");
        for (auto & p : v) fprintf(f, "%u %u %u\n", (uint32_t) (p.second >> 32), (uint32_t) p.second, p.first);
        fclose(f);
    }

    void maybe_print() {
        if (!stats) return;
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now - t_last < 10.0) return;
        t_last = now;
        char pfs[256]; learn_stats(pfs, sizeof pfs);
        fprintf(stderr, "[vk-sparse] decode hit %llu miss %llu (%.1f%%) | scan hit %llu miss %llu (%.1f%%) | evict %llu calls %llu (scan %llu) stalls %llu rebinds %llu | read %.2f GiB bind %.1f ms read %.1f ms ensure %.1f ms | lru %.1f/%.1f transient %.1f/%.1f GiB pool %.1f%s\n",
                      (unsigned long long) n_hit, (unsigned long long) n_miss, 100.0 * n_miss / (double) (n_hit + n_miss + 1),
                      (unsigned long long) n_scan_hit, (unsigned long long) n_scan_miss, 100.0 * n_scan_miss / (double) (n_scan_hit + n_scan_miss + 1),
                      (unsigned long long) n_evict, (unsigned long long) n_calls, (unsigned long long) n_scan_calls,
                      (unsigned long long) n_stall, (unsigned long long) n_rebind, bytes_read / 1073741824.0, t_bind_us / 1000.0, t_read_us / 1000.0, t_ensure_us / 1000.0,
                      lru_bytes / 1073741824.0, lru_cap / 1073741824.0, tr_bytes / 1073741824.0, transient_cap / 1073741824.0,
                      chunk_bytes / 1073741824.0, pfs);
    }
};

void vk_sparse_rpool::main() {
    // O_DIRECT needs 4 KiB-aligned file offsets and memory; file offsets are only 32 B
    // aligned, so read the aligned superset into a bounce buffer and copy the slab out
    const size_t bounce_sz = 16u << 20;
    uint8_t * bounce = nullptr;
    if (posix_memalign((void **) &bounce, 4096, bounce_sz) != 0) bounce = nullptr;
    for (;;) {
        vk_sparse_job j;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [this] { return stop || !jobs.empty(); });
            if (stop && jobs.empty()) break;
            j = jobs.front();
            jobs.pop_front();
        }
        if (j.direct && bounce && j.len + 8192 <= bounce_sz) {
            const off_t  a   = j.off & ~(off_t) 4095;
            const size_t pre = (size_t) (j.off - a);
            const size_t tot = (pre + j.len + 4095) & ~(size_t) 4095;
            size_t done = 0;
            while (done < tot) {
                ssize_t n = pread(j.fd, bounce + done, tot - done, a + (off_t) done);
                if (n <= 0) {
                    if (n < 0 && errno == EINTR) continue;
                    if (n == 0 && done >= pre + j.len) break;   // superset ran past EOF, slab complete
                    GGML_LOG_ERROR("ggml_vulkan: sparse O_DIRECT pread(fd %d, off %lld, len %zu) failed: %s\n",
                                   j.fd, (long long) a, tot, n < 0 ? strerror(errno) : "EOF");
                    err.store(1);
                    break;
                }
                done += (size_t) n;
            }
            memcpy(j.dst, bounce + pre, j.len);
            std::lock_guard<std::mutex> lk(mu);
            if (j.ctr) j.ctr->fetch_sub(1);
            any.notify_all();
            if (--inflight == 0) dcv.notify_all();
            continue;
        }
        size_t done = 0;
        while (done < j.len) {
            ssize_t n = pread(j.fd, j.dst + done, j.len - done, j.off + (off_t) done);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                GGML_LOG_ERROR("ggml_vulkan: sparse pread(fd %d, off %lld, len %zu) failed: %s\n",
                               j.fd, (long long) j.off, j.len, n < 0 ? strerror(errno) : "EOF");
                err.store(1);
                break;
            }
            done += (size_t) n;
        }
        // Drop the page cache copy: with ~60 GiB pinned in GTT, a full page cache starves
        // amdgpu's own page allocation ("Not enough memory for command submission").
        {
            off_t a = j.off & ~(off_t) (VK_SPARSE_PAGE - 1);
            off_t b = (j.off + (off_t) j.len + (off_t) VK_SPARSE_PAGE - 1) & ~(off_t) (VK_SPARSE_PAGE - 1);
            posix_fadvise(j.fd, a, b - a, POSIX_FADV_DONTNEED);
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            if (j.ctr) j.ctr->fetch_sub(1);
            any.notify_all();
            if (--inflight == 0) dcv.notify_all();
        }
    }
    free(bounce);
}


#include "ggml-vk-sparse-learn.inc"


// ---------------------------------------------------------------- buffer creation

static vk_buffer ggml_vk_create_sparse_buffer(vk_device & device, size_t size) {
    GGML_ASSERT(device->sparse && "sparse binding not supported on this device");
    if (size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested sparse buffer size exceeds device buffer size limit");
    }
    vk_buffer buf = std::make_shared<vk_buffer_struct>();
    if (size == 0) {
        buf->size = 0;
        return buf;
    }
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    if (device->buffer_device_address) {
        usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    vk::BufferCreateInfo bci(vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
                             size, usage, vk::SharingMode::eExclusive, 0, nullptr);
    buf->buffer = device->device.createBuffer(bci);

    vk::MemoryRequirements mr = device->device.getBufferMemoryRequirements(buf->buffer);
    if (mr.alignment > VK_SPARSE_PAGE || (VK_SPARSE_PAGE % mr.alignment) != 0) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::SystemError(vk::make_error_code(vk::Result::eErrorFeatureNotPresent),
                              "sparse block alignment is not 4 KiB-compatible");
    }
    auto & mgr = *device->sparse;
    if (mgr.memtype == UINT32_MAX) {
        // DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT, preferring a type without AMD's uncached bit
        auto mp = device->physical_device.getMemoryProperties();
        const auto want = vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        int best = -1, best_score = -1;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if (!(mr.memoryTypeBits & (1u << i)) || ggml_vk_memtype_skipped(i)) continue;
            auto f = mp.memoryTypes[i].propertyFlags;
            if ((f & want) != want) continue;
            int score = (f & vk::MemoryPropertyFlagBits::eDeviceUncachedAMD) ? 0 : 1;
            if (score > best_score) { best = (int) i; best_score = score; }
        }
        if (best < 0) {
            device->device.destroyBuffer(buf->buffer);
            throw vk::SystemError(vk::make_error_code(vk::Result::eErrorFeatureNotPresent),
                                  "no host-visible device-local memory type can back sparse buffers");
        }
        mgr.memtype = (uint32_t) best;
        fprintf(stderr, "ggml_vulkan: sparse experts: memtype %u, pool %.1f GiB (transient %.1f), 64 KiB granules\n",
                      mgr.memtype, mgr.cap / 1073741824.0, mgr.transient_cap / 1073741824.0);
    }
    buf->memory_property_flags = {};   // not host-mappable as a whole: writes go through staging
    buf->ptr    = nullptr;
    buf->device = device;
    buf->size   = size;
    buf->sparse = new vk_sparse_buf();
    if (device->buffer_device_address) {
        buf->bda_addr = device->device.getBufferAddress(vk::BufferDeviceAddressInfo(buf->buffer));
    }
    device->memory_logger->log_allocation(buf, size);
    return buf;
}

// called from ~vk_buffer_struct before the VkBuffer is destroyed
static void ggml_vk_sparse_buffer_destroyed(vk_buffer_struct * b) {
    if (b->device && b->device->sparse) {
        b->device->sparse->buffer_destroyed(b);
    }
    delete b->sparse;
    b->sparse = nullptr;
}

static bool ggml_vk_sparse_is_sparse_buffer(ggml_backend_buffer_t buffer) {
    if (!buffer || !ggml_backend_buffer_is_vk(buffer)) return false;
    return ((ggml_backend_vk_buffer_context *) buffer->context)->dev_buffer->sparse != nullptr;
}

// ---------------------------------------------------------------- buffer interface

static void ggml_backend_vk_sparse_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (size == 0) return;
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) buffer->context;
    buf_ctx->dev_buffer->device->sparse->pin_range(buf_ctx->dev_buffer, vk_tensor_offset(tensor) + tensor->view_offs + offset, size);
    ggml_backend_vk_buffer_memset_tensor(buffer, tensor, value, offset, size);
}

static void ggml_backend_vk_sparse_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (size == 0) return;
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) buffer->context;
    buf_ctx->dev_buffer->device->sparse->pin_range(buf_ctx->dev_buffer, vk_tensor_offset(tensor) + tensor->view_offs + offset, size);
    ggml_backend_vk_buffer_set_tensor(buffer, tensor, data, offset, size);
}

static void ggml_backend_vk_sparse_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                                        size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (size == 0 || n_copies == 0) return;
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) buffer->context;
    buf_ctx->dev_buffer->device->sparse->pin_range(buf_ctx->dev_buffer, vk_tensor_offset(tensor) + tensor->view_offs + offset,
                                                   (n_copies - 1) * stride_tensor + size);
    ggml_backend_vk_buffer_set_tensor_2d(buffer, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static bool ggml_backend_vk_sparse_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_nbytes(src) == 0) return true;
    if (!ggml_backend_buffer_is_vk(src->buffer)) return false;
    ggml_backend_vk_buffer_context * dst_ctx = (ggml_backend_vk_buffer_context *) dst->buffer->context;
    dst_ctx->dev_buffer->device->sparse->pin_range(dst_ctx->dev_buffer, vk_tensor_offset(dst) + dst->view_offs, ggml_nbytes(src));
    return ggml_backend_vk_buffer_cpy_tensor(buffer, src, dst);
}

static void ggml_backend_vk_sparse_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *) buffer->context;
    ctx->dev_buffer->device->sparse->pin_range(ctx->dev_buffer, 0, buffer->size);
    ggml_backend_vk_buffer_clear(buffer, value);
}

static ggml_backend_buffer_i ggml_backend_vk_sparse_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_sparse_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_sparse_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,      // unbound pages read as zeros
    /* .set_tensor_2d   = */ ggml_backend_vk_sparse_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_sparse_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_sparse_buffer_clear,
    /* .reset           = */ NULL,
};

// ---------------------------------------------------------------- buffer type

static ggml_backend_buffer_t ggml_backend_vk_sparse_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    vk_buffer dev_buffer = nullptr;
    try {
        dev_buffer = ggml_vk_create_sparse_buffer(ctx->device, size);
    } catch (const vk::SystemError & e) {
        GGML_LOG_ERROR("ggml_vulkan: sparse buffer allocation of %zu bytes failed: %s\n", size, e.what());
        return nullptr;
    }
    ggml_backend_vk_buffer_context * bufctx = new ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);
    return ggml_backend_buffer_init(buft, ggml_backend_vk_sparse_buffer_interface, bufctx, size);
}

static size_t ggml_backend_vk_sparse_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return std::max<size_t>(VK_SPARSE_PAGE, ctx->device->properties.limits.minStorageBufferOffsetAlignment);
}

static size_t ggml_backend_vk_sparse_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->max_buffer_size;
}

// Same get_name as the device buffer type on purpose: that function pointer is how
// ggml-vulkan recognises its own buffers (supports_buft, is_vk, every context cast).
static ggml_backend_buffer_type_i ggml_backend_vk_sparse_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_vk_sparse_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_vk_sparse_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_vk_sparse_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

// ---------------------------------------------------------------- public API

ggml_backend_buffer_type_t ggml_backend_vk_sparse_buffer_type(size_t dev_num) {
    ggml_vk_instance_init();
    vk_device dev = ggml_vk_get_device(dev_num);
    return dev->sparse ? &dev->sparse_buffer_type : nullptr;
}

bool ggml_backend_buft_is_vk_sparse(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.alloc_buffer == ggml_backend_vk_sparse_buffer_type_alloc_buffer;
}

bool ggml_backend_vk_sparse_set_source(struct ggml_tensor * tensor, int fd, size_t file_offset) {
    if (!tensor || !tensor->buffer || !ggml_vk_sparse_is_sparse_buffer(tensor->buffer)) {
        return false;
    }
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *) tensor->buffer->context;
    return ctx->dev_buffer->device->sparse->set_source(tensor, fd, file_offset);
}

void ggml_backend_vk_sparse_load_done(ggml_backend_dev_t dev) {
    if (!dev || dev->iface.get_name != ggml_backend_vk_device_get_name) return;
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *) dev->context;
    vk_device device = ggml_vk_get_device(ctx->device);
    if (device->sparse) device->sparse->load_done();
}

static ggml_backend_buffer_type_t * ggml_backend_vk_device_get_extra_bufts(ggml_backend_dev_t dev) {
    static std::vector<std::vector<ggml_backend_buffer_type_t>> lists(GGML_VK_MAX_DEVICES);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *) dev->context;
    auto & l = lists[ctx->device];
    if (l.empty()) {
        if (auto * buft = ggml_backend_vk_sparse_buffer_type(ctx->device)) {
            l.push_back(buft);
        }
        l.push_back(nullptr);
    }
    return l.data();
}

// ---------------------------------------------------------------- glue for ggml-vulkan.cpp

static void ggml_vk_sparse_init(std::shared_ptr<vk_device_struct> & device) {
    if (!device->sparse_binding || getenv("GGML_VK_SPARSE_DISABLE")) {
        return;
    }
    // needs a queue family with sparse binding; ggml uses the compute family for everything
    auto qprops = device->physical_device.getQueueFamilyProperties();
    const uint32_t qf = device->compute_queue->queue_family_index;
    if (qf >= qprops.size() || !(qprops[qf].queueFlags & vk::QueueFlagBits::eSparseBinding)) {
        return;
    }
    device->sparse = new vk_sparse_mgr(device.get());
    device->sparse_buffer_type = {
        /* .iface    = */ ggml_backend_vk_sparse_buffer_type_interface,
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), device->idx),
        /* .context  = */ new ggml_backend_vk_buffer_type_context{ device->name + "_Sparse", device },
    };
}

static void ggml_vk_sparse_destroy(vk_device_struct * device) {
    delete device->sparse;
    device->sparse = nullptr;
}

static void ggml_vk_sparse_begin_graph(vk_device_struct * device) {
    device->sparse->begin_graph();
}

static void ggml_vk_sparse_ensure(vk_device_struct * device, const ggml_tensor * t, const int32_t * ids, size_t n_ids, size_t n_tokens) {
    device->sparse->ensure(t, ids, n_ids, n_tokens);
}

static void ggml_vk_sparse_ensure_many(vk_device_struct * device, const ggml_tensor * const * ts, size_t n_t, const int32_t * ids, size_t n_ids, size_t n_tokens) {
    device->sparse->ensure_many(ts, n_t, ids, n_ids, n_tokens);
}

static void ggml_vk_sparse_note_stall(vk_device_struct * device) {
    device->sparse->n_stall++;
    device->sparse->release_pins();
}
