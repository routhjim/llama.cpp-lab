#include "state-file.h"

#include "log.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <malloc.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static constexpr size_t STATE_FILE_BUF   = 4u*1024*1024;
static constexpr size_t STATE_FILE_ALIGN = 4096;

namespace {

struct aligned_buf {
    uint8_t * ptr = nullptr;

    aligned_buf() {
#ifdef _WIN32
        ptr = (uint8_t *) _aligned_malloc(STATE_FILE_BUF, STATE_FILE_ALIGN);
#else
        void * p = nullptr;
        if (posix_memalign(&p, STATE_FILE_ALIGN, STATE_FILE_BUF) == 0) {
            ptr = (uint8_t *) p;
        }
#endif
    }

    ~aligned_buf() {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

#ifdef _WIN32
static int64_t pwrite_raw(int fd, const void * src, size_t n, uint64_t off) {
    if (_lseeki64(fd, (__int64) off, SEEK_SET) < 0) {
        return -1;
    }
    return _write(fd, src, (unsigned int) n);
}

static int64_t pread_raw(int fd, void * dst, size_t n, uint64_t off) {
    if (_lseeki64(fd, (__int64) off, SEEK_SET) < 0) {
        return -1;
    }
    return _read(fd, dst, (unsigned int) n);
}
#else
static int64_t pwrite_raw(int fd, const void * src, size_t n, uint64_t off) {
    ssize_t r;
    do {
        r = ::pwrite(fd, src, n, (off_t) off);
    } while (r < 0 && errno == EINTR);
    return r;
}

static int64_t pread_raw(int fd, void * dst, size_t n, uint64_t off) {
    ssize_t r;
    do {
        r = ::pread(fd, dst, n, (off_t) off);
    } while (r < 0 && errno == EINTR);
    return r;
}

static void raise_fd_limit() {
    // each checkpoint keeps one file open
    static std::once_flag once;
    std::call_once(once, [] {
        struct rlimit rl;
        if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < rl.rlim_max) {
            rl.rlim_cur = rl.rlim_max;
            setrlimit(RLIMIT_NOFILE, &rl);
        }
    });
}
#endif

} // namespace

struct common_state_file::writer {
    common_state_file & f;
    aligned_buf buf;
    size_t   fill = 0;
    uint64_t off  = 0;
    size_t   total = 0;

    explicit writer(common_state_file & f) : f(f) {}

    bool flush() {
        size_t n = fill;
        if (f.direct) {
            n = (n + STATE_FILE_ALIGN - 1) / STATE_FILE_ALIGN * STATE_FILE_ALIGN;
            memset(buf.ptr + fill, 0, n - fill);
        }
        for (size_t done = 0; done < n; ) {
            const int64_t r = pwrite_raw(f.fd, buf.ptr + done, n - done, off + done);
#ifdef O_DIRECT
            if (r < 0 && errno == EINVAL && f.direct) {
                // the filesystem accepted O_DIRECT at open but refuses the I/O
                fcntl(f.fd, F_SETFL, fcntl(f.fd, F_GETFL) & ~O_DIRECT);
                f.direct = false;
                continue;
            }
#endif
            if (r <= 0) {
                LOG_ERR("%s: write failed: %s\n", __func__, strerror(errno));
                return false;
            }
            done += (size_t) r;
        }
        off  += n;
        fill  = 0;
        return true;
    }

    bool put(const void * src, size_t n) {
        if (buf.ptr == nullptr) {
            return false;
        }
        const uint8_t * p = (const uint8_t *) src;
        while (n > 0) {
            const size_t k = std::min(n, STATE_FILE_BUF - fill);
            memcpy(buf.ptr + fill, p, k);
            fill  += k;
            total += k;
            p     += k;
            n     -= k;
            if (fill == STATE_FILE_BUF && !flush()) {
                return false;
            }
        }
        return true;
    }

    bool finish() {
        if (fill > 0 && !flush()) {
            return false;
        }
#ifndef _WIN32
        if (ftruncate(f.fd, (off_t) off) != 0) {
            LOG_WRN("%s: ftruncate failed: %s\n", __func__, strerror(errno));
        }
#if defined(POSIX_FADV_DONTNEED)
        if (!f.direct) {
            // drop the written pages from the page cache
            fdatasync(f.fd);
            posix_fadvise(f.fd, 0, 0, POSIX_FADV_DONTNEED);
        }
#endif
#endif
        f.n_bytes = total;
        return true;
    }

    static bool cb(const void * src, size_t size, void * user_data) {
        return ((writer *) user_data)->put(src, size);
    }
};

struct common_state_file::reader {
    const common_state_file & f;
    aligned_buf buf;
    size_t   pos  = 0;
    size_t   len  = 0;
    uint64_t off  = 0;

    explicit reader(const common_state_file & f) : f(f) {}

    bool refill() {
        if (off >= f.n_bytes) {
            return false;
        }
        const int64_t r = pread_raw(f.fd, buf.ptr, STATE_FILE_BUF, off);
        if (r <= 0) {
            LOG_ERR("%s: read failed: %s\n", __func__, r < 0 ? strerror(errno) : "unexpected end of file");
            return false;
        }
        len  = std::min<size_t>((size_t) r, f.n_bytes - off);
        off += (size_t) r;
        pos  = 0;
        return true;
    }

    bool get(void * dst, size_t n) {
        if (buf.ptr == nullptr) {
            return false;
        }
        uint8_t * p = (uint8_t *) dst;
        while (n > 0) {
            if (pos == len && !refill()) {
                return false;
            }
            const size_t k = std::min(n, len - pos);
            memcpy(p, buf.ptr + pos, k);
            pos += k;
            p   += k;
            n   -= k;
        }
        return true;
    }

    static bool cb(void * dst, size_t size, void * user_data) {
        return ((reader *) user_data)->get(dst, size);
    }
};

std::shared_ptr<common_state_file> common_state_file::create(const std::string & dir) {
    std::shared_ptr<common_state_file> res(new common_state_file());

#ifdef _WIN32
    static std::atomic<uint64_t> counter{0};
    const std::string path = dir + "\\llama-state-" + std::to_string(_getpid()) + "-" + std::to_string(counter++);
    res->fd = _open(path.c_str(), _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY | _O_TEMPORARY, _S_IREAD | _S_IWRITE);
#else
    raise_fd_limit();
#ifdef O_TMPFILE
    res->fd = open(dir.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
#endif
    if (res->fd < 0) {
        std::string tmpl = dir + "/llama-state-XXXXXX";
        res->fd = mkstemp(tmpl.data());
        if (res->fd >= 0) {
            unlink(tmpl.c_str());
            fcntl(res->fd, F_SETFD, FD_CLOEXEC);
        }
    }
#ifdef O_DIRECT
    if (res->fd >= 0 && fcntl(res->fd, F_SETFL, fcntl(res->fd, F_GETFL) | O_DIRECT) == 0) {
        res->direct = true;
    }
#endif
#endif

    if (res->fd < 0) {
        LOG_ERR("%s: cannot create a state file in '%s': %s\n", __func__, dir.c_str(), strerror(errno));
        return nullptr;
    }

    return res;
}

common_state_file::~common_state_file() {
    if (fd >= 0) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
    }
}

bool common_state_file::save(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    n_bytes = 0;

    writer w(*this);
    const size_t n = llama_state_seq_get_data_stream(ctx, writer::cb, &w, seq_id, flags);

    return n > 0 && n == w.total && w.finish();
}

bool common_state_file::load(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    reader r(*this);

    return llama_state_seq_set_data_stream(ctx, reader::cb, &r, n_bytes, seq_id, flags) == n_bytes;
}

bool common_state_file::assign(const void * data, size_t size) {
    n_bytes = 0;

    writer w(*this);

    return w.put(data, size) && w.finish();
}

bool common_state_file::read(const std::function<bool(const void * data, size_t size)> & fn) const {
    reader r(*this);
    if (r.buf.ptr == nullptr) {
        return false;
    }
    while (r.off < n_bytes) {
        if (!r.refill() || !fn(r.buf.ptr, r.len)) {
            return false;
        }
    }
    return true;
}
