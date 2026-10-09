#pragma once

#include "llama.h"

#include <functional>
#include <memory>
#include <string>

// a sequence state kept in an unnamed file instead of host memory
// the state streams through a 4 MiB buffer, and O_DIRECT (where supported) keeps it out of the page cache
// the file is removed when the last reference is dropped, or when the process ends
struct common_state_file {
    // returns nullptr on failure
    static std::shared_ptr<common_state_file> create(const std::string & dir);

    ~common_state_file();

    common_state_file(const common_state_file &) = delete;
    common_state_file & operator=(const common_state_file &) = delete;

    // replace the contents with the state of seq_id
    bool save(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags);

    bool load(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) const;

    // replace the contents with the given bytes
    bool assign(const void * data, size_t size);

    // pass the contents to fn in order, one piece at a time
    bool read(const std::function<bool(const void * data, size_t size)> & fn) const;

    size_t size() const { return n_bytes; }

    struct writer;
    struct reader;

    int    fd      = -1;
    bool   direct  = false; // O_DIRECT: all I/O is block-aligned
    size_t n_bytes = 0;

private:
    common_state_file() = default;
};
