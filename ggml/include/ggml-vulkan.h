#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

// Sparse expert buffers: a buffer type whose memory is bound on demand, per expert, by a
// CPU-managed residency cache (see ggml-vk-sparse.inc). Select it for MoE expert tensors
// with -ot 'exps=Vulkan0_Sparse'. NULL if the device has no sparse binding support.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_sparse_buffer_type(size_t dev_num);
GGML_BACKEND_API bool ggml_backend_buft_is_vk_sparse(ggml_backend_buffer_type_t buft);
// Register where a sparse tensor's bytes live on disk instead of copying them in. Returns
// false if the tensor is not in a sparse buffer. Also reachable through the backend registry
// as "ggml_backend_sparse_set_source" so llama.cpp does not need to link this backend.
GGML_BACKEND_API bool ggml_backend_vk_sparse_set_source(struct ggml_tensor * tensor, int fd, size_t file_offset);
// All sources registered: start admitting the warm set (GGML_VK_SPARSE_WARM), if any.
// Registry name: "ggml_backend_sparse_load_done".
GGML_BACKEND_API void ggml_backend_vk_sparse_load_done(ggml_backend_dev_t dev);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
