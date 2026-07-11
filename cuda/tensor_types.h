#ifndef FUCINA_TENSOR_TYPES_H
#define FUCINA_TENSOR_TYPES_H

#include <stdint.h>
#include <type_traits>

// Runtime encoding and layout are properties of an individual physical weight, not of the model.
// Keep these descriptors POD: CUDA graph hot paths read them directly without allocation or lookup.
enum class WeightEncoding : uint8_t {
    F32,
    BF16,
    Q8_0,
    Q4_0,
    Q4_K,
    Q6_K,
    FP8_BLOCK_128,
    FP8_ROW,
    NVFP4_LINEAR,
    NVFP4_SWIZZLED,
};

enum class TensorLayout : uint8_t {
    ROW_MAJOR,
    GGML_NATIVE,
    Q4K_PACKED,
    NVFP4_SCALE_LINEAR,
    NVFP4_SCALE_SWIZZLED,
};

enum WeightFlags : uint16_t {
    WEIGHT_FLAG_NONE    = 0,
    WEIGHT_FLAG_TIED    = 1u << 0,
    WEIGHT_FLAG_PACKED  = 1u << 1,
    WEIGHT_FLAG_PRIMARY = 1u << 2,
    WEIGHT_FLAG_CACHE   = 1u << 3,
    WEIGHT_FLAG_GROUPED = 1u << 4,
};

struct WeightRef {
    const uint8_t *data;
    const void *scale;
    const float *global_scale;
    int32_t out_dim;
    int32_t in_dim;
    WeightEncoding encoding;
    TensorLayout layout;
    uint16_t flags;
};

struct ExpertWeightRef {
    WeightRef weight;
    int32_t expert_count;
    int64_t weight_stride;
    int64_t scale_stride;
};

enum class WorkspaceKind : uint8_t {
    DECODE,
    PREFILL,
    MOE,
    ATTENTION,
    RECURRENT_STATE,
    KV_CACHE,
};

struct WorkspaceRef {
    uint8_t *data;
    uint64_t bytes;
    uint32_t alignment;
    WorkspaceKind kind;
    uint8_t flags;
    uint16_t reserved;
};

static_assert(std::is_trivially_copyable<WeightRef>::value, "WeightRef must remain trivially copyable");
static_assert(std::is_trivially_copyable<ExpertWeightRef>::value,
              "ExpertWeightRef must remain trivially copyable");
static_assert(std::is_trivially_copyable<WorkspaceRef>::value,
              "WorkspaceRef must remain trivially copyable");

#endif  // FUCINA_TENSOR_TYPES_H
