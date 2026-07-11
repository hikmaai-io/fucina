#ifndef FUCINA_MODEL_PLAN_H
#define FUCINA_MODEL_PLAN_H

#include "tensor_types.h"

#include <stdint.h>
#include <stddef.h>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

enum class TensorTransform : uint8_t {
    COPY,
    BF16_TO_F32,
    FP8_TO_Q4K,
    FP8_TO_NVFP4,
    NVFP4_REBASE,
    BF16_TO_Q8_0,
    QUANT_TO_BF16,
};

enum class AllocationClass : uint8_t {
    CORE_WEIGHTS,
    SCALES,
    EXPERT_SLABS,
    EMBEDDING_HEAD,
    PREFILL_CACHE,
    WORKSPACE,
};

struct SourceTensorSpec {
    std::string logical_name;
    std::string source_name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t bytes = 0;
};

struct PlannedTensor {
    uint32_t id = 0;
    SourceTensorSpec source;
    TensorTransform transform = TensorTransform::COPY;
    WeightEncoding destination = WeightEncoding::F32;
    AllocationClass arena = AllocationClass::CORE_WEIGHTS;
    std::string consumer;
    size_t bytes = 0;
    size_t alignment = 1;
    size_t arena_offset = 0;
    int32_t aliases = -1;
};

class ModelPlan {
public:
    bool add(PlannedTensor tensor, std::string &error) {
        if (finalized_) { error = "cannot add to finalized model plan"; return false; }
        if (tensor.source.logical_name.empty()) { error = "planned tensor has no logical name"; return false; }
        if (tensor.bytes == 0 && tensor.aliases < 0) {
            error = tensor.source.logical_name + ": zero destination bytes"; return false;
        }
        if (tensor.alignment == 0 || (tensor.alignment & (tensor.alignment - 1)) != 0) {
            error = tensor.source.logical_name + ": alignment is not a power of two"; return false;
        }
        for (int64_t dim : tensor.source.shape) if (dim <= 0) {
            error = tensor.source.logical_name + ": non-positive source dimension"; return false;
        }
        tensor.id = (uint32_t)tensors_.size();
        tensors_.push_back(std::move(tensor));
        return true;
    }

    bool finalize(std::string &error) {
        if (finalized_) return true;
        size_t cursor[(int)AllocationClass::WORKSPACE + 1] = {};
        for (PlannedTensor &tensor : tensors_) {
            if (tensor.aliases >= 0) {
                if ((uint32_t)tensor.aliases >= tensor.id) {
                    error = tensor.source.logical_name + ": alias must reference an earlier tensor";
                    return false;
                }
                tensor.arena_offset = tensors_[(size_t)tensor.aliases].arena_offset;
                continue;
            }
            size_t &at = cursor[(int)tensor.arena];
            if (at > std::numeric_limits<size_t>::max() - (tensor.alignment - 1)) {
                error = tensor.source.logical_name + ": alignment overflow"; return false;
            }
            at = (at + tensor.alignment - 1) & ~(tensor.alignment - 1);
            if (at > std::numeric_limits<size_t>::max() - tensor.bytes) {
                error = tensor.source.logical_name + ": arena size overflow"; return false;
            }
            tensor.arena_offset = at;
            at += tensor.bytes;
        }
        for (int i = 0; i <= (int)AllocationClass::WORKSPACE; ++i) totals_[i] = cursor[i];
        finalized_ = true;
        return true;
    }

    const std::vector<PlannedTensor> &tensors() const { return tensors_; }
    size_t bytes(AllocationClass arena) const { return totals_[(int)arena]; }
    bool finalized() const { return finalized_; }

    std::string json() const {
        std::ostringstream out;
        out << "{\"version\":1,\"finalized\":" << (finalized_ ? "true" : "false") << ",\"totals\":{";
        for (int i = 0; i <= (int)AllocationClass::WORKSPACE; ++i) {
            if (i) out << ',';
            out << '\"' << allocation_name((AllocationClass)i) << "\":" << totals_[i];
        }
        out << "},\"tensors\":[";
        for (size_t i = 0; i < tensors_.size(); ++i) {
            const PlannedTensor &t = tensors_[i];
            if (i) out << ',';
            out << "{\"id\":" << t.id
                << ",\"logical_name\":\"" << escape(t.source.logical_name)
                << "\",\"source_name\":\"" << escape(t.source.source_name)
                << "\",\"dtype\":\"" << escape(t.source.dtype) << "\",\"shape\":[";
            for (size_t d = 0; d < t.source.shape.size(); ++d) { if (d) out << ','; out << t.source.shape[d]; }
            out << "],\"source_bytes\":" << t.source.bytes
                << ",\"transform\":\"" << transform_name(t.transform)
                << "\",\"destination\":\"" << encoding_name(t.destination)
                << "\",\"arena\":\"" << allocation_name(t.arena)
                << "\",\"consumer\":\"" << escape(t.consumer)
                << "\",\"bytes\":" << t.bytes << ",\"alignment\":" << t.alignment
                << ",\"offset\":" << t.arena_offset << ",\"aliases\":" << t.aliases << '}';
        }
        out << "]}";
        return out.str();
    }

    static const char *encoding_name(WeightEncoding v) {
        switch (v) {
            case WeightEncoding::F32: return "F32"; case WeightEncoding::BF16: return "BF16";
            case WeightEncoding::Q8_0: return "Q8_0"; case WeightEncoding::Q4_0: return "Q4_0";
            case WeightEncoding::Q4_K: return "Q4_K"; case WeightEncoding::Q6_K: return "Q6_K";
            case WeightEncoding::FP8_BLOCK_128: return "FP8_BLOCK_128";
            case WeightEncoding::FP8_ROW: return "FP8_ROW";
            case WeightEncoding::NVFP4_LINEAR: return "NVFP4_LINEAR";
            case WeightEncoding::NVFP4_SWIZZLED: return "NVFP4_SWIZZLED";
        }
        return "UNKNOWN";
    }

private:
    static const char *transform_name(TensorTransform v) {
        switch (v) {
            case TensorTransform::COPY: return "COPY"; case TensorTransform::BF16_TO_F32: return "BF16_TO_F32";
            case TensorTransform::FP8_TO_Q4K: return "FP8_TO_Q4K";
            case TensorTransform::FP8_TO_NVFP4: return "FP8_TO_NVFP4";
            case TensorTransform::NVFP4_REBASE: return "NVFP4_REBASE";
            case TensorTransform::BF16_TO_Q8_0: return "BF16_TO_Q8_0";
            case TensorTransform::QUANT_TO_BF16: return "QUANT_TO_BF16";
        }
        return "UNKNOWN";
    }
    static const char *allocation_name(AllocationClass v) {
        switch (v) {
            case AllocationClass::CORE_WEIGHTS: return "core_weights";
            case AllocationClass::SCALES: return "scales";
            case AllocationClass::EXPERT_SLABS: return "expert_slabs";
            case AllocationClass::EMBEDDING_HEAD: return "embedding_head";
            case AllocationClass::PREFILL_CACHE: return "prefill_cache";
            case AllocationClass::WORKSPACE: return "workspace";
        }
        return "unknown";
    }
    static std::string escape(const std::string &s) {
        std::string r; r.reserve(s.size());
        for (unsigned char c : s) {
            if (c == '\\' || c == '\"') { r.push_back('\\'); r.push_back((char)c); }
            else if (c == '\n') r += "\\n";
            else if (c == '\r') r += "\\r";
            else if (c == '\t') r += "\\t";
            else if (c >= 0x20) r.push_back((char)c);
        }
        return r;
    }

    std::vector<PlannedTensor> tensors_;
    size_t totals_[(int)AllocationClass::WORKSPACE + 1] = {};
    bool finalized_ = false;
};

#endif  // FUCINA_MODEL_PLAN_H
