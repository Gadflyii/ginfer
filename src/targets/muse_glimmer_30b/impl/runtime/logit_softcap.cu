#include "targets/muse_glimmer_30b/impl/runtime/logit_softcap.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace ginfer::targets::muse_glimmer_30b::detail {
namespace {

__global__ void logit_softcap_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int n,
                                     float multiplier, float cap) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) { return; }
    const float scaled = __bfloat162float(in[i]) * multiplier;
    const float y      = tanhf(scaled / cap) * cap;
    out[i]             = __float2bfloat16(y);
}

} // namespace

void logit_softcap(const Tensor& logits, Tensor& out, float multiplier, float cap,
                   cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("logit_softcap: tensors must be BF16");
    }
    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("logit_softcap: tensors must be contiguous");
    }
    if (logits.numel() != out.numel()) {
        throw std::invalid_argument("logit_softcap: numel mismatch");
    }
    if (!std::isfinite(multiplier) || !std::isfinite(cap) || cap <= 0.0F) {
        throw std::invalid_argument("logit_softcap: multiplier/cap invalid");
    }
    const int n = static_cast<int>(logits.numel());
    if (n <= 0) { throw std::invalid_argument("logit_softcap: empty"); }
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    logit_softcap_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<__nv_bfloat16*>(out.data), n,
        multiplier, cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ginfer::targets::muse_glimmer_30b::detail
