#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ginfer::targets::muse_glimmer_30b::detail {

// out[i] = tanh((float(x[i]) * multiplier) / cap) * cap, written as BF16.
void logit_softcap(const Tensor& logits, Tensor& out, float multiplier, float cap,
                   cudaStream_t stream);

} // namespace ginfer::targets::muse_glimmer_30b::detail
