#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ginfer::ops::detail {

void cast_fp32_to_bf16_launch(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace ginfer::ops::detail
