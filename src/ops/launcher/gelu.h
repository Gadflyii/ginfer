#pragma once

#include "core/tensor.h"
#include "ginfer/ops/gelu.h"

#include <cuda_runtime.h>

namespace ginfer::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace ginfer::ops::detail
