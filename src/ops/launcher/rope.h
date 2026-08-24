#pragma once

// ginfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/gpu_capability.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ginfer::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 const OccupancyPolicy& occupancy, cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        const OccupancyPolicy& occupancy, cudaStream_t stream);

} // namespace ginfer::ops::detail
