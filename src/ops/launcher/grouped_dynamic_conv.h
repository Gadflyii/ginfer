#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ginfer::ops::detail {

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& proj,
                                         const Tensor& base, Tensor& dynamic, Tensor& out,
                                         std::int32_t group_size, cudaStream_t stream);

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Weight& projection,
                                         const Tensor& base, Tensor& dynamic, Tensor& out,
                                         std::int32_t group_size, cudaStream_t stream);

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& dynamic,
                                        const Tensor& base, Tensor& out, std::int32_t group_size,
                                        cudaStream_t stream);

} // namespace ginfer::ops::detail
