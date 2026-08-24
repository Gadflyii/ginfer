#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ginfer::ops {

/**
 * Two-tap grouped dynamic causal conv used by DFlash2.
 *
 * `hidden` and `out` are contiguous BF16 [H,W,B]. The B=1 form may be passed as [H,W]. `proj` is
 * a PyTorch-contiguous BF16 matrix
 * [2*K*G, H] with H fastest (nn.Linear weight). `base` is PyTorch-contiguous BF16 [2,K,H]
 * (stage, tap, hidden). `dynamic` is contiguous BF16 [2*K*G,W,B] produced by prepare and
 * consumed by finish. Each batch row is an independent causal sequence: tap one at column zero
 * reads zero rather than the previous batch row's tail. group_size*G == H, kernel K=2, stages=2.
 * W is 1..16, B is 1..8, and tensor storage does not overlap.
 *
 * prepare: dynamic = hidden^T @ proj^T, then out = conv(hidden, dynamic[stage0], base[0]).
 * finish: out = conv(hidden, dynamic[stage1], base[1]).
 * The Muse H=6656, group_size=16, B*W<=128 projection has an sm_120a BF16 Tensor Core route.
 */
void grouped_dynamic_conv_prepare(const Tensor& hidden, const Tensor& proj, const Tensor& base,
                                  Tensor& dynamic, Tensor& out, std::int32_t group_size,
                                  cudaStream_t stream);

/**
 * Type-native DFlash2 projection route. `projection` is logical [4*(H/group_size),H] in
 * BF16_CTRL, Q4G64_F16S, or W8G32_F16S. Every format produces the same observable BF16
 * `dynamic` boundary before the grouped convolution; quantized formats are decoded directly from
 * their artifact planes.
 */
void grouped_dynamic_conv_prepare(const Tensor& hidden, const Weight& projection,
                                  const Tensor& base, Tensor& dynamic, Tensor& out,
                                  std::int32_t group_size, cudaStream_t stream);

void grouped_dynamic_conv_finish(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                                 Tensor& out, std::int32_t group_size, cudaStream_t stream);

} // namespace ginfer::ops
