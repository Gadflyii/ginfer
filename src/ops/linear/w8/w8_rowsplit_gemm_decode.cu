#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ginfer::ops::detail {
namespace {

constexpr int kValuesPerLane = 8;

template <int Hidden, int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_rowsplit_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    static_assert(Hidden > 0 && (Hidden % (32 * kValuesPerLane)) == 0);
    constexpr int kGroups         = Hidden / 32;
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = Hidden / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const auto* code_row = codes + static_cast<std::int64_t>(row) * Hidden;
    const auto* scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * sizeof(std::uint16_t);

    float acc = 0.0f;
#pragma unroll 8
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kGroupsPerPhase) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        scale_bits        = __shfl_sync(kMask, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 xv[4] = {
            bf16x2_bits_to_float2(values.x),
            bf16x2_bits_to_float2(values.y),
            bf16x2_bits_to_float2(values.z),
            bf16x2_bits_to_float2(values.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t word = (&packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const float2 activation_pair = xv[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float weight_value =
                    static_cast<float>(static_cast<std::int8_t>(word >> (byte * 8))) * scale;
                acc = fmaf(weight_value, activation, acc);
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
}

template <int Rows, int Hidden, int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert((Rows % RowsPerCta) == 0);
    static_assert((Hidden % (32 * kValuesPerLane)) == 0);
    if (x.ne[0] != Hidden || x.ne[1] != 1 || out.ne[0] != Rows || out.ne[1] != 1 || w.n != Rows ||
        w.k != Hidden || w.padded_shape[1] != Hidden) {
        throw std::invalid_argument("W8 decode requires matching [N,K] and T=1");
    }
    w8_rowsplit_decode_kernel<Hidden, RowsPerCta>
        <<<Rows / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_w8_decode_r4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_decode<2048, 16384, 4>(x, w, out, stream);
}

void launch_w8_decode_n202048_k6656(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_decode<202048, 6656, 16>(x, w, out, stream);
}

void w8_rowsplit_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_decode<2048, 16384, 16>(x, w, out, stream);
}

} // namespace ginfer::ops::detail
