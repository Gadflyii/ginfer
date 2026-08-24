#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ginfer::ops::detail {

struct Q4RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct Q4SimtDecodeAtom {
    __device__ static __forceinline__ void
    decode_eight(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[8]) {
        const std::uint32_t word = packed ^ 0x88888888u;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            const __half2 decoded    = __hsub2(half2_from_bits(bits), bias);
            const float2 values      = __half22float2(decoded);
            weights[pair]            = values.x * scale;
            weights[pair + 4]        = values.y * scale;
        }
    }

    __device__ static __forceinline__ void
    decode_pair(std::uint8_t packed, std::uint16_t scale_bits, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int q0      = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1      = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        w0                = static_cast<float>(q0) * scale;
        w1                = static_cast<float>(q1) * scale;
    }

    // Signed 4-bit two's complement extract. Same mapping as decode_eight's
    // (nibble ^ 8) - 8 / FP16-mantissa path, but one BFE instead of hsub2+cvt.
    template <int Shift>
    __device__ static __forceinline__ int signed_nibble(std::uint32_t packed) {
        static_assert(Shift >= 0 && Shift <= 28 && (Shift % 4) == 0);
        int q;
        asm("bfe.s32 %0, %1, %2, 4;" : "=r"(q) : "r"(packed), "n"(Shift));
        return q;
    }

    // Sequential nibbles 0..7 match the 8 consecutive activations in `activation_bits`.
    // Scale is applied once so the inner products stay I2F + FMA.
    __device__ static __forceinline__ float fma_eight(std::uint32_t packed, std::uint16_t scale_bits,
                                                     uint4 activation_bits, float acc) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        float local       = 0.0f;
        {
            const float2 x = bf16x2_bits_to_float2(activation_bits.x);
            local          = fmaf(static_cast<float>(signed_nibble<0>(packed)), x.x, local);
            local          = fmaf(static_cast<float>(signed_nibble<4>(packed)), x.y, local);
        }
        {
            const float2 x = bf16x2_bits_to_float2(activation_bits.y);
            local          = fmaf(static_cast<float>(signed_nibble<8>(packed)), x.x, local);
            local          = fmaf(static_cast<float>(signed_nibble<12>(packed)), x.y, local);
        }
        {
            const float2 x = bf16x2_bits_to_float2(activation_bits.z);
            local          = fmaf(static_cast<float>(signed_nibble<16>(packed)), x.x, local);
            local          = fmaf(static_cast<float>(signed_nibble<20>(packed)), x.y, local);
        }
        {
            const float2 x = bf16x2_bits_to_float2(activation_bits.w);
            local          = fmaf(static_cast<float>(signed_nibble<24>(packed)), x.x, local);
            local          = fmaf(static_cast<float>(signed_nibble<28>(packed)), x.y, local);
        }
        return fmaf(local, scale, acc);
    }
};

struct Q4MmaDecodeAtom {
    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* codes,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
        const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

} // namespace ginfer::ops::detail
