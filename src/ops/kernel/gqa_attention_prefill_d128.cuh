#pragma once

// Muse Glimmer GQA prompt kernels: D=128, Q=32, KV=2. BF16 K/V stay on the FA-2
// tensor-core path. INT8-G64 dequantizes into the same BF16 smem tiles so the
// MMA schedule is shared. Sliding window_size>0 admits keys in
// [max(0, p-window_size+1), p]; window_size==0 is full causal history.

#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <math_constants.h>

namespace ginfer::ops {

inline constexpr int kGqaPrefillD128SmemBytes =
    (kGqaPrefillBr + 2 * kGqaPrefillBc) * 128 * static_cast<int>(sizeof(__nv_bfloat16));

template <typename Geometry, bool Int8>
__device__ __forceinline__ void gqa_prefill_d128_stage_kv(
    __nv_bfloat16* dst, const void* cache, const __half* scales, int kv_head, int k0,
    int max_query_abs, int min_key, int physical_page, int tid) {
    static_assert(Geometry::HeadDim == 128);
    constexpr int D         = Geometry::HeadDim;
    constexpr int Bc        = kGqaPrefillBc;
    constexpr int Threads   = kGqaPrefillThreads;
    constexpr int VecPerRow = D / 8;
    const bool full_tile    = (k0 >= min_key) && ((k0 + Bc - 1) <= max_query_abs);
    const int page_off0     = k0 & kPagedKVPageMask;
    if constexpr (Int8) {
        const std::int8_t* cache_i8 = static_cast<const std::int8_t*>(cache);
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk / VecPerRow;
            const int d      = (chunk - key_l * VecPerRow) * 8;
            const int key    = k0 + key_l;
            __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            if (key >= min_key && key <= max_query_abs) {
                const int grp        = d >> 6;
                const std::int64_t off = gqa_kv_quant_code_index<Geometry>(
                    physical_page, kv_head, d, page_off0 + key_l);
                const std::int64_t scale_off = gqa_kv_quant_scale_index<Geometry>(
                    physical_page, kv_head, grp, page_off0 + key_l);
                store_vec(p, gqa_kv_dequant_i8x8_from(&cache_i8[off],
                                                      __half2float(scales[scale_off])));
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
        (void)full_tile;
    } else {
        const __nv_bfloat16* cache_block =
            static_cast<const __nv_bfloat16*>(cache) +
            paged_kv_element_offset<D, Geometry::KVHeads>(physical_page, kv_head, page_off0, 0);
        if (full_tile) {
#pragma unroll
            for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
                const int key_l  = chunk / VecPerRow;
                const int d      = (chunk - key_l * VecPerRow) * 8;
                __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
                cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
                const int key_l  = chunk / VecPerRow;
                const int d      = (chunk - key_l * VecPerRow) * 8;
                const int key    = k0 + key_l;
                __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
                if (key >= min_key && key <= max_query_abs) {
                    cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }
}

template <typename Geometry, typename Metadata, bool Int8>
__launch_bounds__(kGqaPrefillThreads, 1) __global__
    void gqa_attention_prefill_d128_kernel(const __nv_bfloat16* __restrict__ q, const void* cache_k,
                                           const void* cache_v, const __half* cache_k_scale,
                                           const __half* cache_v_scale, Metadata metadata,
                                           const std::int32_t* __restrict__ positions, float scale,
                                           std::int32_t window_size, __nv_bfloat16* __restrict__ out,
                                           std::int32_t width) {
    static_assert(Geometry::HeadDim == 128);
    constexpr int D             = Geometry::HeadDim;
    constexpr int Br            = kGqaPrefillBr;
    constexpr int Bc            = kGqaPrefillBc;
    constexpr int Threads       = kGqaPrefillThreads;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int kRowBytes     = D * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 128);
    static_assert(kGqaPrefillD128SmemBytes == (Br + 2 * Bc) * kRowBytes);

    extern __shared__ __align__(16) __nv_bfloat16 gqa_d128_smem[];
    __nv_bfloat16* q_s = gqa_d128_smem;
    __nv_bfloat16* k_s = q_s + Br * D;
    __nv_bfloat16* v_s = k_s + Bc * D;

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    const unsigned q_lane_base =
        q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * kRowBytes);
    const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r  = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = k_sbase + static_cast<unsigned>(b_rin * kRowBytes) +
                                 (static_cast<unsigned>(lane >> 4) * 8u * kRowBytes);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = v_sbase +
                                 static_cast<unsigned>(((lane >> 3) & 1) * 8 * kRowBytes) +
                                 static_cast<unsigned>(b_rin * kRowBytes);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads;
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk / VecPerRow;
                const int d      = (chunk - row * VecPerRow) * 8;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk / VecPerRow;
                const int d      = (chunk - row * VecPerRow) * 8;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int min_key       = gqa_sliding_key_lo(base_pos + q0, window_size);
    const int n_block_max   = (max_query_abs / Bc) + 1;
    const int kb_begin      = min_key / Bc;
    const float scale_l2    = scale * Log2E;
    int physical_page       = block_table[kb_begin];

    ginfer::ops::cp_commit();
    if (kb_begin < n_block_max) {
        gqa_prefill_d128_stage_kv<Geometry, Int8>(k_s, cache_k, cache_k_scale, kv_head,
                                                  kb_begin * Bc, max_query_abs, min_key,
                                                  physical_page, tid);
        ginfer::ops::cp_commit();
    }

    for (int kb = kb_begin; kb < n_block_max; ++kb) {
        const int k0                 = kb * Bc;
        const int next_physical_page = (kb + 1 < n_block_max) ? block_table[kb + 1] : physical_page;

        ginfer::ops::cp_wait<0>();
        __syncthreads();

        gqa_prefill_d128_stage_kv<Geometry, Int8>(v_s, cache_v, cache_v_scale, kv_head, k0,
                                                  max_query_abs, min_key, physical_page, tid);
        ginfer::ops::cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 8 * kRowBytes),
                                                 0u, k_as, k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(
                            k_lane_base + static_cast<unsigned>(nt2 * 8 * kRowBytes), ck, k_as,
                            k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1            = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile = window_size <= 0 && (q0 + Br <= tokens) &&
                                     ((k0 + Bc - 1) <= (base_pos + q0));

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0] =
                    (qrow0 < tokens && gqa_key_admitted(key0, qabs0, window_size)) ? score[nt][0]
                                                                                   : -CUDART_INF_F;
                score[nt][1] =
                    (qrow0 < tokens && gqa_key_admitted(key1, qabs0, window_size)) ? score[nt][1]
                                                                                   : -CUDART_INF_F;
                score[nt][2] =
                    (qrow1 < tokens && gqa_key_admitted(key0, qabs1, window_size)) ? score[nt][2]
                                                                                   : -CUDART_INF_F;
                score[nt][3] =
                    (qrow1 < tokens && gqa_key_admitted(key1, qabs1, window_size)) ? score[nt][3]
                                                                                   : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0     =
            (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     =
            (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        ginfer::ops::cp_wait<0>();
        __syncthreads();

        if (kb + 1 < n_block_max) {
            physical_page = next_physical_page;
            gqa_prefill_d128_stage_kv<Geometry, Int8>(k_s, cache_k, cache_k_scale, kv_head,
                                                      (kb + 1) * Bc, max_query_abs, min_key,
                                                      physical_page, tid);
            ginfer::ops::cp_commit();
        }

        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        {
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          gqa_prefill_swz_addr(v_lane_base, 0u, v_as, v_r));
        }
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              gqa_prefill_swz_addr(
                                  v_lane_base + static_cast<unsigned>(k2 * 16 * kRowBytes), ckv,
                                  v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);

    const float inv_l0 = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, Threads);
}

} // namespace ginfer::ops
