#pragma once

// Muse Glimmer split-KV GQA small-T kernel: D=128, group 16. WarpsPerCta must
// cover TokenTile*16 rows. INT8-G64 dequantizes into the BF16 MMA tiles.
// window_size follows the same contract as the D=128 prompt kernel.

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cuda_fp16.h>
#include <math_constants.h>

namespace ginfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          bool Int8, typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, 1) __global__
    void gqa_attention_d128_partial_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
        __nv_bfloat16* cache_v, std::int8_t* cache_k_i8, std::int8_t* cache_v_i8,
        __half* cache_k_scale, __half* cache_v_scale, const std::int32_t* block_tables,
        const std::int32_t* valid_columns, const std::int32_t* table_rows,
        std::int32_t table_stride, std::int32_t tokens, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float scale,
        std::int32_t window_size, __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(Geometry::HeadDim == 128);
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 6);
    static_assert(TokenTile * Geometry::GroupSize <= WarpsPerCta * 16);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = Geometry::HeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    [[maybe_unused]] constexpr int Groups = Geometry::QuantGroups;
    constexpr int PageIds       = 64;
    constexpr int StagingRows   = (Br > 2 * Bc) ? Br : (2 * Bc);
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ __align__(16) __nv_bfloat16 qkv_s[StagingRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int hist_hi = last_pos + 1;
    const int hist_lo = gqa_sliding_key_lo(first_pos, window_size);
    const int window  = hist_hi;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, Int8>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start_raw = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit     = split_start_raw + units_per_split * (tile_split ? Bc : 1);
    const int split_start     = split_start_raw > hist_lo ? split_start_raw : hist_lo;
    const int split_end       = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        if constexpr (Int8) {
            for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
                const int token    = pair / Groups;
                const int grp      = pair - token * Groups;
                const int position = pos[token];
                if (position < split_start || position >= split_end) { continue; }
                int physical_page     = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
                const int page_offset = position & kPagedKVPageMask;
                const int d0          = grp * kGqaKvQuantGroup + lane;
                const int d1          = d0 + 32;
                const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
                const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
                const float kv0         = __bfloat162float(input.k[src0]);
                const float kv1         = __bfloat162float(input.k[src1]);
                const float vv0         = __bfloat162float(input.v[src0]);
                const float vv1         = __bfloat162float(input.v[src1]);
                float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
                float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
                kamax                   = warp_max(kamax, FullMask);
                vamax                   = warp_max(vamax, FullMask);
                const __half ksh        = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
                const __half vsh        = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
                const float ks          = __half2float(ksh);
                const float vs          = __half2float(vsh);
                const float k_inv       = ks > 0.0f ? 1.0f / ks : 0.0f;
                const float v_inv       = vs > 0.0f ? 1.0f / vs : 0.0f;
                physical_page           = __shfl_sync(FullMask, physical_page, 0);
                cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                             page_offset)] =
                    gqa_kv_quant_code(kv0, k_inv);
                cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                             page_offset)] =
                    gqa_kv_quant_code(kv1, k_inv);
                cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                             page_offset)] =
                    gqa_kv_quant_code(vv0, v_inv);
                cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                             page_offset)] =
                    gqa_kv_quant_code(vv1, v_inv);
                if (lane == 0) {
                    const std::int64_t so = gqa_kv_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = ksh;
                    cache_v_scale[so] = vsh;
                }
            }
        } else {
            for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
                const int token = chunk / (D / 8);
                const int d     = (chunk - token * (D / 8)) * 8;
                const int p_tok = pos[token];
                if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                    p_tok < logical_capacity) {
                    const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                    // T=1 exposes only 16 vector chunks per KV head, so only half a warp enters
                    // this loop. A full-mask shuffle here deadlocks waiting for lanes 16..31.
                    // Every active lane instead reads the same tiny page-table entry directly.
                    const int physical_page = paged_kv_physical_page(block_table, p_tok);
                    const std::int64_t cache_off =
                        gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                  p_tok & kPagedKVPageMask);
                    store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                    store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
                }
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                if constexpr (Int8) {
                    const int grp = d >> 6;
                    const std::int64_t code_off = gqa_kv_quant_code_index<Geometry>(
                        physical_page, kv_head, d, key & kPagedKVPageMask);
                    const std::int64_t scale_off = gqa_kv_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, key & kPagedKVPageMask);
                    const float ks = __half2float(cache_k_scale[scale_off]);
                    const float vs = __half2float(cache_v_scale[scale_off]);
                    store_vec(k_dst, gqa_kv_dequant_i8x8_from(&cache_k_i8[code_off], ks));
                    store_vec(v_dst, gqa_kv_dequant_i8x8_from(&cache_v_i8[code_off], vs));
                } else if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    const bool from_new =
                        new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                    if (from_new) {
                        const std::int64_t off = gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                        ginfer::ops::cp_async<16>(k_dst, &input.k[off]);
                        ginfer::ops::cp_async<16>(v_dst, &input.v[off]);
                    } else {
                        const std::int64_t off = gqa_cache_index<Geometry>(
                            physical_page, kv_head, d, key & kPagedKVPageMask);
                        ginfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                        ginfer::ops::cp_async<16>(v_dst, &cache_v[off]);
                    }
                } else {
                    const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                                       key & kPagedKVPageMask);
                    ginfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                    ginfer::ops::cp_async<16>(v_dst, &cache_v[off]);
                }
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (!Int8) {
            ginfer::ops::cp_commit();
            ginfer::ops::cp_wait<0>();
        }
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            const bool a0 =
                row0 < row_count && key0 >= split_start && key0 < split_end &&
                gqa_key_admitted(key0, qabs0, window_size);
            const bool a1 =
                row0 < row_count && key1 >= split_start && key1 < split_end &&
                gqa_key_admitted(key1, qabs0, window_size);
            const bool b0 =
                row1 < row_count && key0 >= split_start && key0 < split_end &&
                gqa_key_admitted(key0, qabs1, window_size);
            const bool b1 =
                row1 < row_count && key1 >= split_start && key1 < split_end &&
                gqa_key_admitted(key1, qabs1, window_size);
            score[nt][0] = a0 ? score[nt][0] * scale : -CUDART_INF_F;
            score[nt][1] = a1 ? score[nt][1] * scale : -CUDART_INF_F;
            score[nt][2] = b0 ? score[nt][2] * scale : -CUDART_INF_F;
            score[nt][3] = b1 ? score[nt][3] * scale : -CUDART_INF_F;
            bm0          = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1          = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

} // namespace ginfer::ops
