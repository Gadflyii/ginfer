#pragma once

// Implements: include/ginfer/ops/sampling.h
// Match: contiguous BF16 logits, physical stride >= token domain, and at most
// sixteen columns on the multi-block route.
// Algorithm assumptions: Qwen uses 256-thread/2-item partial tiles and bounded top-20 group
// merges. Muse's 202048-token domain uses 256-thread/4-item partial tiles and exact hierarchical
// top-64 selection. Other non-multiblock geometries retain the single-block fallback.

#include "ops/kernel/sampling_device.cuh"

namespace ginfer::ops {

struct SamplingWideSelectStorage {
    int histogram[256];
    unsigned long long prefix;
    int rank;
    int output_count;
};

// Exact block-wide radix selection. Keys include the token-id tie break and are therefore unique.
// The builder needs only the best `cap` keys, so selecting one radix bucket per byte avoids fully
// sorting the other 960 entries in every Muse partial tile. Output order is intentionally
// unspecified; the next hierarchy level ranks the selected union.
template <int ItemsPerThread>
__device__ inline void sampling_select_top_keys(
    const unsigned long long (&keys)[ItemsPerThread], int cap, unsigned long long* output,
    SamplingWideSelectStorage& storage) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        storage.prefix = 0ull;
        storage.rank   = cap;
    }
    __syncthreads();

    unsigned long long prefix_mask = 0ull;
#pragma unroll
    for (int shift = 56; shift >= 0; shift -= 8) {
        storage.histogram[tid] = 0;
        __syncthreads();
        const unsigned long long prefix = storage.prefix;
#pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item) {
            const unsigned long long key = keys[item];
            if (key != 0ull && (key & prefix_mask) == prefix) {
                atomicAdd(&storage.histogram[(key >> shift) & 0xffu], 1);
            }
        }
        __syncthreads();
        if (tid == 0) {
            int rank = storage.rank;
            for (int digit = 255; digit >= 0; --digit) {
                const int count = storage.histogram[digit];
                if (rank > count) {
                    rank -= count;
                } else {
                    storage.prefix |= static_cast<unsigned long long>(digit) << shift;
                    storage.rank = rank;
                    break;
                }
            }
        }
        __syncthreads();
        prefix_mask |= 0xffull << shift;
    }

    if (tid == 0) { storage.output_count = 0; }
    __syncthreads();
    const unsigned long long threshold = storage.prefix;
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const unsigned long long key = keys[item];
        if (key >= threshold) {
            const int output_rank = atomicAdd(&storage.output_count, 1);
            if (output_rank < cap) { output[output_rank] = key; }
        }
    }
    __syncthreads();
}

__launch_bounds__(kSamplerBlock) __global__
    void sample_row_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                           const SamplingConfig* configs, const std::int32_t* logical_positions,
                           std::int32_t purpose, std::int32_t token_domain,
                           std::int32_t physical_rows) {
    const int row            = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(row) * physical_rows;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = configs[row];

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // Greedy: exact argmax over raw logits. Bit-identical to argmax().
    if (!(cfg.temperature > 0.0f)) {
        float bv = -CUDART_INF_F;
        int bi   = INT_MAX;
        for (int v = tid; v < token_domain; v += blockDim.x) {
            const float x = __bfloat162float(logits[base + v]);
            if (sampling_better(x, v, bv, bi)) {
                bv = x;
                bi = v;
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) { out[row] = red_idx[0]; }
        return;
    }

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape (see sample_batch_launch).
    if (sampler_multiblock_ok(token_domain, static_cast<int>(gridDim.x), partial_blocks,
                              group_count)) {
        return;
    }

    __shared__ float cand_val[kSamplerWideCandidates];
    __shared__ int cand_idx[kSamplerWideCandidates];
    __shared__ float prob[kSamplerWideCandidates];
    __shared__ int n_support;
    __shared__ float stage_val[32 * kSamplerWideCandidates];
    __shared__ int stage_idx[32 * kSamplerWideCandidates];

    if (token_domain <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, token_domain, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support, nullptr, 0,
                                       kSamplerWideCandidates);
    } else {
        sampling_build_truncated_block_wide<kSamplerWideCandidates>(
            logits, base, token_domain, cfg, stage_val, stage_idx, cand_val, cand_idx, prob,
            &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, logical_positions[row], purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j]; // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[row] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

// Muse top-64 stage 1: each CTA ranks one 1024-token tile and publishes its exact top-k list.
// The fixed tile gives the 202048-token row enough independent CTAs to occupy sm_120a without
// spilling a 64-entry insertion list in every thread.
template <bool RadixSelect>
__launch_bounds__(kSamplerBlock) __global__ void sampling_wide_partial_topk_kernel(
    const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr, std::int32_t token_domain,
    std::int32_t physical_rows, WideSamplingWorkspace workspace) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = cfg_ptr[col];
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    __shared__ typename SamplingWideSort::TempStorage sort_storage;
    __shared__ SamplingWideSelectStorage select_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    unsigned long long keys[kSamplerWideItemsPerThread];
    const bool greedy       = !(cfg.temperature > 0.0f);
    const int cap           = greedy ? 1 : sampling_candidate_cap(
                                               cfg, token_domain, kSamplerWideCandidates);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * kSamplerWidePartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
        const int token = tile_start + item * blockDim.x + threadIdx.x;
        if (token < token_domain) {
            const float raw = __bfloat162float(logits[base + token]);
            const float value = greedy ? raw : sampling_adjusted_logit(raw, token, cfg);
            keys[item]        = sampling_sort_key(value, token);
        } else {
            keys[item] = 0ull;
        }
    }

    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerWideItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int offset = sampling_wide_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[offset] = best;
        }
        return;
    }

    if constexpr (RadixSelect) {
        const int output_offset = sampling_wide_partial_offset(workspace, col, partial, 0);
        sampling_select_top_keys(keys, cap, workspace.partial_keys + output_offset, select_storage);
    } else {
        SamplingWideSort(sort_storage).SortDescending(keys);
#pragma unroll
        for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
            const int rank = threadIdx.x * kSamplerWideItemsPerThread + item;
            if (rank < cap) {
                const int offset = sampling_wide_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[offset] = keys[item];
            }
        }
    }
}

// Muse top-64 stage 2: each CTA merges sixteen partial lists. The last completed group CTA then
// merges the thirteen group lists, normalizes the support, and performs the unchanged inverse CDF.
template <bool RadixSelect>
__launch_bounds__(kSamplerGroupBlock) __global__ void
sampling_wide_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* logical_positions,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
    std::int32_t group_count, WideSamplingWorkspace workspace) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    __shared__ typename SamplingWideSort::TempStorage sort_storage;
    __shared__ SamplingWideSelectStorage select_storage;
    __shared__ float cand_val[kSamplerWideCandidates];
    __shared__ int cand_idx[kSamplerWideCandidates];
    __shared__ float prob[kSamplerWideCandidates];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[kSamplerGroupBlock / 32];
    unsigned long long keys[kSamplerWideItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain, kSamplerWideCandidates);
    const int group_begin = group * kSamplerWidePartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerWidePartialsPerGroup) {
        group_partials = kSamplerWidePartialsPerGroup;
    }

    if (greedy) {
        unsigned long long best = 0ull;
        for (int partial = tid; partial < group_partials; partial += blockDim.x) {
            const int offset =
                sampling_wide_partial_offset(workspace, col, group_begin + partial, 0);
            if (workspace.partial_keys[offset] > best) { best = workspace.partial_keys[offset]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int offset =
                sampling_wide_partial_offset(workspace, col, partial_blocks + group, 0);
            workspace.partial_keys[offset] = best;
            __threadfence();
            const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
            is_last        = (done == group_count) ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int candidate = tid; candidate < group_count; candidate += blockDim.x) {
            const int offset =
                sampling_wide_partial_offset(workspace, col, partial_blocks + candidate, 0);
            if (workspace.partial_keys[offset] > best) { best = workspace.partial_keys[offset]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            out[col]                  = sampling_key_index(best);
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int group_items = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
        const int source = item * blockDim.x + tid;
        if (source < group_items) {
            const int partial = group_begin + source / cap;
            const int rank    = source - (source / cap) * cap;
            const int offset  = sampling_wide_partial_offset(workspace, col, partial, rank);
            keys[item]        = workspace.partial_keys[offset];
        } else {
            keys[item] = 0ull;
        }
    }
    if constexpr (RadixSelect) {
        const int group_output_offset =
            sampling_wide_partial_offset(workspace, col, partial_blocks + group, 0);
        sampling_select_top_keys(keys, cap, workspace.partial_keys + group_output_offset,
                                 select_storage);
    } else {
        SamplingWideSort(sort_storage).SortDescending(keys);
#pragma unroll
        for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
            const int rank = tid * kSamplerWideItemsPerThread + item;
            if (rank < cap) {
                const int offset =
                    sampling_wide_partial_offset(workspace, col, partial_blocks + group, rank);
                workspace.partial_keys[offset] = keys[item];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_items = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
        const int source = item * blockDim.x + tid;
        if (source < final_items) {
            const int candidate = source / cap;
            const int rank      = source - candidate * cap;
            const int offset = sampling_wide_partial_offset(
                workspace, col, partial_blocks + candidate, rank);
            keys[item] = workspace.partial_keys[offset];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingWideSort(sort_storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
        const int rank = tid * kSamplerWideItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u     = sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u);
        float cumulative  = 0.0f;
        int picked        = cand_idx[support - 1];
        for (int rank = 0; rank < support; ++rank) {
            cumulative += prob[rank];
            if (u < cumulative) {
                picked = cand_idx[rank];
                break;
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        workspace.group_done[col] = 0;
    }
}

__launch_bounds__(kSamplerBlock) __global__
    void sampling_partial_topk_kernel(const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr,
                                      std::int32_t token_domain, std::int32_t physical_rows,
                                      SamplingWorkspace workspace) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = cfg_ptr[col];
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    unsigned long long keys[kSamplerItemsPerThread];

    const bool greedy       = !(cfg.temperature > 0.0f);
    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float raw = __bfloat162float(logits[base + v]);
            const float x   = greedy ? raw : sampling_adjusted_logit(raw, v, cfg);
            keys[item]      = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off               = sampling_partial_offset(workspace, col, partial, rank);
            workspace.partial_keys[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* logical_positions,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[kSamplerGroupBlock / 32];
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes group_done[col], so caller-owned
    // workspace does not rely on prior contents or a separate memset launch.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }

    if (greedy) {
        unsigned long long best = 0ull;
        for (int p = tid; p < group_partials; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, group_begin + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int out_off = sampling_partial_offset(workspace, col, partial_blocks + group, 0);
            workspace.partial_keys[out_off] = best;
            __threadfence();
            const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
            is_last        = (done == group_count) ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int p = tid; p < group_count; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, partial_blocks + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            out[col]                  = sampling_key_index(best);
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u     = sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u);
        float acc         = 0.0f;
        int picked        = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        workspace.group_done[col] = 0;
    }
}

} // namespace ginfer::ops
