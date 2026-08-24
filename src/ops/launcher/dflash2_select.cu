#include "ops/launcher/dflash2_select.h"

#include "core/device.h"
#include "ops/common/dflash2_accept_workspace.h"
#include "ops/common/dflash2_selector_workspace.h"
#include "ops/common/sampling_workspace.h"
#include "ops/kernel/sampling_device.cuh"
#include "ginfer/ops/linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ginfer::ops::detail {
namespace {

constexpr int kMaxTopK = kDFlash2SelectorMaximumTopK;
constexpr int kMaxRank = 256;

__global__ void gemm_pt_a_ginfer_b_kernel(const __nv_bfloat16* __restrict__ a_nk,
                                          const __nv_bfloat16* __restrict__ b_kt,
                                          __nv_bfloat16* __restrict__ c_nt, int n, int k,
                                          int columns) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    if (row >= n || col >= columns) { return; }
    float acc = 0.0f;
    const __nv_bfloat16* a_row = a_nk + static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* b_col = b_kt + static_cast<std::int64_t>(col) * k;
    for (int i = 0; i < k; ++i) {
        acc += __bfloat162float(a_row[i]) * __bfloat162float(b_col[i]);
    }
    c_nt[row + static_cast<std::int64_t>(col) * n] = __float2bfloat16(acc);
}

__device__ void insert_top(float* values, int* ids, int count, float value, int id) {
    if (!sampling_better(value, id, values[count - 1], ids[count - 1])) { return; }
    int slot = count - 1;
    while (slot > 0 && sampling_better(value, id, values[slot - 1], ids[slot - 1])) {
        values[slot] = values[slot - 1];
        ids[slot]    = ids[slot - 1];
        --slot;
    }
    values[slot] = value;
    ids[slot]    = id;
}

__global__ void topk_column_fallback_kernel(const __nv_bfloat16* __restrict__ logits, float* unary,
                                            int* candidates, int token_domain,
                                            int physical_rows, int tokens, int top_k) {
    const int token  = static_cast<int>(blockIdx.x);
    const int batch  = static_cast<int>(blockIdx.y);
    const int column = token + tokens * batch;
    const int lane   = static_cast<int>(threadIdx.x);
    float best_v[kMaxTopK];
    int best_i[kMaxTopK];
#pragma unroll
    for (int i = 0; i < kMaxTopK; ++i) {
        best_v[i] = -CUDART_INF_F;
        best_i[i] = INT_MAX;
    }
    const __nv_bfloat16* col = logits + static_cast<std::int64_t>(column) * physical_rows;
    for (int row = lane; row < token_domain; row += static_cast<int>(blockDim.x)) {
        insert_top(best_v, best_i, top_k, __bfloat162float(col[row]), row);
    }
    __shared__ float shared_v[256 * kMaxTopK];
    __shared__ int shared_i[256 * kMaxTopK];
    for (int i = 0; i < top_k; ++i) {
        shared_v[lane * kMaxTopK + i] = best_v[i];
        shared_i[lane * kMaxTopK + i] = best_i[i];
    }
    __syncthreads();
    if (lane == 0) {
        float acc_v[kMaxTopK];
        int acc_i[kMaxTopK];
#pragma unroll
        for (int i = 0; i < kMaxTopK; ++i) {
            acc_v[i] = -CUDART_INF_F;
            acc_i[i] = INT_MAX;
        }
        for (int thread = 0; thread < static_cast<int>(blockDim.x); ++thread) {
            for (int i = 0; i < top_k; ++i) {
                insert_top(acc_v, acc_i, top_k, shared_v[thread * kMaxTopK + i],
                           shared_i[thread * kMaxTopK + i]);
            }
        }
        const std::int64_t output = static_cast<std::int64_t>(column) * top_k;
        for (int i = 0; i < top_k; ++i) {
            unary[output + i]      = acc_v[i];
            candidates[output + i] = acc_i[i];
        }
    }
}

using DFlash2SelectorPartialSort =
    cub::BlockRadixSort<unsigned long long, kDFlash2SelectorBlock,
                        kDFlash2SelectorItemsPerThread>;
using DFlash2SelectorGroupSort =
    cub::BlockRadixSort<unsigned long long, kDFlash2SelectorBlock,
                        kDFlash2SelectorGroupItemsPerThread>;

__device__ __forceinline__ int dflash2_selector_partial_offset(
    int partial_stride, int top_k, int column, int partial, int rank) {
    return ((column * partial_stride + partial) * top_k) + rank;
}

// Every CTA ranks one contiguous 1024-token tile for one independent draft column.
__launch_bounds__(kDFlash2SelectorBlock) __global__ void dflash2_selector_partial_kernel(
    const __nv_bfloat16* __restrict__ logits, unsigned long long* __restrict__ partial_keys,
    int* __restrict__ group_done, int token_domain, int physical_rows, int top_k,
    int partial_stride) {
    const int partial = static_cast<int>(blockIdx.x);
    const int column  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    if (partial == 0 && tid == 0) { group_done[column] = 0; }

    __shared__ typename DFlash2SelectorPartialSort::TempStorage sort_storage;
    unsigned long long keys[kDFlash2SelectorItemsPerThread];
    const int tile_start = partial * kDFlash2SelectorPartialTileItems;
#pragma unroll
    for (int item = 0; item < kDFlash2SelectorItemsPerThread; ++item) {
        const int token = tile_start + item * blockDim.x + tid;
        keys[item] = token < token_domain
                         ? sampling_sort_key(
                               __bfloat162float(logits[token + static_cast<std::int64_t>(column) *
                                                           physical_rows]),
                               token)
                         : 0ull;
    }
    DFlash2SelectorPartialSort(sort_storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kDFlash2SelectorItemsPerThread; ++item) {
        const int rank = tid * kDFlash2SelectorItemsPerThread + item;
        if (rank < top_k) {
            partial_keys[dflash2_selector_partial_offset(partial_stride, top_k, column, partial,
                                                         rank)] = keys[item];
        }
    }
}

// Merge 32 partial lists per CTA. The last group to finish also performs the exact final merge,
// avoiding a third launch while preserving one complete descending candidate order per column.
__launch_bounds__(kDFlash2SelectorBlock) __global__ void dflash2_selector_group_kernel(
    unsigned long long* __restrict__ partial_keys, int* __restrict__ group_done,
    float* __restrict__ unary, int* __restrict__ candidates, int top_k, int partial_count,
    int group_count, int partial_stride) {
    const int group  = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const int group_begin = group * kDFlash2SelectorPartialsPerGroup;
    int group_partials    = partial_count - group_begin;
    if (group_partials > kDFlash2SelectorPartialsPerGroup) {
        group_partials = kDFlash2SelectorPartialsPerGroup;
    }

    __shared__ typename DFlash2SelectorGroupSort::TempStorage sort_storage;
    __shared__ int is_last;
    __shared__ unsigned long long group_top[kDFlash2SelectorMaximumTopK];
    unsigned long long keys[kDFlash2SelectorGroupItemsPerThread];
    const int group_items = group_partials * top_k;
#pragma unroll
    for (int item = 0; item < kDFlash2SelectorGroupItemsPerThread; ++item) {
        const int source = item * blockDim.x + tid;
        if (source < group_items) {
            const int partial = group_begin + source / top_k;
            const int rank    = source - (source / top_k) * top_k;
            keys[item] = partial_keys[dflash2_selector_partial_offset(
                partial_stride, top_k, column, partial, rank)];
        } else {
            keys[item] = 0ull;
        }
    }
    DFlash2SelectorGroupSort(sort_storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kDFlash2SelectorGroupItemsPerThread; ++item) {
        const int rank = tid * kDFlash2SelectorGroupItemsPerThread + item;
        if (rank < top_k) { group_top[rank] = keys[item]; }
    }
    __syncthreads();
    if (tid == 0) {
        for (int rank = 0; rank < top_k; ++rank) {
            partial_keys[dflash2_selector_partial_offset(
                partial_stride, top_k, column, partial_count + group, rank)] = group_top[rank];
        }
        __threadfence();
        is_last = atomicAdd(&group_done[column], 1) + 1 == group_count;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_items = group_count * top_k;
#pragma unroll
    for (int item = 0; item < kDFlash2SelectorGroupItemsPerThread; ++item) {
        const int source = item * blockDim.x + tid;
        if (source < final_items) {
            const int source_group = source / top_k;
            const int rank         = source - source_group * top_k;
            keys[item] = partial_keys[dflash2_selector_partial_offset(
                partial_stride, top_k, column, partial_count + source_group, rank)];
        } else {
            keys[item] = 0ull;
        }
    }
    __syncthreads();
    DFlash2SelectorGroupSort(sort_storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kDFlash2SelectorGroupItemsPerThread; ++item) {
        const int rank = tid * kDFlash2SelectorGroupItemsPerThread + item;
        if (rank < top_k) {
            const unsigned long long key = keys[item];
            unary[column * top_k + rank] = sampling_key_float(key);
            candidates[column * top_k + rank] = sampling_key_index(key);
        }
    }
}

enum : int {
    kDenseSelector = 0,
    kQ4Selector    = 1,
    kW8Selector    = 2,
};

template <int Format>
__device__ __forceinline__ float codebook_value(const void* data, const void* scales, int row,
                                                int component, int rank, int padded_rank) {
    if constexpr (Format == kDenseSelector) {
        const auto* dense = static_cast<const __nv_bfloat16*>(data);
        return __bfloat162float(dense[static_cast<std::int64_t>(row) * rank + component]);
    } else if constexpr (Format == kQ4Selector) {
        const auto* codes = static_cast<const std::uint8_t*>(data);
        const auto* scale_bits = static_cast<const std::uint16_t*>(scales);
        const int groups_per_row = padded_rank / 64;
        const int group = component / 64;
        const int lane = (component & 63) >> 1;
        const std::uint8_t packed =
            codes[(static_cast<std::int64_t>(row) * groups_per_row + group) * 32 + lane];
        const int nibble = (component & 1) == 0 ? packed & 0x0f : packed >> 4;
        const int code = (nibble ^ 0x08) - 0x08;
        const float scale = __half2float(
            __ushort_as_half(scale_bits[static_cast<std::int64_t>(row) * groups_per_row + group]));
        return static_cast<float>(code) * scale;
    } else {
        const auto* codes = static_cast<const std::int8_t*>(data);
        const auto* scale_bits = static_cast<const std::uint16_t*>(scales);
        const int groups_per_row = padded_rank / 32;
        const int group = component / 32;
        const int code = static_cast<int>(
            codes[(static_cast<std::int64_t>(row) * groups_per_row + group) * 32 +
                  (component & 31)]);
        const float scale = __half2float(
            __ushort_as_half(scale_bits[static_cast<std::int64_t>(row) * groups_per_row + group]));
        return static_cast<float>(code) * scale;
    }
}

template <int Format>
__launch_bounds__(256) __global__ void select_path_kernel(
    const float* unary, const int* candidates, const __nv_bfloat16* hidden_p,
    const void* pred_book, const void* pred_scales, const void* succ_book,
    const void* succ_scales,
    const std::int32_t* anchors, const SamplingConfig* configs, const std::int32_t* lengths,
    const std::int32_t* current_extents, int* drafts, float* q_probs, int tokens, int top_k,
    int rank, int padded_rank) {
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int batch  = static_cast<int>(blockIdx.x);
    const int extent = current_extents[batch];
    int* row_drafts  = drafts + static_cast<std::int64_t>(batch) * tokens;
    float* row_q     = q_probs + static_cast<std::int64_t>(batch) * tokens * top_k;
    const int* row_candidates =
        candidates + static_cast<std::int64_t>(batch) * tokens * top_k;
    const float* row_unary = unary + static_cast<std::int64_t>(batch) * tokens * top_k;
    const __nv_bfloat16* row_hidden =
        hidden_p + static_cast<std::int64_t>(batch) * tokens * rank;
    for (int pos = tid; pos < tokens; pos += blockDim.x) { row_drafts[pos] = 0; }
    for (int item = tid; item < tokens * top_k; item += blockDim.x) { row_q[item] = 0.0f; }
    if (extent < 0 || extent > tokens) { return; }

    __shared__ float gated[kMaxRank];
    __shared__ float scores[kMaxTopK];
    __shared__ int predecessor;
    if (tid == 0) { predecessor = anchors[batch]; }
    __syncthreads();
    for (int pos = 0; pos < extent; ++pos) {
        const __nv_bfloat16* hid  = row_hidden + static_cast<std::int64_t>(pos) * rank;
        for (int r = tid; r < rank; r += blockDim.x) {
            gated[r] = codebook_value<Format>(pred_book, pred_scales, predecessor, r, rank,
                                              padded_rank) *
                       __bfloat162float(hid[r]);
        }
        __syncthreads();
        for (int j = warp; j < top_k; j += blockDim.x / 32) {
            const int candidate = row_candidates[j + pos * top_k];
            float score = 0.0f;
            for (int r = lane; r < rank; r += 32) {
                score +=
                    gated[r] * codebook_value<Format>(succ_book, succ_scales, candidate, r, rank,
                                                       padded_rank);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                score += __shfl_down_sync(0xffffffffu, score, offset);
            }
            if (lane == 0) { scores[j] = row_unary[j + pos * top_k] + score; }
        }
        __syncthreads();

        if (tid == 0) {
            const SamplingConfig cfg = configs[batch];
            int pick                 = 0;
            if (!(cfg.temperature > 0.0f)) {
                for (int j = 1; j < top_k; ++j) {
                    const int candidate      = row_candidates[j + pos * top_k];
                    const int best_candidate = row_candidates[pick + pos * top_k];
                    if (sampling_better(scores[j], candidate, scores[pick], best_candidate)) {
                        pick = j;
                    }
                }
                for (int j = 0; j < top_k; ++j) {
                    row_q[j + pos * top_k] = (j == pick) ? 1.0f : 0.0f;
                }
            } else {
                float peak = scores[0];
                for (int j = 1; j < top_k; ++j) { peak = fmaxf(peak, scores[j]); }
                float mass = 0.0f;
                float probabilities[kMaxTopK];
                for (int j = 0; j < top_k; ++j) {
                    probabilities[j] = __expf((scores[j] - peak) / cfg.temperature);
                    mass += probabilities[j];
                }
                const float inverse = mass > 0.0f ? 1.0f / mass : 0.0f;
                for (int j = 0; j < top_k; ++j) {
                    probabilities[j] *= inverse;
                    row_q[j + pos * top_k] = probabilities[j];
                }
                const float u = sampling_uniform(cfg.seed, lengths[batch] + 1 + pos,
                                                 kSamplePurposeDFlashSelect, 0u);
                float cumulative = 0.0f;
                pick             = top_k - 1;
                for (int j = 0; j < top_k; ++j) {
                    cumulative += probabilities[j];
                    if (u < cumulative) {
                        pick = j;
                        break;
                    }
                }
            }
            predecessor     = row_candidates[pick + pos * top_k];
            row_drafts[pos] = predecessor;
        }
        __syncthreads();
    }
}

__device__ inline int pick_residual(const int* candidate_ids, const float* probability, int count,
                                    const int* selector_ids, const float* selector_q, int top_k,
                                    float u) {
    float total = 0.0f;
    for (int j = 0; j < count; ++j) {
        float p = probability[j];
        for (int c = 0; c < top_k; ++c) {
            if (candidate_ids[j] == selector_ids[c]) { p -= selector_q[c]; }
        }
        total += fmaxf(p, 0.0f);
    }
    if (total <= 0.0f) {
        return sampling_pick_from_support(candidate_ids, probability, count, -1, u);
    }
    const float goal = u * total;
    float cumulative = 0.0f;
    int picked       = candidate_ids[0];
    for (int j = 0; j < count; ++j) {
        float p = probability[j];
        for (int c = 0; c < top_k; ++c) {
            if (candidate_ids[j] == selector_ids[c]) { p -= selector_q[c]; }
        }
        cumulative += fmaxf(p, 0.0f);
        picked = candidate_ids[j];
        if (goal < cumulative) { return picked; }
    }
    return picked;
}

__launch_bounds__(kSamplerBlock) __global__ void dflash2_accept_fallback_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* candidates,
    const float* q_probs, const std::int32_t* current_extents, const SamplingConfig* configs,
    std::int32_t* lengths, std::int32_t* anchors, std::int32_t* licensed,
    std::int32_t* licensed_counts, std::int32_t* accepted, std::int32_t token_domain,
    std::int32_t physical_rows, std::int32_t physical_k, std::int32_t top_k) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int batch = static_cast<int>(blockIdx.x);
    const int extent = current_extents[batch];
    if (extent < 0 || extent > physical_k) { return; }
    logits += static_cast<std::int64_t>(batch) * (physical_k + 1) * physical_rows;
    drafts += static_cast<std::int64_t>(batch) * physical_k;
    candidates += static_cast<std::int64_t>(batch) * physical_k * top_k;
    q_probs += static_cast<std::int64_t>(batch) * physical_k * top_k;
    licensed += static_cast<std::int64_t>(batch) * (physical_k + 1);
    const SamplingConfig cfg = configs[batch];

    __shared__ float stage_val[32 * kSamplerWideCandidates];
    __shared__ int stage_idx[32 * kSamplerWideCandidates];
    __shared__ float candidate_value[kSamplerWideCandidates];
    __shared__ int candidate_index[kSamplerWideCandidates];
    __shared__ float probability[kSamplerWideCandidates];
    __shared__ unsigned long long warp_keys[kSamplerBlock / 32];
    __shared__ int support_count;
    __shared__ int accepted_shared;
    __shared__ int done_shared;
    __shared__ int terminal_shared;
    __shared__ int initial_length_shared;

    if (tid == 0) {
        accepted_shared       = 0;
        done_shared           = 0;
        terminal_shared       = 0;
        initial_length_shared = lengths[batch];
    }
    __syncthreads();

    if (!(cfg.temperature > 0.0f)) {
        for (int column = 0; column <= extent; ++column) {
            unsigned long long key = 0;
            const std::int64_t base = static_cast<std::int64_t>(column) * physical_rows;
            for (int token = tid; token < token_domain; token += blockDim.x) {
                const float value = __bfloat162float(logits[base + token]);
                const unsigned long long candidate = sampling_sort_key(value, token);
                if (candidate > key) { key = candidate; }
            }
            key = sampling_block_max_key(key, warp_keys);
            if (tid == 0) {
                const int target = sampling_key_index(key);
                if (column < extent && target == drafts[column]) {
                    accepted_shared = column + 1;
                } else {
                    terminal_shared = target;
                    done_shared     = 1;
                }
            }
            __syncthreads();
            if (done_shared) { break; }
        }
    } else {
        for (int column = 0; column <= extent; ++column) {
            const std::int64_t base = static_cast<std::int64_t>(column) * physical_rows;
            sampling_build_truncated_block_wide<kSamplerWideCandidates>(
                logits, base, token_domain, cfg, stage_val, stage_idx, candidate_value,
                candidate_index, probability, &support_count, drafts, column);
            if (tid == 0 && done_shared == 0) {
                const int logical_position = initial_length_shared + column + 1;
                if (column < extent) {
                    const int draft = drafts[column];
                    float target_probability = 0.0f;
                    for (int j = 0; j < support_count; ++j) {
                        if (candidate_index[j] == draft) {
                            target_probability = probability[j];
                            break;
                        }
                    }
                    float proposal_probability = 0.0f;
                    for (int j = 0; j < top_k; ++j) {
                        if (candidates[j + column * top_k] == draft) {
                            proposal_probability = q_probs[j + column * top_k];
                            break;
                        }
                    }
                    const float accept_u = sampling_uniform(
                        cfg.seed, logical_position, kSamplePurposeSpeculativeAccept, 0u);
                    if (accept_u * proposal_probability < target_probability) {
                        accepted_shared = column + 1;
                    } else {
                        const float correction_u = sampling_uniform(
                            cfg.seed, logical_position, kSamplePurposeSpeculativeCorrection, 0u);
                        terminal_shared = pick_residual(
                            candidate_index, probability, support_count,
                            candidates + column * top_k, q_probs + column * top_k, top_k,
                            correction_u);
                        done_shared = 1;
                    }
                } else {
                    const float bonus_u = sampling_uniform(
                        cfg.seed, logical_position, kSamplePurposeSpeculativeBonus, 0u);
                    terminal_shared = sampling_pick_from_support(
                        candidate_index, probability, support_count, -1, bonus_u);
                    done_shared = 1;
                }
            }
            __syncthreads();
            if (done_shared) { break; }
        }
    }

    if (tid == 0) {
        for (int i = 0; i <= physical_k; ++i) { licensed[i] = 0; }
        for (int i = 0; i < accepted_shared; ++i) { licensed[i] = drafts[i]; }
        licensed[accepted_shared] = terminal_shared;
        const int produced        = accepted_shared + 1;
        licensed_counts[batch]    = produced;
        accepted[batch]           = accepted_shared;
        anchors[batch]            = terminal_shared;
        lengths[batch]            = initial_length_shared + produced;
        if (cfg.temperature > 0.0f && cfg.token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[licensed[i]], 1); }
        }
    }
}

struct DFlash2SelectStorage {
    int histogram[256];
    unsigned long long prefix;
    int rank;
    int output_count;
};

// Exact radix selection of the best `cap` unique (adjusted-logit, token-id) keys. The selected
// order is irrelevant until the final merge, which performs a complete descending sort.
template <int ItemsPerThread>
__device__ inline void dflash2_select_top_keys(
    const unsigned long long (&keys)[ItemsPerThread], int cap, unsigned long long* output,
    DFlash2SelectStorage& storage) {
    const int tid = static_cast<int>(threadIdx.x);
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
            const int rank = atomicAdd(&storage.output_count, 1);
            if (rank < cap) { output[rank] = key; }
        }
    }
    __syncthreads();
}

__device__ inline int dflash2_partial_offset(int partial_stride, int column, int partial,
                                             int rank) {
    return ((column * partial_stride + partial) * kSamplerWideCandidates) + rank;
}

// Stage 1: one CTA per 1024-token tile and verification column. All valid columns are independent
// because the round-local penalty overlay for column c is exactly drafts[0:c].
template <bool RadixSelect>
__launch_bounds__(kSamplerBlock) __global__ void dflash2_accept_partial_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, const SamplingConfig* configs,
    unsigned long long* partial_keys, std::int32_t* group_done, std::int32_t token_domain,
    std::int32_t physical_rows, std::int32_t physical_k, std::int32_t partial_stride) {
    const int partial        = static_cast<int>(blockIdx.x);
    const int flat_column    = static_cast<int>(blockIdx.y);
    const int columns_per_row = physical_k + 1;
    const int batch          = flat_column / columns_per_row;
    const int column         = flat_column - batch * columns_per_row;
    const int extent         = current_extents[batch];
    if (partial == 0 && threadIdx.x == 0) { group_done[flat_column] = 0; }
    if (extent < 0 || extent > physical_k || column > extent) { return; }

    const SamplingConfig cfg = configs[batch];
    const bool greedy        = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain,
                                                         kSamplerWideCandidates);
    const std::int64_t base = static_cast<std::int64_t>(flat_column) * physical_rows;
    const std::int32_t* row_drafts = drafts + static_cast<std::int64_t>(batch) * physical_k;
    const int tile_start = partial * kSamplerWidePartialTileItems;

    __shared__ typename SamplingWideSort::TempStorage sort_storage;
    __shared__ DFlash2SelectStorage select_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    unsigned long long keys[kSamplerWideItemsPerThread];
#pragma unroll
    for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
        const int token = tile_start + item * blockDim.x + threadIdx.x;
        if (token < token_domain) {
            const float raw = __bfloat162float(logits[base + token]);
            const float value = greedy ? raw : sampling_adjusted_logit(
                                                   raw, token, cfg, row_drafts, column);
            keys[item] = sampling_sort_key(value, token);
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
            partial_keys[dflash2_partial_offset(partial_stride, flat_column, partial, 0)] = best;
        }
        return;
    }

    const int output_offset = dflash2_partial_offset(partial_stride, flat_column, partial, 0);
    if constexpr (RadixSelect) {
        dflash2_select_top_keys(keys, cap, partial_keys + output_offset, select_storage);
    } else {
        SamplingWideSort(sort_storage).SortDescending(keys);
#pragma unroll
        for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
            const int rank = static_cast<int>(threadIdx.x) * kSamplerWideItemsPerThread + item;
            if (rank < cap) { partial_keys[output_offset + rank] = keys[item]; }
        }
    }
}

// Stage 2: merge sixteen partial lists per CTA. The last group for a column performs the small
// final merge and publishes that column's normalized exact top-64 distribution.
template <bool RadixSelect>
__launch_bounds__(kSamplerGroupBlock) __global__ void dflash2_accept_group_kernel(
    const std::int32_t* current_extents, const SamplingConfig* configs,
    unsigned long long* partial_keys, std::int32_t* group_done, std::int32_t* dist_indices,
    float* dist_probabilities, std::int32_t* dist_support, std::int32_t token_domain,
    std::int32_t physical_k, std::int32_t partial_blocks, std::int32_t group_count,
    std::int32_t partial_stride) {
    const int group           = static_cast<int>(blockIdx.x);
    const int flat_column     = static_cast<int>(blockIdx.y);
    const int tid             = static_cast<int>(threadIdx.x);
    const int columns_per_row = physical_k + 1;
    const int batch           = flat_column / columns_per_row;
    const int column          = flat_column - batch * columns_per_row;
    const int extent          = current_extents[batch];
    if (extent < 0 || extent > physical_k || column > extent) { return; }

    const SamplingConfig cfg = configs[batch];
    const bool greedy        = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain,
                                                         kSamplerWideCandidates);
    const int group_begin = group * kSamplerWidePartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerWidePartialsPerGroup) {
        group_partials = kSamplerWidePartialsPerGroup;
    }

    __shared__ typename SamplingWideSort::TempStorage sort_storage;
    __shared__ DFlash2SelectStorage select_storage;
    __shared__ float candidate_value[kSamplerWideCandidates];
    __shared__ int candidate_index[kSamplerWideCandidates];
    __shared__ float probability[kSamplerWideCandidates];
    __shared__ int support_count;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[kSamplerGroupBlock / 32];
    unsigned long long keys[kSamplerWideItemsPerThread];

    if (greedy) {
        unsigned long long best = 0ull;
        for (int partial = tid; partial < group_partials; partial += blockDim.x) {
            const int offset = dflash2_partial_offset(
                partial_stride, flat_column, group_begin + partial, 0);
            if (partial_keys[offset] > best) { best = partial_keys[offset]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            partial_keys[dflash2_partial_offset(
                partial_stride, flat_column, partial_blocks + group, 0)] = best;
            __threadfence();
            is_last = atomicAdd(&group_done[flat_column], 1) + 1 == group_count;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int candidate = tid; candidate < group_count; candidate += blockDim.x) {
            const int offset = dflash2_partial_offset(
                partial_stride, flat_column, partial_blocks + candidate, 0);
            if (partial_keys[offset] > best) { best = partial_keys[offset]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int dist_base       = flat_column * kSamplerWideCandidates;
            dist_indices[dist_base]   = sampling_key_index(best);
            dist_probabilities[dist_base] = 1.0f;
            dist_support[flat_column] = 1;
            group_done[flat_column]   = 0;
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
            keys[item] = partial_keys[dflash2_partial_offset(
                partial_stride, flat_column, partial, rank)];
        } else {
            keys[item] = 0ull;
        }
    }
    const int group_output = dflash2_partial_offset(
        partial_stride, flat_column, partial_blocks + group, 0);
    if constexpr (RadixSelect) {
        dflash2_select_top_keys(keys, cap, partial_keys + group_output, select_storage);
    } else {
        SamplingWideSort(sort_storage).SortDescending(keys);
#pragma unroll
        for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
            const int rank = tid * kSamplerWideItemsPerThread + item;
            if (rank < cap) { partial_keys[group_output + rank] = keys[item]; }
        }
        __syncthreads();
    }

    if (tid == 0) {
        __threadfence();
        is_last = atomicAdd(&group_done[flat_column], 1) + 1 == group_count;
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
            keys[item] = partial_keys[dflash2_partial_offset(
                partial_stride, flat_column, partial_blocks + candidate, rank)];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingWideSort(sort_storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kSamplerWideItemsPerThread; ++item) {
        const int rank = tid * kSamplerWideItemsPerThread + item;
        if (rank < cap) {
            candidate_value[rank] = sampling_key_float(keys[item]);
            candidate_index[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();
    sampling_normalize_support(cfg, candidate_value, candidate_index, probability,
                               &support_count, cap);
    const int dist_base = flat_column * kSamplerWideCandidates;
    for (int rank = tid; rank < support_count; rank += blockDim.x) {
        dist_indices[dist_base + rank]       = candidate_index[rank];
        dist_probabilities[dist_base + rank] = probability[rank];
    }
    if (tid == 0) {
        dist_support[flat_column] = support_count;
        group_done[flat_column]   = 0;
    }
}

// Stage 3: only acceptance ordering is sequential. Distribution construction has completed for
// every valid column, so this kernel touches at most 16*64 values per row.
__global__ void dflash2_accept_commit_kernel(
    const std::int32_t* drafts, const std::int32_t* candidates, const float* q_probs,
    const std::int32_t* current_extents, const SamplingConfig* configs,
    const std::int32_t* dist_indices, const float* dist_probabilities,
    const std::int32_t* dist_support, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed, std::int32_t* licensed_counts, std::int32_t* accepted,
    std::int32_t physical_k, std::int32_t top_k) {
    if (threadIdx.x != 0) { return; }
    const int batch  = static_cast<int>(blockIdx.x);
    const int extent = current_extents[batch];
    if (extent < 0 || extent > physical_k) { return; }

    drafts += static_cast<std::int64_t>(batch) * physical_k;
    candidates += static_cast<std::int64_t>(batch) * physical_k * top_k;
    q_probs += static_cast<std::int64_t>(batch) * physical_k * top_k;
    licensed += static_cast<std::int64_t>(batch) * (physical_k + 1);
    const SamplingConfig cfg = configs[batch];
    const int initial_length = lengths[batch];
    const int flat_base      = batch * (physical_k + 1);
    int accepted_count       = 0;
    int terminal             = 0;

    if (!(cfg.temperature > 0.0f)) {
        for (int column = 0; column <= extent; ++column) {
            const int target = dist_indices[
                (flat_base + column) * kSamplerWideCandidates];
            if (column < extent && target == drafts[column]) {
                accepted_count = column + 1;
            } else {
                terminal = target;
                break;
            }
        }
    } else {
        for (int column = 0; column <= extent; ++column) {
            const int flat_column = flat_base + column;
            const int dist_base   = flat_column * kSamplerWideCandidates;
            const int support     = dist_support[flat_column];
            const int* target_ids = dist_indices + dist_base;
            const float* target_p = dist_probabilities + dist_base;
            const int logical_position = initial_length + column + 1;
            if (column < extent) {
                const int draft = drafts[column];
                float target_probability = 0.0f;
                for (int rank = 0; rank < support; ++rank) {
                    if (target_ids[rank] == draft) {
                        target_probability = target_p[rank];
                        break;
                    }
                }
                float proposal_probability = 0.0f;
                for (int rank = 0; rank < top_k; ++rank) {
                    if (candidates[column * top_k + rank] == draft) {
                        proposal_probability = q_probs[column * top_k + rank];
                        break;
                    }
                }
                const float accept_u = sampling_uniform(
                    cfg.seed, logical_position, kSamplePurposeSpeculativeAccept, 0u);
                if (accept_u * proposal_probability < target_probability) {
                    accepted_count = column + 1;
                    continue;
                }
                const float correction_u = sampling_uniform(
                    cfg.seed, logical_position, kSamplePurposeSpeculativeCorrection, 0u);
                terminal = pick_residual(target_ids, target_p, support,
                                         candidates + column * top_k,
                                         q_probs + column * top_k, top_k, correction_u);
                break;
            }
            const float bonus_u = sampling_uniform(
                cfg.seed, logical_position, kSamplePurposeSpeculativeBonus, 0u);
            terminal = sampling_pick_from_support(target_ids, target_p, support, -1, bonus_u);
        }
    }

    for (int i = 0; i <= physical_k; ++i) { licensed[i] = 0; }
    for (int i = 0; i < accepted_count; ++i) { licensed[i] = drafts[i]; }
    licensed[accepted_count] = terminal;
    const int produced       = accepted_count + 1;
    licensed_counts[batch]   = produced;
    accepted[batch]          = accepted_count;
    anchors[batch]           = terminal;
    lengths[batch]           = initial_length + produced;
    if (cfg.temperature > 0.0f && cfg.token_counts != nullptr) {
        for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[licensed[i]], 1); }
    }
}

} // namespace

__global__ void dflash2_fuse_context_copy_kernel(
    const std::int32_t* learned_drafts,
    const std::int32_t* learned_candidates,
    const float* learned_q_probs,
    const std::int32_t* context_tokens,
    const std::int32_t* context_match_lengths,
    const std::int32_t* context_extents,
    std::int32_t minimum_match,
    std::int32_t* drafts,
    std::int32_t* candidates,
    float* q_probs,
    int learned,
    int target,
    int top_k) {
    const int batch = static_cast<int>(blockIdx.x);
    const bool use_context = context_match_lengths[batch] >= minimum_match;
    int context_extent = use_context ? context_extents[batch] : 0;
    context_extent = max(0, min(context_extent, target));

    for (int flat = static_cast<int>(threadIdx.x); flat < target * top_k;
         flat += static_cast<int>(blockDim.x)) {
        const int rank = flat % top_k;
        const int column = flat / top_k;
        const int output = rank + top_k * (column + target * batch);
        if (column < context_extent) {
            const std::int32_t token = context_tokens[column + target * batch];
            candidates[output] = rank == 0 ? token : 0;
            q_probs[output] = rank == 0 ? 1.0F : 0.0F;
        } else if (!use_context && column < learned) {
            const int input = rank + top_k * (column + learned * batch);
            candidates[output] = learned_candidates[input];
            q_probs[output] = learned_q_probs[input];
        } else {
            candidates[output] = 0;
            q_probs[output] = 0.0F;
        }
        if (rank == 0) {
            const int draft_output = column + target * batch;
            if (column < context_extent) {
                drafts[draft_output] = context_tokens[draft_output];
            } else if (!use_context && column < learned) {
                drafts[draft_output] = learned_drafts[column + learned * batch];
            } else {
                drafts[draft_output] = 0;
            }
        }
    }
}

void dflash2_fuse_context_copy_launch(const Tensor& learned_drafts,
                                      const Tensor& learned_candidates,
                                      const Tensor& learned_q_probs,
                                      const Tensor& context_tokens,
                                      const Tensor& context_match_lengths,
                                      const Tensor& context_extents,
                                      std::int32_t minimum_match,
                                      Tensor& drafts,
                                      Tensor& candidates,
                                      Tensor& q_probs,
                                      cudaStream_t stream) {
    dflash2_fuse_context_copy_kernel<<<static_cast<unsigned>(drafts.ne[1]), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(learned_drafts.data),
        static_cast<const std::int32_t*>(learned_candidates.data),
        static_cast<const float*>(learned_q_probs.data),
        static_cast<const std::int32_t*>(context_tokens.data),
        static_cast<const std::int32_t*>(context_match_lengths.data),
        static_cast<const std::int32_t*>(context_extents.data), minimum_match,
        static_cast<std::int32_t*>(drafts.data),
        static_cast<std::int32_t*>(candidates.data), static_cast<float*>(q_probs.data),
        learned_drafts.ne[0], drafts.ne[0], candidates.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_select_path_launch(
    const Tensor& logits, const Tensor& hidden, const Weight& proj,
    const Weight& predecessor_codebook, const Weight& successor_codebook, const Tensor& anchors,
    const SamplingConfig* configs, const Tensor& lengths, const Tensor& current_extents,
    std::int32_t token_domain, std::int32_t top_k, Tensor& unary_scratch,
    Tensor& hidden_scratch, Tensor& drafts, Tensor& candidates, Tensor& q_probs,
    const Tensor& partial_keys, const Tensor& group_done, cudaStream_t stream) {
    const int physical_rows = logits.ne[0];
    const int tokens  = logits.ne[1];
    const int batch   = logits.ne[2];
    const int hdim    = hidden.ne[0];
    const int rank    = proj.n;
    const int columns = tokens * batch;

    if (proj.qtype == QType::Q4G64_F16S || proj.qtype == QType::W8G32_F16S) {
        Tensor hidden_matrix = hidden.reshape({hdim, columns});
        Tensor projected_matrix = hidden_scratch.reshape({rank, columns});
        ginfer::ops::linear(hidden_matrix, proj, projected_matrix, stream);
    } else {
        const dim3 gemm_block(32, 1, 1);
        const dim3 gemm_grid(static_cast<unsigned>((rank + 31) / 32),
                             static_cast<unsigned>(columns), 1u);
        gemm_pt_a_ginfer_b_kernel<<<gemm_grid, gemm_block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(proj.qdata),
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<__nv_bfloat16*>(hidden_scratch.data), rank, hdim, columns);
        CUDA_CHECK(cudaGetLastError());
    }

    if (partial_keys.data != nullptr) {
        const int partial_count  = dflash2_selector_partial_count(token_domain);
        const int group_count    = dflash2_selector_group_count(partial_count);
        const int partial_stride = dflash2_selector_partial_stride(token_domain);
        const dim3 partial_grid(static_cast<unsigned>(partial_count),
                                static_cast<unsigned>(columns), 1u);
        dflash2_selector_partial_kernel<<<partial_grid, kDFlash2SelectorBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<unsigned long long*>(partial_keys.data),
            static_cast<int*>(group_done.data), token_domain, physical_rows, top_k,
            partial_stride);
        CUDA_CHECK(cudaGetLastError());
        const dim3 group_grid(static_cast<unsigned>(group_count),
                              static_cast<unsigned>(columns), 1u);
        dflash2_selector_group_kernel<<<group_grid, kDFlash2SelectorBlock, 0, stream>>>(
            static_cast<unsigned long long*>(partial_keys.data),
            static_cast<int*>(group_done.data), static_cast<float*>(unary_scratch.data),
            static_cast<int*>(candidates.data), top_k, partial_count, group_count,
            partial_stride);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const dim3 topk_grid(static_cast<unsigned>(tokens), static_cast<unsigned>(batch), 1u);
        topk_column_fallback_kernel<<<topk_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<float*>(unary_scratch.data), static_cast<int*>(candidates.data),
            token_domain, physical_rows, tokens, top_k);
        CUDA_CHECK(cudaGetLastError());
    }

    if (proj.qtype == QType::Q4G64_F16S) {
        select_path_kernel<kQ4Selector><<<static_cast<unsigned>(batch), 256, 0, stream>>>(
            static_cast<const float*>(unary_scratch.data),
            static_cast<const int*>(candidates.data),
            static_cast<const __nv_bfloat16*>(hidden_scratch.data), predecessor_codebook.qdata,
            predecessor_codebook.scales, successor_codebook.qdata, successor_codebook.scales,
            static_cast<const std::int32_t*>(anchors.data), configs,
            static_cast<const std::int32_t*>(lengths.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(drafts.data), static_cast<float*>(q_probs.data), tokens,
            top_k, rank, predecessor_codebook.padded_shape[1]);
    } else if (proj.qtype == QType::W8G32_F16S) {
        select_path_kernel<kW8Selector><<<static_cast<unsigned>(batch), 256, 0, stream>>>(
            static_cast<const float*>(unary_scratch.data),
            static_cast<const int*>(candidates.data),
            static_cast<const __nv_bfloat16*>(hidden_scratch.data), predecessor_codebook.qdata,
            predecessor_codebook.scales, successor_codebook.qdata, successor_codebook.scales,
            static_cast<const std::int32_t*>(anchors.data), configs,
            static_cast<const std::int32_t*>(lengths.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(drafts.data), static_cast<float*>(q_probs.data), tokens,
            top_k, rank, predecessor_codebook.padded_shape[1]);
    } else {
        select_path_kernel<kDenseSelector><<<static_cast<unsigned>(batch), 256, 0, stream>>>(
            static_cast<const float*>(unary_scratch.data),
            static_cast<const int*>(candidates.data),
            static_cast<const __nv_bfloat16*>(hidden_scratch.data), predecessor_codebook.qdata,
            nullptr, successor_codebook.qdata, nullptr,
            static_cast<const std::int32_t*>(anchors.data), configs,
            static_cast<const std::int32_t*>(lengths.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(drafts.data), static_cast<float*>(q_probs.data), tokens,
            top_k, rank, rank);
    }
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_accept_launch(const Tensor& logits, const Tensor& drafts, const Tensor& candidates,
                           const Tensor& q_probs, const Tensor& current_extents,
                           const SamplingConfig* configs, std::int32_t token_domain,
                           Tensor& lengths, Tensor& anchors, Tensor& licensed,
                           Tensor& licensed_counts, Tensor& accepted,
                           const Tensor& partial_keys, const Tensor& group_done,
                           const Tensor& dist_indices, const Tensor& dist_probabilities,
                           const Tensor& dist_support, cudaStream_t stream) {
    const int physical_k = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const int top_k      = candidates.ne[0];
    const int columns    = (physical_k + 1) * batch;
    if (!dflash2_accept_hierarchy_ok(token_domain, columns)) {
        dflash2_accept_fallback_kernel<<<static_cast<unsigned>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(candidates.data),
            static_cast<const float*>(q_probs.data),
            static_cast<const std::int32_t*>(current_extents.data), configs,
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), token_domain, logits.ne[0], physical_k,
            top_k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int partial_blocks = sampler_wide_partial_count(token_domain);
    const int group_count    = sampler_wide_group_count(partial_blocks);
    const int partial_stride = dflash2_accept_partial_stride(token_domain);
    const dim3 partial_grid(static_cast<unsigned>(partial_blocks),
                            static_cast<unsigned>(columns));
    const bool radix_select = columns > 2;
    if (radix_select) {
        dflash2_accept_partial_kernel<true><<<partial_grid, kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data), configs,
            static_cast<unsigned long long*>(partial_keys.data),
            static_cast<std::int32_t*>(group_done.data), token_domain, logits.ne[0], physical_k,
            partial_stride);
    } else {
        dflash2_accept_partial_kernel<false><<<partial_grid, kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data), configs,
            static_cast<unsigned long long*>(partial_keys.data),
            static_cast<std::int32_t*>(group_done.data), token_domain, logits.ne[0], physical_k,
            partial_stride);
    }
    CUDA_CHECK(cudaGetLastError());

    const dim3 group_grid(static_cast<unsigned>(group_count), static_cast<unsigned>(columns));
    if (radix_select) {
        dflash2_accept_group_kernel<true><<<group_grid, kSamplerGroupBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(current_extents.data), configs,
            static_cast<unsigned long long*>(partial_keys.data),
            static_cast<std::int32_t*>(group_done.data),
            static_cast<std::int32_t*>(dist_indices.data),
            static_cast<float*>(dist_probabilities.data),
            static_cast<std::int32_t*>(dist_support.data), token_domain, physical_k,
            partial_blocks, group_count, partial_stride);
    } else {
        dflash2_accept_group_kernel<false><<<group_grid, kSamplerGroupBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(current_extents.data), configs,
            static_cast<unsigned long long*>(partial_keys.data),
            static_cast<std::int32_t*>(group_done.data),
            static_cast<std::int32_t*>(dist_indices.data),
            static_cast<float*>(dist_probabilities.data),
            static_cast<std::int32_t*>(dist_support.data), token_domain, physical_k,
            partial_blocks, group_count, partial_stride);
    }
    CUDA_CHECK(cudaGetLastError());

    dflash2_accept_commit_kernel<<<static_cast<unsigned>(batch), 1, 0, stream>>>(
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(candidates.data), static_cast<const float*>(q_probs.data),
        static_cast<const std::int32_t*>(current_extents.data), configs,
        static_cast<const std::int32_t*>(dist_indices.data),
        static_cast<const float*>(dist_probabilities.data),
        static_cast<const std::int32_t*>(dist_support.data),
        static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
        static_cast<std::int32_t*>(licensed.data),
        static_cast<std::int32_t*>(licensed_counts.data),
        static_cast<std::int32_t*>(accepted.data), physical_k, top_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ginfer::ops::detail
