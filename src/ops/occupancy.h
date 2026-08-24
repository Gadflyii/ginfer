#pragma once

#include "core/gpu_capability.h"
#include "ops/kernel/gqa_attention_geometry.cuh"

namespace ginfer::ops {

[[nodiscard]] constexpr int gqa_runtime_decode_splits(int gqa_decode_splits_base,
                                                      int decode_split_scale) {
    const int scaled = gqa_decode_splits_base * decode_split_scale;
    return scaled < DecodeSplitsMax ? scaled : DecodeSplitsMax;
}

[[nodiscard]] inline int gqa_runtime_decode_splits(const OccupancyPolicy& occupancy,
                                                   int decode_split_scale) {
    return gqa_runtime_decode_splits(occupancy.gqa_decode_splits_base, decode_split_scale);
}

[[nodiscard]] inline int bidirectional_gqa_runtime_max_split(const OccupancyPolicy& occupancy) {
    return occupancy.gqa_decode_splits_base < DecodeSplitsMax ? occupancy.gqa_decode_splits_base
                                                              : DecodeSplitsMax;
}

[[nodiscard]] constexpr int gqa_eight_k_split_max(int sm_count, int decode_split_scale) {
    return (sm_count / 4) * decode_split_scale;
}

} // namespace ginfer::ops
