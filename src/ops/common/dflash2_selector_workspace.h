#pragma once

// Private workspace geometry for the DFlash2 unary top-k hierarchy. The registered Muse
// selector ranks at most sixteen candidates for each of up to 15*8 independent draft columns.

#include "ops/common/math.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ginfer::ops {

inline constexpr int kDFlash2SelectorBlock            = 256;
inline constexpr int kDFlash2SelectorItemsPerThread   = 4;
inline constexpr int kDFlash2SelectorPartialTileItems =
    kDFlash2SelectorBlock * kDFlash2SelectorItemsPerThread;
inline constexpr int kDFlash2SelectorGroupItemsPerThread = 2;
inline constexpr int kDFlash2SelectorGroupTileItems =
    kDFlash2SelectorBlock * kDFlash2SelectorGroupItemsPerThread;
inline constexpr int kDFlash2SelectorPartialsPerGroup = 32;
inline constexpr int kDFlash2SelectorMaximumTopK      = 16;
inline constexpr int kDFlash2SelectorMaximumDrafts    = 16;
inline constexpr int kDFlash2SelectorMaximumBatch     = 8;
inline constexpr int kDFlash2SelectorMaximumColumns   =
    kDFlash2SelectorMaximumDrafts * kDFlash2SelectorMaximumBatch;

__host__ __device__ inline std::int32_t
dflash2_selector_partial_count(std::int32_t token_domain) {
    return div_up(token_domain, kDFlash2SelectorPartialTileItems);
}

__host__ __device__ inline std::int32_t
dflash2_selector_group_count(std::int32_t partial_count) {
    return div_up(partial_count, kDFlash2SelectorPartialsPerGroup);
}

__host__ __device__ inline bool dflash2_selector_hierarchy_ok(
    std::int32_t token_domain, std::int32_t top_k, std::int32_t columns) {
    if (token_domain <= kDFlash2SelectorPartialTileItems || top_k <= 0 ||
        top_k > kDFlash2SelectorMaximumTopK || top_k > token_domain || columns <= 0 ||
        columns > kDFlash2SelectorMaximumColumns) {
        return false;
    }
    const std::int32_t partials = dflash2_selector_partial_count(token_domain);
    const std::int32_t groups   = dflash2_selector_group_count(partials);
    return partials > 0 && groups > 0 &&
           kDFlash2SelectorPartialsPerGroup * top_k <=
               kDFlash2SelectorGroupTileItems &&
           groups * top_k <= kDFlash2SelectorGroupTileItems;
}

__host__ __device__ inline std::int32_t
dflash2_selector_partial_stride(std::int32_t token_domain) {
    const std::int32_t partials = dflash2_selector_partial_count(token_domain);
    return partials + dflash2_selector_group_count(partials);
}

} // namespace ginfer::ops
