#pragma once

// Private workspace geometry for the DFlash2 target-distribution hierarchy. The hierarchy keeps
// the exact best 64 (adjusted-logit, token-id) keys for every verification column, then leaves
// only the inherently sequential accept/correct/bonus state transition to one small row kernel.

#include "ops/common/math.h"
#include "ops/common/sampling_workspace.h"

#include <cstdint>

namespace ginfer::ops {

inline constexpr int kDFlash2AcceptMaxDrafts = 15;
inline constexpr int kDFlash2AcceptMaxBatch  = 8;
inline constexpr int kDFlash2AcceptMaxColumns =
    (kDFlash2AcceptMaxDrafts + 1) * kDFlash2AcceptMaxBatch;

__host__ __device__ inline bool dflash2_accept_hierarchy_ok(std::int32_t token_domain,
                                                            std::int32_t columns) {
    if (token_domain <= kSamplerTileItems || columns <= 0 ||
        columns > kDFlash2AcceptMaxColumns) {
        return false;
    }
    const std::int32_t partials = sampler_wide_partial_count(token_domain);
    const std::int32_t groups   = sampler_wide_group_count(partials);
    return partials > 0 && groups > 0 &&
           kSamplerWidePartialsPerGroup * kSamplerWideCandidates <=
               kSamplerWideMergeTileItems &&
           groups * kSamplerWideCandidates <= kSamplerWideMergeTileItems;
}

__host__ __device__ inline std::int32_t
dflash2_accept_partial_stride(std::int32_t token_domain) {
    const std::int32_t partials = sampler_wide_partial_count(token_domain);
    return partials + sampler_wide_group_count(partials);
}

} // namespace ginfer::ops
