#pragma once

#include <ginfer/targets/muse_glimmer_30b/package.h>

#include "core/paged_kv_cache.h"

namespace ginfer::targets::muse_glimmer_30b::detail {

[[nodiscard]] SequencePlanner make_sequence_planner(DeviceContext& device,
                                                    const EngineOptions& options,
                                                    WeightsProfile weights_profile);

[[nodiscard]] PagedKVPoolSpec make_text_kv_spec(const SequenceLayout& layout);

} // namespace ginfer::targets::muse_glimmer_30b::detail
