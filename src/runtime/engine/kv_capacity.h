#pragma once

#include "runtime/contract/types.h"

#include <cstddef>

namespace ginfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

} // namespace ginfer::runtime
