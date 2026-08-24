#pragma once

#include "core/layout.h"
#include "ginfer/ops/sampling.h"
#include "targets/muse_glimmer_30b/impl/config.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ginfer::targets::muse_glimmer_30b::detail {

// All Program-owned device state is one startup-planned persistent region. Keeping the region
// recipe shared by sequence planning and binding prevents per-field cudaMallocs from escaping the
// Engine's frozen reservation and memory report.
struct ProgramStorageLayout {
    TensorRegion sample_ids;
    TensorRegion sample_positions;
    LayoutRegion sampling_configs;
    TensorRegion token_counts;
    TensorRegion table_rows;
    TensorRegion q_norm_weight;
    TensorRegion k_norm_weight;
    TensorRegion embed_norm_weight;
    TensorRegion decode_token;
    TensorRegion decode_position;
    TensorRegion decode_embed_raw;
    TensorRegion decode_hidden;
    std::size_t bytes = 0;
};

[[nodiscard]] inline ProgramStorageLayout plan_program_storage(std::uint32_t max_concurrency) {
    if (max_concurrency == 0 || max_concurrency > 8) {
        throw std::invalid_argument("Muse Program storage concurrency must be in [1,8]");
    }
    const auto lanes = static_cast<std::int32_t>(max_concurrency);
    LayoutBuilder layout;
    ProgramStorageLayout out;
    out.sample_ids       = layout.add_tensor(DType::I32, {lanes}, 256, "sample ids");
    out.sample_positions = layout.add_tensor(DType::I32, {lanes}, 256, "sample positions");
    out.sampling_configs = layout.add(sizeof(ops::SamplingConfig) * max_concurrency, 256,
                                      "sampling configs");
    out.token_counts = layout.add_tensor(DType::I32, {TextConfig::vocab, lanes}, 256,
                                         "sampling token counts");
    out.table_rows = layout.add_tensor(DType::I32, {lanes}, 256, "KV table rows");
    out.q_norm_weight =
        layout.add_tensor(DType::BF16, {TextConfig::head_dim}, 256, "Q norm synthesis");
    out.k_norm_weight =
        layout.add_tensor(DType::BF16, {TextConfig::head_dim}, 256, "K norm synthesis");
    out.embed_norm_weight =
        layout.add_tensor(DType::BF16, {TextConfig::hidden}, 256, "embedding norm synthesis");
    out.decode_token = layout.add_tensor(DType::I32, {lanes}, 256, "decode tokens");
    out.decode_position =
        layout.add_tensor(DType::I32, {1, lanes}, 256, "decode positions");
    out.decode_embed_raw =
        layout.add_tensor(DType::BF16, {TextConfig::hidden, lanes}, 256,
                          "decode raw embeddings");
    out.decode_hidden =
        layout.add_tensor(DType::BF16, {TextConfig::hidden, lanes}, 256, "decode hidden");
    out.bytes = layout.finish(256, "Muse Program storage");
    return out;
}

} // namespace ginfer::targets::muse_glimmer_30b::detail
