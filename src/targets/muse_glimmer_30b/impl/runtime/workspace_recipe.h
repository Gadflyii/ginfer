#pragma once

#include "core/tensor.h"
#include "ginfer/ops/linear.h"
#include "targets/muse_glimmer_30b/impl/config.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace ginfer::targets::muse_glimmer_30b::detail {

struct TextLayersWorkspace {
    Tensor h;
    Tensor alt;
    Tensor q_flat;
    Tensor k_flat;
    Tensor q_normed;
    Tensor k_normed;
    Tensor v_flat;
    Tensor gate_flat;
    Tensor attn;
    Tensor attn_proj;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_mid;
    Tensor ffn_down;
    // Four fixed private arenas preserve the established projection streams while
    // NVFP4 routes materialize their per-call A4 activation.  Slot assignment is
    // main/query/output, key/up, value, and gate respectively.
    std::array<DeviceSpan, 4> linear_scratch{};
};

[[nodiscard]] inline std::size_t nvfp4_linear_scratch(std::int32_t output_rows,
                                                      std::int32_t input_rows,
                                                      std::int32_t columns) {
    return ops::linear_workspace_capacity_bytes(QType::NVFP4, output_rows, input_rows,
                                                ops::LinearPolicy::AllowA4, columns, columns);
}

template <class Allocator>
[[nodiscard]] TextLayersWorkspace allocate_text_layers_workspace(Allocator& workspace,
                                                                 std::int32_t width,
                                                                 std::int32_t batch = 1,
                                                                 bool nvfp4 = false) {
    const std::int32_t columns = width * batch;
    TextLayersWorkspace out{
        .h = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .alt = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .q_flat = workspace.alloc(DType::BF16, {TextConfig::query_size, columns}),
        .k_flat = workspace.alloc(DType::BF16, {TextConfig::kv_size, columns}),
        .q_normed = workspace.alloc(DType::BF16, {TextConfig::query_size, columns}),
        .k_normed = workspace.alloc(DType::BF16, {TextConfig::kv_size, columns}),
        .v_flat = workspace.alloc(DType::BF16, {TextConfig::kv_size, columns}),
        .gate_flat = workspace.alloc(DType::BF16, {TextConfig::query_size, columns}),
        .attn = workspace.alloc(
            DType::BF16, {TextConfig::head_dim, TextConfig::query_heads, width, batch}),
        .attn_proj = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .ffn_gate = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns}),
        .ffn_up = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns}),
        .ffn_mid = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns}),
        .ffn_down = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
    };
    if (!nvfp4) {
        return out;
    }

    const std::size_t query_gate =
        nvfp4_linear_scratch(TextConfig::query_size, TextConfig::hidden, columns);
    const std::size_t key_value =
        nvfp4_linear_scratch(TextConfig::kv_size, TextConfig::hidden, columns);
    const std::size_t attention_output =
        nvfp4_linear_scratch(TextConfig::hidden, TextConfig::query_size, columns);
    const std::size_t mlp_input =
        nvfp4_linear_scratch(TextConfig::intermediate, TextConfig::hidden, columns);
    const std::size_t mlp_output =
        nvfp4_linear_scratch(TextConfig::hidden, TextConfig::intermediate, columns);
    const std::array<std::size_t, 4> capacities = {
        std::max({query_gate, attention_output, mlp_input, mlp_output}),
        std::max(key_value, mlp_input),
        key_value,
        query_gate,
    };
    for (std::size_t index = 0; index < capacities.size(); ++index) {
        if (capacities[index] != 0) {
            out.linear_scratch[index] = workspace.alloc_bytes(capacities[index], 256);
        }
    }
    return out;
}

struct DFlashProjectionWorkspace {
    Tensor projected;
    Tensor context;
    Tensor k_flat;
    Tensor v_flat;
    Tensor k_normed;
};

template <class Allocator>
[[nodiscard]] DFlashProjectionWorkspace
allocate_dflash_projection_workspace(Allocator& workspace, std::int32_t width,
                                     std::int32_t batch = 1) {
    constexpr int kv_rows = DFlash2Config::kv_heads * TextConfig::head_dim;
    const std::int32_t columns = width * batch;
    return {
        .projected = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .context = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .k_flat = workspace.alloc(DType::BF16, {kv_rows, columns}),
        .v_flat = workspace.alloc(DType::BF16, {kv_rows, columns}),
        .k_normed = workspace.alloc(DType::BF16, {kv_rows, columns}),
    };
}

struct DFlashLayersWorkspace {
    Tensor h;
    Tensor conv_in;
    Tensor conv_out;
    Tensor dynamic;
    Tensor q_flat;
    Tensor k_flat;
    Tensor v_flat;
    Tensor q_normed;
    Tensor k_normed;
    Tensor attn;
    Tensor attn_proj;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_mid;
    Tensor ffn_down;
};

template <class Allocator>
[[nodiscard]] DFlashLayersWorkspace allocate_dflash_layers_workspace(Allocator& workspace,
                                                                     std::int32_t width,
                                                                     std::int32_t batch = 1) {
    constexpr int kv_rows = DFlash2Config::kv_heads * TextConfig::head_dim;
    const std::int32_t columns = width * batch;
    return {
        .h = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .conv_in = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .conv_out = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .dynamic =
            workspace.alloc(DType::BF16, {DFlash2Config::kernel_projection_rows, columns}),
        .q_flat = workspace.alloc(DType::BF16, {TextConfig::query_size, columns}),
        .k_flat = workspace.alloc(DType::BF16, {kv_rows, columns}),
        .v_flat = workspace.alloc(DType::BF16, {kv_rows, columns}),
        .q_normed = workspace.alloc(DType::BF16, {TextConfig::query_size, columns}),
        .k_normed = workspace.alloc(DType::BF16, {kv_rows, columns}),
        .attn = workspace.alloc(
            DType::BF16, {TextConfig::head_dim, TextConfig::query_heads, width, batch}),
        .attn_proj = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
        .ffn_gate = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns}),
        .ffn_up = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns}),
        .ffn_mid = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns}),
        .ffn_down = workspace.alloc(DType::BF16, {TextConfig::hidden, columns}),
    };
}

} // namespace ginfer::targets::muse_glimmer_30b::detail
