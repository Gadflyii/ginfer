#include "targets/muse_glimmer_30b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ginfer::targets::muse_glimmer_30b::detail {
namespace {

using artifact::NumericFormat;

std::string layer_name(std::size_t layer, std::string_view suffix) {
    return "text/layers/" + std::to_string(layer) + "/" + std::string(suffix);
}

std::string dflash_name(std::size_t layer, std::string_view suffix) {
    return "dflash2/layers/" + std::to_string(layer) + "/" + std::string(suffix);
}

std::string take_string(artifact::MaterializedArtifact& materialized,
                        artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

FrontendResourcePlan bind_frontend(artifact::Binder& binder) {
    return FrontendResourcePlan{
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json =
            artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
        .processor_config_json =
            artifact::bind_raw_resource(binder, "frontend/processor_config.json"),
    };
}

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - static_cast<std::size_t>(offset) < 4) {
        throw artifact::ArtifactError(std::string(label) + ": FP32 word is outside payload");
    }
    const std::byte* value = bytes.data() + static_cast<std::size_t>(offset);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8U) |
           (std::to_integer<std::uint32_t>(value[2]) << 16U) |
           (std::to_integer<std::uint32_t>(value[3]) << 24U);
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw artifact::ArtifactError(std::string(label) + ": divisor must be finite and positive");
    }
}

WeightPlan bind_text_weight(artifact::Binder& binder, std::string_view name,
                            NumericFormat format,
                            std::initializer_list<std::uint64_t> shape) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("Muse NVFP4 weight requires a paired input divisor");
    }
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

std::uint32_t bind_input_divisor(artifact::Binder& binder, std::string_view name) {
    const artifact::ObjectHandle handle = artifact::bind_tensor(
        binder, name, NumericFormat::FP32, {}, artifact::TensorPlacement::ValidateOnly);
    const std::uint32_t bits = read_u32_le(binder.payload(handle).data, 0, name);
    require_positive_finite(bits, name);
    return bits;
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name,
                             std::int32_t rows, std::int32_t columns,
                             std::uint32_t input_divisor_bits) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    require_positive_finite(weight_bits, name);
    return WeightPlan{.object = parent,
                      .format = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits = input_divisor_bits};
}

Weight materialized_text_weight(const artifact::MaterializedArtifact& materialized,
                                const WeightPlan& plan, std::int32_t rows,
                                std::int32_t columns) {
    if (plan.format != NumericFormat::NVFP4) {
        return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
    }

    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(plan.object));
    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = rows;
    out.k                    = columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = rows;
    out.shape[1]             = columns;
    out.padded_shape[0]      = rows;
    out.padded_shape[1]      = columns;
    out.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(plan.input_scale_divisor_bits);
    return out;
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               bool enable_dflash2) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = bind_frontend(binder);

    out.token_embedding = artifact::bind_device_tensor(binder, "text/token_embedding",
                                                       NumericFormat::W8G32_F16S,
                                                       {TextConfig::vocab, TextConfig::hidden});

    const bool nvfp4 = weights_profile == WeightsProfile::Nvfp4;

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& dst   = out.text_layers[layer];
        dst.full_attention   = TextConfig::is_full_attention(static_cast<int>(layer));
        dst.input_norm       = artifact::bind_device_tensor(
            binder, layer_name(layer, "input_norm"), NumericFormat::BF16, {TextConfig::hidden});
        if (nvfp4) {
            const std::uint32_t attention_input = bind_input_divisor(
                binder, layer_name(layer, "attention/input_projection/input_scale_divisor"));
            dst.attention.query = bind_nvfp4_weight(
                binder, layer_name(layer, "attention/query"), TextConfig::query_size,
                TextConfig::hidden, attention_input);
            dst.attention.key = bind_nvfp4_weight(
                binder, layer_name(layer, "attention/key"), TextConfig::kv_size,
                TextConfig::hidden, attention_input);
            dst.attention.value = bind_nvfp4_weight(
                binder, layer_name(layer, "attention/value"), TextConfig::kv_size,
                TextConfig::hidden, attention_input);
            dst.attention.gate = bind_nvfp4_weight(
                binder, layer_name(layer, "attention/gate"), TextConfig::query_size,
                TextConfig::hidden, attention_input);
            const std::uint32_t attention_output = bind_input_divisor(
                binder, layer_name(layer, "attention/output_projection/input_scale_divisor"));
            dst.attention.output = bind_nvfp4_weight(
                binder, layer_name(layer, "attention/output"), TextConfig::hidden,
                TextConfig::query_size, attention_output);
        } else {
            dst.attention.query = bind_text_weight(
                binder, layer_name(layer, "attention/query"), NumericFormat::Q4G64_F16S,
                {TextConfig::query_size, TextConfig::hidden});
            dst.attention.key = bind_text_weight(
                binder, layer_name(layer, "attention/key"), NumericFormat::Q4G64_F16S,
                {TextConfig::kv_size, TextConfig::hidden});
            dst.attention.value = bind_text_weight(
                binder, layer_name(layer, "attention/value"), NumericFormat::Q4G64_F16S,
                {TextConfig::kv_size, TextConfig::hidden});
            dst.attention.gate = bind_text_weight(
                binder, layer_name(layer, "attention/gate"), NumericFormat::Q4G64_F16S,
                {TextConfig::query_size, TextConfig::hidden});
            dst.attention.output = bind_text_weight(
                binder, layer_name(layer, "attention/output"), NumericFormat::Q4G64_F16S,
                {TextConfig::hidden, TextConfig::query_size});
        }
        dst.post_attention_norm = artifact::bind_device_tensor(
            binder, layer_name(layer, "post_attention_norm"), NumericFormat::BF16,
            {TextConfig::hidden});
        dst.pre_feedforward_norm = artifact::bind_device_tensor(
            binder, layer_name(layer, "pre_feedforward_norm"), NumericFormat::BF16,
            {TextConfig::hidden});
        if (nvfp4) {
            const std::uint32_t gate_up = bind_input_divisor(
                binder, layer_name(layer, "mlp/gate_up_projection/input_scale_divisor"));
            dst.mlp.gate = bind_nvfp4_weight(binder, layer_name(layer, "mlp/gate"),
                                              TextConfig::intermediate, TextConfig::hidden,
                                              gate_up);
            dst.mlp.up = bind_nvfp4_weight(binder, layer_name(layer, "mlp/up"),
                                            TextConfig::intermediate, TextConfig::hidden,
                                            gate_up);
            const std::uint32_t down = bind_input_divisor(
                binder, layer_name(layer, "mlp/down_projection/input_scale_divisor"));
            dst.mlp.down = bind_nvfp4_weight(binder, layer_name(layer, "mlp/down"),
                                              TextConfig::hidden, TextConfig::intermediate, down);
        } else {
            dst.mlp.gate = bind_text_weight(
                binder, layer_name(layer, "mlp/gate"), NumericFormat::Q4G64_F16S,
                {TextConfig::intermediate, TextConfig::hidden});
            dst.mlp.up = bind_text_weight(
                binder, layer_name(layer, "mlp/up"), NumericFormat::Q4G64_F16S,
                {TextConfig::intermediate, TextConfig::hidden});
            dst.mlp.down = bind_text_weight(
                binder, layer_name(layer, "mlp/down"), NumericFormat::Q4G64_F16S,
                {TextConfig::hidden, TextConfig::intermediate});
        }
        dst.post_feedforward_norm = artifact::bind_device_tensor(
            binder, layer_name(layer, "post_feedforward_norm"), NumericFormat::BF16,
            {TextConfig::hidden});
    }

    out.output_norm = artifact::bind_device_tensor(binder, "text/output_norm", NumericFormat::BF16,
                                                   {TextConfig::hidden});
    out.output_head = artifact::bind_device_tensor(binder, "text/output_head",
                                                   NumericFormat::W8G32_F16S,
                                                   {TextConfig::vocab, TextConfig::hidden});

    const artifact::TensorPlacement dflash2_placement =
        enable_dflash2 ? artifact::TensorPlacement::Device
                       : artifact::TensorPlacement::ValidateOnly;
    const NumericFormat dflash2_matrix_format =
        weights_profile == WeightsProfile::GroupwiseIntDFlashQ4 ||
            weights_profile == WeightsProfile::Nvfp4
            ? NumericFormat::Q4G64_F16S
            : NumericFormat::BF16;
    const auto bind_dflash2 = [&](std::string_view name, NumericFormat format,
                                  std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, dflash2_placement);
    };
    DFlash2Plan dflash2;
    constexpr int kCaptured = TextConfig::hidden * 5;
    dflash2.fc = bind_dflash2("dflash2/fc", NumericFormat::Q4G64_F16S,
                             {TextConfig::hidden, kCaptured});
    dflash2.hidden_norm =
        bind_dflash2("dflash2/hidden_norm", NumericFormat::BF16, {TextConfig::hidden});

    for (std::size_t layer = 0; layer < kDflashLayers; ++layer) {
        DFlash2LayerPlan& dst = dflash2.layers[layer];
        dst.input_layernorm = bind_dflash2(dflash_name(layer, "input_layernorm"),
                                           NumericFormat::BF16, {TextConfig::hidden});
        dst.q_proj = bind_dflash2(dflash_name(layer, "self_attn/q_proj"),
                                  NumericFormat::Q4G64_F16S,
                                  {TextConfig::query_size, TextConfig::hidden});
        dst.k_proj = bind_dflash2(
            dflash_name(layer, "self_attn/k_proj"), NumericFormat::Q4G64_F16S,
            {DFlash2Config::kv_heads * TextConfig::head_dim, TextConfig::hidden});
        dst.v_proj = bind_dflash2(
            dflash_name(layer, "self_attn/v_proj"), NumericFormat::Q4G64_F16S,
            {DFlash2Config::kv_heads * TextConfig::head_dim, TextConfig::hidden});
        dst.o_proj = bind_dflash2(dflash_name(layer, "self_attn/o_proj"),
                                  NumericFormat::Q4G64_F16S,
                                  {TextConfig::hidden, TextConfig::query_size});
        dst.q_norm = bind_dflash2(dflash_name(layer, "self_attn/q_norm"), NumericFormat::BF16,
                                  {TextConfig::head_dim});
        dst.k_norm = bind_dflash2(dflash_name(layer, "self_attn/k_norm"), NumericFormat::BF16,
                                  {TextConfig::head_dim});
        dst.attention_conv_base = bind_dflash2(
            dflash_name(layer, "attention_conv/base_kernel"), NumericFormat::BF16,
            {DFlash2Config::conv_kernel, DFlash2Config::conv_kernel, TextConfig::hidden});
        dst.attention_conv_proj = bind_dflash2(
            dflash_name(layer, "attention_conv/kernel_projection"), dflash2_matrix_format,
            {DFlash2Config::kernel_projection_rows, TextConfig::hidden});
        dst.post_attention_layernorm =
            bind_dflash2(dflash_name(layer, "post_attention_layernorm"), NumericFormat::BF16,
                         {TextConfig::hidden});
        dst.mlp_gate = bind_dflash2(dflash_name(layer, "mlp/gate_proj"),
                                    NumericFormat::Q4G64_F16S,
                                    {TextConfig::intermediate, TextConfig::hidden});
        dst.mlp_up = bind_dflash2(dflash_name(layer, "mlp/up_proj"),
                                  NumericFormat::Q4G64_F16S,
                                  {TextConfig::intermediate, TextConfig::hidden});
        dst.mlp_down = bind_dflash2(dflash_name(layer, "mlp/down_proj"),
                                    NumericFormat::Q4G64_F16S,
                                    {TextConfig::hidden, TextConfig::intermediate});
        dst.mlp_conv_base = bind_dflash2(
            dflash_name(layer, "mlp_conv/base_kernel"), NumericFormat::BF16,
            {DFlash2Config::conv_kernel, DFlash2Config::conv_kernel, TextConfig::hidden});
        dst.mlp_conv_proj = bind_dflash2(
            dflash_name(layer, "mlp_conv/kernel_projection"), dflash2_matrix_format,
            {DFlash2Config::kernel_projection_rows, TextConfig::hidden});
    }

    dflash2.norm = bind_dflash2("dflash2/norm", NumericFormat::BF16, {TextConfig::hidden});
    dflash2.selector_hidden = bind_dflash2(
        "dflash2/candidate_selector/hidden_projection", dflash2_matrix_format,
        {DFlash2Config::selector_rank, TextConfig::hidden});
    dflash2.predecessor_codebook = bind_dflash2(
        "dflash2/candidate_selector/predecessor_codebook", dflash2_matrix_format,
        {TextConfig::vocab, DFlash2Config::selector_rank});
    dflash2.successor_codebook = bind_dflash2(
        "dflash2/candidate_selector/successor_codebook", dflash2_matrix_format,
        {TextConfig::vocab, DFlash2Config::selector_rank});
    if (enable_dflash2) { out.dflash2.emplace(std::move(dflash2)); }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(WeightsProfile weights_profile, BindingPlan plan,
                                 artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    const NumericFormat dflash2_matrix_format =
        weights_profile == WeightsProfile::GroupwiseIntDFlashQ4 ||
            weights_profile == WeightsProfile::Nvfp4
            ? NumericFormat::Q4G64_F16S
            : NumericFormat::BF16;
    frontend.tokenizer_json        = take_string(backing, plan.frontend.tokenizer_json);
    frontend.tokenizer_config_json = take_string(backing, plan.frontend.tokenizer_config_json);
    frontend.chat_template_jinja   = take_string(backing, plan.frontend.chat_template_jinja);
    frontend.generation_config_json = take_string(backing, plan.frontend.generation_config_json);
    frontend.processor_config_json  = take_string(backing, plan.frontend.processor_config_json);

    runtime.weights_arena = &backing.device_arena();
    runtime.token_embedding = artifact::materialized_weight(
        backing, plan.token_embedding, NumericFormat::W8G32_F16S, TextConfig::vocab,
        TextConfig::hidden);

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& src = plan.text_layers[layer];
        TextLayerWeights& dst    = runtime.text_layers[layer];
        dst.full_attention       = src.full_attention;
        dst.input_norm           = artifact::materialized_tensor(backing, src.input_norm,
                                                                 NumericFormat::BF16, {TextConfig::hidden});
        dst.attention.query = materialized_text_weight(
            backing, src.attention.query, TextConfig::query_size, TextConfig::hidden);
        dst.attention.key = materialized_text_weight(
            backing, src.attention.key, TextConfig::kv_size, TextConfig::hidden);
        dst.attention.value = materialized_text_weight(
            backing, src.attention.value, TextConfig::kv_size, TextConfig::hidden);
        dst.attention.gate = materialized_text_weight(
            backing, src.attention.gate, TextConfig::query_size, TextConfig::hidden);
        dst.attention.output = materialized_text_weight(
            backing, src.attention.output, TextConfig::hidden, TextConfig::query_size);
        dst.post_attention_norm = artifact::materialized_tensor(
            backing, src.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        dst.pre_feedforward_norm = artifact::materialized_tensor(
            backing, src.pre_feedforward_norm, NumericFormat::BF16, {TextConfig::hidden});
        dst.mlp.gate = materialized_text_weight(backing, src.mlp.gate,
                                                TextConfig::intermediate, TextConfig::hidden);
        dst.mlp.up = materialized_text_weight(backing, src.mlp.up,
                                              TextConfig::intermediate, TextConfig::hidden);
        dst.mlp.down = materialized_text_weight(backing, src.mlp.down, TextConfig::hidden,
                                                TextConfig::intermediate);
        dst.post_feedforward_norm = artifact::materialized_tensor(
            backing, src.post_feedforward_norm, NumericFormat::BF16, {TextConfig::hidden});
    }

    runtime.output_norm = artifact::materialized_tensor(backing, plan.output_norm,
                                                        NumericFormat::BF16, {TextConfig::hidden});
    runtime.output_head = artifact::materialized_weight(
        backing, plan.output_head, NumericFormat::W8G32_F16S, TextConfig::vocab, TextConfig::hidden);

    if (!plan.dflash2.has_value()) { return; }
    runtime.dflash2_loaded      = true;
    const DFlash2Plan& dflash2 = *plan.dflash2;
    constexpr int kCaptured = TextConfig::hidden * 5;
    runtime.dflash2.fc      = artifact::materialized_weight(
        backing, dflash2.fc, NumericFormat::Q4G64_F16S, TextConfig::hidden, kCaptured);
    runtime.dflash2.hidden_norm = artifact::materialized_tensor(
        backing, dflash2.hidden_norm, NumericFormat::BF16, {TextConfig::hidden});

    const int dflash_kv = DFlash2Config::kv_heads * TextConfig::head_dim;
    for (std::size_t layer = 0; layer < kDflashLayers; ++layer) {
        const DFlash2LayerPlan& src = dflash2.layers[layer];
        DFlash2LayerWeights& dst    = runtime.dflash2.layers[layer];
        dst.input_layernorm         = artifact::materialized_tensor(
            backing, src.input_layernorm, NumericFormat::BF16, {TextConfig::hidden});
        dst.q_proj = artifact::materialized_weight(backing, src.q_proj, NumericFormat::Q4G64_F16S,
                                                   TextConfig::query_size, TextConfig::hidden);
        dst.k_proj = artifact::materialized_weight(backing, src.k_proj, NumericFormat::Q4G64_F16S,
                                                   dflash_kv, TextConfig::hidden);
        dst.v_proj = artifact::materialized_weight(backing, src.v_proj, NumericFormat::Q4G64_F16S,
                                                   dflash_kv, TextConfig::hidden);
        dst.o_proj = artifact::materialized_weight(backing, src.o_proj, NumericFormat::Q4G64_F16S,
                                                   TextConfig::hidden, TextConfig::query_size);
        dst.q_norm = artifact::materialized_tensor(backing, src.q_norm, NumericFormat::BF16,
                                                   {TextConfig::head_dim});
        dst.k_norm = artifact::materialized_tensor(backing, src.k_norm, NumericFormat::BF16,
                                                   {TextConfig::head_dim});
        dst.attention_conv_base = artifact::materialized_tensor(
            backing, src.attention_conv_base, NumericFormat::BF16,
            {TextConfig::hidden, DFlash2Config::conv_kernel, DFlash2Config::conv_kernel});
        dst.attention_conv_proj = artifact::materialized_weight(
            backing, src.attention_conv_proj, dflash2_matrix_format,
            DFlash2Config::kernel_projection_rows, TextConfig::hidden);
        dst.post_attention_layernorm = artifact::materialized_tensor(
            backing, src.post_attention_layernorm, NumericFormat::BF16, {TextConfig::hidden});
        dst.mlp_gate = artifact::materialized_weight(
            backing, src.mlp_gate, NumericFormat::Q4G64_F16S, TextConfig::intermediate,
            TextConfig::hidden);
        dst.mlp_up = artifact::materialized_weight(backing, src.mlp_up, NumericFormat::Q4G64_F16S,
                                                   TextConfig::intermediate, TextConfig::hidden);
        dst.mlp_down = artifact::materialized_weight(
            backing, src.mlp_down, NumericFormat::Q4G64_F16S, TextConfig::hidden,
            TextConfig::intermediate);
        dst.mlp_conv_base = artifact::materialized_tensor(
            backing, src.mlp_conv_base, NumericFormat::BF16,
            {TextConfig::hidden, DFlash2Config::conv_kernel, DFlash2Config::conv_kernel});
        dst.mlp_conv_proj = artifact::materialized_weight(
            backing, src.mlp_conv_proj, dflash2_matrix_format,
            DFlash2Config::kernel_projection_rows, TextConfig::hidden);
    }

    runtime.dflash2.norm = artifact::materialized_tensor(backing, dflash2.norm,
                                                         NumericFormat::BF16, {TextConfig::hidden});
    runtime.dflash2.selector_hidden = artifact::materialized_weight(
        backing, dflash2.selector_hidden, dflash2_matrix_format,
        DFlash2Config::selector_rank, TextConfig::hidden);
    runtime.dflash2.predecessor_codebook = artifact::materialized_weight(
        backing, dflash2.predecessor_codebook, dflash2_matrix_format, TextConfig::vocab,
        DFlash2Config::selector_rank);
    runtime.dflash2.successor_codebook = artifact::materialized_weight(
        backing, dflash2.successor_codebook, dflash2_matrix_format, TextConfig::vocab,
        DFlash2Config::selector_rank);
}

} // namespace ginfer::targets::muse_glimmer_30b::detail
