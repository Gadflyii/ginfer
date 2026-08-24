#pragma once

#include <ginfer/targets/muse_glimmer_30b/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "targets/muse_glimmer_30b/impl/config.h"

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace ginfer::targets::muse_glimmer_30b::detail {

inline constexpr std::size_t kTextLayers   = TextConfig::layers;
inline constexpr std::size_t kDflashLayers = DFlash2Config::layers;

struct FrontendResourcePlan {
    artifact::ObjectHandle tokenizer_json;
    artifact::ObjectHandle tokenizer_config_json;
    artifact::ObjectHandle chat_template_jinja;
    artifact::ObjectHandle generation_config_json;
    artifact::ObjectHandle processor_config_json;
};

struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string processor_config_json;
};

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::Q4G64_F16S;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
};

struct AttentionPlan {
    WeightPlan query;
    WeightPlan key;
    WeightPlan value;
    WeightPlan gate;
    WeightPlan output;
};

struct MlpPlan {
    WeightPlan gate;
    WeightPlan up;
    WeightPlan down;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle pre_feedforward_norm;
    MlpPlan mlp;
    artifact::ObjectHandle post_feedforward_norm;
    bool full_attention = false;
};

struct DFlash2LayerPlan {
    artifact::ObjectHandle input_layernorm;
    artifact::ObjectHandle q_proj;
    artifact::ObjectHandle k_proj;
    artifact::ObjectHandle v_proj;
    artifact::ObjectHandle o_proj;
    artifact::ObjectHandle q_norm;
    artifact::ObjectHandle k_norm;
    artifact::ObjectHandle attention_conv_base;
    artifact::ObjectHandle attention_conv_proj;
    artifact::ObjectHandle post_attention_layernorm;
    artifact::ObjectHandle mlp_gate;
    artifact::ObjectHandle mlp_up;
    artifact::ObjectHandle mlp_down;
    artifact::ObjectHandle mlp_conv_base;
    artifact::ObjectHandle mlp_conv_proj;
};

struct DFlash2Plan {
    artifact::ObjectHandle fc;
    artifact::ObjectHandle hidden_norm;
    std::array<DFlash2LayerPlan, kDflashLayers> layers;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle selector_hidden;
    artifact::ObjectHandle predecessor_codebook;
    artifact::ObjectHandle successor_codebook;
};

struct BindingPlan {
    FrontendResourcePlan frontend;
    artifact::ObjectHandle token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle output_norm;
    artifact::ObjectHandle output_head;
    // Engaged only when startup selected DFlash.  Unselected DFlash artifact
    // objects are descriptor-validated but have no target load-plan ownership.
    std::optional<DFlash2Plan> dflash2;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

struct AttentionWeights {
    Weight query;
    Weight key;
    Weight value;
    Weight gate;
    Weight output;
};

struct MlpWeights {
    Weight gate;
    Weight up;
    Weight down;
};

struct TextLayerWeights {
    Tensor input_norm;
    AttentionWeights attention;
    Tensor post_attention_norm;
    Tensor pre_feedforward_norm;
    MlpWeights mlp;
    Tensor post_feedforward_norm;
    bool full_attention = false;
};

struct DFlash2LayerWeights {
    Tensor input_layernorm;
    Weight q_proj;
    Weight k_proj;
    Weight v_proj;
    Weight o_proj;
    Tensor q_norm;
    Tensor k_norm;
    Tensor attention_conv_base;
    Weight attention_conv_proj;
    Tensor post_attention_layernorm;
    Weight mlp_gate;
    Weight mlp_up;
    Weight mlp_down;
    Tensor mlp_conv_base;
    Weight mlp_conv_proj;
};

struct DFlash2Weights {
    Weight fc;
    Tensor hidden_norm;
    std::array<DFlash2LayerWeights, kDflashLayers> layers;
    Tensor norm;
    Weight selector_hidden;
    Weight predecessor_codebook;
    Weight successor_codebook;
};

struct RuntimeModel {
    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<TextLayerWeights, kTextLayers> text_layers;
    Tensor output_norm;
    Weight output_head;
    bool dflash2_loaded = false;
    DFlash2Weights dflash2;
};

class LoadedModelData {
public:
    LoadedModelData(WeightsProfile weights_profile, BindingPlan plan,
                    artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    FrontendResources frontend;
    RuntimeModel runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in),
          data(weights_profile_in, std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               bool enable_dflash2);

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources,
                                     const EngineOptions& options);

} // namespace ginfer::targets::muse_glimmer_30b::detail
