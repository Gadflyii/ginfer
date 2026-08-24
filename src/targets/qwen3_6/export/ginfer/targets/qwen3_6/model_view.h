#pragma once

#include <ginfer/targets/qwen3_6/startup_features.h>
#include <ginfer/targets/qwen3_6/vision.h>

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <optional>

namespace ginfer {

class DeviceArena;

namespace targets::qwen3_6 {

template <class ProjectionPayload, class PostMixerPayload>
struct FullAttentionWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class ProjectionPayload, class PostMixerPayload>
struct GdnWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class AttentionPayload, class PostMixerPayload>
struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    AttentionPayload attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
    Tensor final_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

struct DFlashLayerWeights {
    Tensor input_norm;
    Weight query_key_value;
    Weight context_key;
    Weight context_value;
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    Tensor post_attention_norm;
    Weight gate_up;
    Weight down;
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

template <std::size_t Layers>
struct DFlashWeights {
    Weight feature_projection;
    Tensor context_norm;
    std::array<DFlashLayerWeights, Layers> layers;
    Tensor final_norm;
};

template <std::size_t Layers>
struct DFlash2Weights {
    Weight feature_projection;
    Tensor hidden_norm;
    std::array<DFlash2LayerWeights, Layers> layers;
    Tensor final_norm;
    Weight selector_hidden;
    Weight predecessor_codebook;
    Weight successor_codebook;
};

template <class FullProjectionPayload, class GdnProjectionPayload, class MainPostMixerPayload,
          class MtpAttentionPayload, class MtpPostMixerPayload, class DFlashPayload,
          std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct ModelView {
    using FullLayer = FullAttentionWeights<FullProjectionPayload, MainPostMixerPayload>;
    using GdnLayer  = GdnWeights<GdnProjectionPayload, MainPostMixerPayload>;
    using MtpLayer  = MtpWeights<MtpAttentionPayload, MtpPostMixerPayload>;
    using DFlash    = DFlashPayload;

    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<FullLayer, FullAttentionLayers> full_layers;
    std::array<GdnLayer, GdnLayers> gdn_layers;
    Tensor final_norm;
    Weight output_head;
    StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpLayer> mtp;
    std::optional<DFlashPayload> dflash;
    std::optional<VisionWeights> vision;
};

} // namespace targets::qwen3_6
} // namespace ginfer
