#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/context_copy.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "ginfer/ops/argmax.h"
#include "ginfer/ops/attn_input_proj.h"
#include "ginfer/ops/bidirectional_gqa_attention.h"
#include "ginfer/ops/dflash2_select.h"
#include "ginfer/ops/embedding.h"
#include "ginfer/ops/grouped_dynamic_conv.h"
#include "ginfer/ops/kv_cache_append_prefix.h"
#include "ginfer/ops/linear.h"
#include "ginfer/ops/linear_add.h"
#include "ginfer/ops/linear_pair.h"
#include "ginfer/ops/linear_swiglu.h"
#include "ginfer/ops/prepare_masked_block.h"
#include "ginfer/ops/prepare_ragged_prefix.h"
#include "ginfer/ops/rmsnorm.h"
#include "ginfer/ops/residual_add.h"
#include "ginfer/ops/rope.h"
#include "ginfer/ops/scatter.h"
#include "ginfer/ops/scalar.h"
#include "ginfer/ops/silu_mul.h"
#include "ginfer/ops/speculative_round.h"
#include "ginfer/ops/swa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ginfer::targets::qwen3_6::detail::GINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class Config, class Weights>
const Tensor& dflash_context_norm(const Weights& weights) {
    if constexpr (Config::sampled_dflash2) {
        return weights.hidden_norm;
    } else {
        return weights.context_norm;
    }
}

template <class Config, class LayerWeights>
void project_dflash_context_kv(const Tensor& context, const LayerWeights& weights,
                               Tensor& key, Tensor& value, cudaStream_t stream) {
    if constexpr (Config::sampled_dflash2) {
        ops::linear(context, weights.k_proj, key, stream);
        ops::linear(context, weights.v_proj, value, stream);
    } else {
        ops::linear_pair(context, weights.context_key, weights.context_value, key, value, stream);
    }
}

template <class Config, class LayerWeights>
const Tensor& dflash_key_norm(const LayerWeights& weights) {
    if constexpr (Config::sampled_dflash2) {
        return weights.k_norm;
    } else {
        return weights.key_norm;
    }
}

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({Config::feature_rows, columns}),
                    state.execution.model.dflash->feature_projection, projected,
                    state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected,
                     dflash_context_norm<Config>(*state.execution.model.dflash),
                     Config::rms_epsilon, false, context, state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            const bool local_layer  = layer < Config::local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            auto layer_roots =
                workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
            Tensor key_raw =
                layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor value =
                layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
            Tensor value_flat = value.view({Config::kv_size, layer_columns});
            project_dflash_context_kv<Config>(layer_context, weight, key_flat, value_flat,
                                              state.execution.device.stream);
            Tensor key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
            ops::rmsnorm(key_raw, dflash_key_norm<Config>(weight), Config::rms_epsilon, false, key,
                         state.execution.device.stream);
            ops::rope(layer_positions.view({layer_columns}), Config::rotary_dim,
                      Config::rope_theta, key, state.execution.device.occupancy(),
                      state.execution.device.stream);
            Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    state.execution.device.stream);
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(
                        static_cast<std::uint32_t>(layer - Config::local_layers)),
                    state.execution.device.stream);
            }
        }
    }
}

template <class Config, class Weights>
void propose_dflash2_batch(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                           const Weights& weights, std::int32_t batch_size, std::uint32_t k,
                           DFlashEnvelopes envelopes) {
    const std::int32_t drafts_count = static_cast<std::int32_t>(k);
    const std::int32_t width        = drafts_count + 1;
    const std::int32_t columns      = width * batch_size;
    const std::int32_t draft_columns = drafts_count * batch_size;
    Tensor anchors       = frame.anchors.slice(0, 0, batch_size);
    Tensor frontiers     = frame.execution_frontiers.slice(0, 0, batch_size);
    Tensor extents       = frame.proposal_extents.slice(0, 0, batch_size);
    Tensor valid_columns = frame.proposal_valid_columns.slice(0, 0, batch_size);
    Tensor lanes         = frame.lanes.slice(0, 0, batch_size);
    Tensor ids           = frame.proposal_ids.slice(1, 0, batch_size);
    Tensor positions     = frame.proposal_positions.slice(1, 0, batch_size);
    Tensor drafts        = frame.learned_draft_tokens.slice(1, 0, batch_size);
    Tensor candidates    = frame.learned_candidates.slice(2, 0, batch_size);
    Tensor probabilities = frame.learned_probabilities.slice(2, 0, batch_size);

    state.execution.work.reset();
    ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                              positions, state.execution.device.stream);
    Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
    ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                   state.execution.device.stream);

    Tensor flat_positions = positions.view({columns});
    for (int layer = 0; layer < Config::layers; ++layer) {
        auto layer_scope = state.execution.work.scope();
        const auto& weight = weights.layers.at(static_cast<std::size_t>(layer));
        auto roots = workspace_recipe::dflash2_layer<Config>(state.execution.work, columns);
        Tensor hidden_batch = roots.hidden.view({Config::hidden, width, batch_size});
        Tensor convolution_input =
            roots.convolution_input.view({Config::hidden, width, batch_size});
        Tensor convolution_output =
            roots.convolution_output.view({Config::hidden, width, batch_size});
        Tensor dynamic =
            roots.dynamic_kernel.view({Config::kernel_projection_rows, width, batch_size});

        ops::rmsnorm(residual, weight.input_layernorm, Config::rms_epsilon, false, roots.hidden,
                     state.execution.device.stream);
        ops::grouped_dynamic_conv_prepare(
            hidden_batch, weight.attention_conv_proj, weight.attention_conv_base, dynamic,
            convolution_input, Config::conv_group, state.execution.device.stream);
        ops::linear(roots.convolution_input, weight.q_proj, roots.query_raw,
                    state.execution.device.stream);
        ops::linear(roots.convolution_input, weight.k_proj, roots.key_raw,
                    state.execution.device.stream);
        ops::linear(roots.convolution_input, weight.v_proj, roots.value,
                    state.execution.device.stream);
        Tensor query_raw =
            roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
        Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
        Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
        Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
        ops::rmsnorm(query_raw, weight.q_norm, Config::rms_epsilon, false, query,
                     state.execution.device.stream);
        ops::rmsnorm(key_raw, weight.k_norm, Config::rms_epsilon, false, key,
                     state.execution.device.stream);
        ops::rope(flat_positions, Config::rotary_dim, Config::rope_theta, query, key,
                  state.execution.device.occupancy(), state.execution.device.stream);
        Tensor attention =
            roots.attention.view({Config::head_dim, Config::query_heads, width, batch_size});
        ops::swa(query.view({Config::head_dim, Config::query_heads, width, batch_size}),
                 key.view({Config::head_dim, Config::kv_heads, width, batch_size}),
                 roots.value.view({Config::head_dim, Config::kv_heads, width, batch_size}),
                 positions, valid_columns, lanes, Config::attention_scale,
                 dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                 envelopes.local, state.execution.work, attention,
                 state.execution.device.stream);
        ops::linear(roots.attention, weight.o_proj, roots.attention_output,
                    state.execution.device.stream);
        ops::grouped_dynamic_conv_finish(
            roots.attention_output.view({Config::hidden, width, batch_size}), dynamic,
            weight.attention_conv_base, convolution_output, Config::conv_group,
            state.execution.device.stream);
        ops::residual_add(roots.convolution_output, residual, state.execution.device.stream);

        ops::rmsnorm(residual, weight.post_attention_layernorm, Config::rms_epsilon, false,
                     roots.hidden, state.execution.device.stream);
        ops::grouped_dynamic_conv_prepare(hidden_batch, weight.mlp_conv_proj,
                                          weight.mlp_conv_base, dynamic, convolution_input,
                                          Config::conv_group, state.execution.device.stream);
        ops::linear(roots.convolution_input, weight.mlp_gate, roots.gate,
                    state.execution.device.stream);
        ops::linear(roots.convolution_input, weight.mlp_up, roots.up,
                    state.execution.device.stream);
        ops::silu_mul(roots.gate, roots.up, roots.intermediate,
                      state.execution.device.stream);
        ops::linear(roots.intermediate, weight.mlp_down, roots.down,
                    state.execution.device.stream);
        ops::grouped_dynamic_conv_finish(
            roots.down.view({Config::hidden, width, batch_size}), dynamic, weight.mlp_conv_base,
            convolution_output, Config::conv_group, state.execution.device.stream);
        ops::residual_add(roots.convolution_output, residual, state.execution.device.stream);
    }

    Tensor normalized = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
    ops::rmsnorm(residual, weights.final_norm, Config::rms_epsilon, false, normalized,
                 state.execution.device.stream);
    Tensor draft_hidden =
        state.execution.work.alloc(DType::BF16, {Config::hidden, drafts_count, batch_size});
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t draft_row_bytes =
        static_cast<std::size_t>(Config::hidden) * drafts_count * element_bytes;
    const std::size_t source_pitch =
        static_cast<std::size_t>(Config::hidden) * width * element_bytes;
    const auto* source = static_cast<const std::byte*>(normalized.data) +
                         static_cast<std::size_t>(Config::hidden) * element_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(draft_hidden.data, draft_row_bytes, source, source_pitch,
                                 draft_row_bytes, static_cast<std::size_t>(batch_size),
                                 cudaMemcpyDeviceToDevice, state.execution.device.stream));
    Tensor proposal_logits = state.execution.work.alloc(
        DType::BF16, {TextConfig::output_rows, drafts_count, batch_size});
    Tensor proposal_logits_flat =
        proposal_logits.view({TextConfig::output_rows, draft_columns});
    ops::linear(draft_hidden.view({Config::hidden, draft_columns}),
                state.execution.model.output_head, proposal_logits_flat,
                state.execution.device.stream);
    ops::dflash2_select_path(
        proposal_logits, draft_hidden, weights.selector_hidden, weights.predecessor_codebook,
        weights.successor_codebook, anchors, frame.sampling, frontiers, extents,
        TextConfig::token_domain, Config::selector_top_k, state.execution.work, drafts, candidates,
        probabilities, state.execution.device.stream);
    state.execution.work.reset();
}

void target_verify_accept_dflash2(ExecutionCore& execution, Tensor& continuation_hidden_store,
                                  TextContext& card, TargetVerifyFrameView frame,
                                  const Tensor& candidates, const Tensor& probabilities,
                                  ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr || frame.feature_sink == nullptr) {
        throw std::logic_error("sampled DFlash2 verify requires replay and feature storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                             frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                             frame.target_hidden, frame.target_logits, frame.target_tokens,
                             *frame.feature_sink);
    ops::dflash2_accept(
        frame.target_logits, frame.drafts, candidates, probabilities, frame.current_extents,
        frame.sampling, TextConfig::token_domain, frame.frontiers, frame.anchors,
        frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts, execution.work,
        execution.device.stream);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        if constexpr (Config::sampled_dflash2) {
            propose_dflash2_batch<Config>(state, frame, *state.execution.model.dflash, batch_size,
                                          k, envelopes);
        } else {
        const auto legacy = [&]<class Weights>(const Weights& dflash_weights) {
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes               = frame.lanes.slice(0, 0, batch_size);
        Tensor full_rows           = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            const auto& weight = dflash_weights.layers.at(static_cast<std::size_t>(layer));
            {
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.execution.device.stream);
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.occupancy(), state.execution.device.stream);
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                if (layer < Config::local_layers) {
                    ops::swa(query_batch, key_batch, value_batch, positions, valid_columns, lanes,
                             Config::attention_scale,
                             dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, state.execution.work, attention_batch,
                             state.execution.device.stream);
                } else {
                    ops::bidirectional_gqa_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        Config::attention_scale, dflash_state(state).full_batch_layer(0),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.occupancy(), state.execution.device.stream);
                }
                ops::linear_add(roots.attention.view({Config::query_size, columns}),
                                weight.attention_output, residual, state.execution.work,
                                state.execution.device.stream);
            }
            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                ops::linear_swiglu(roots.hidden, weight.gate_up, roots.intermediate,
                                   state.execution.work, state.execution.device.stream);
                ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                state.execution.device.stream);
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, dflash_weights.final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            ops::argmax(logits, flat_drafts, TextConfig::token_domain,
                        state.execution.device.stream);
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.stream);
            ops::argmax(logits, flat_drafts, V::draft_head_rows, state.execution.device.stream);
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.execution.device.stream);
        }
        state.execution.work.reset();
        };
        legacy(*state.execution.model.dflash);
        }
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size,
                              std::uint32_t proposal_k, std::uint32_t target_k,
                              DFlashEnvelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, proposal_k, target_k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            proposal_k == 0 || proposal_k > target_k ||
            target_k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t target_drafts   = static_cast<std::int32_t>(target_k);
        const std::int32_t width           = target_drafts + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor target_extents   = frame.target_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows      = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.lanes.slice(0, 0, batch_size);
        Tensor append_positions(frame.append_positions.data, DType::I32, {width, batch_size});
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts(frame.draft_tokens.data, DType::I32, {target_drafts, batch_size});
        Tensor verify_ids(frame.verify_ids.data, DType::I32, {width, batch_size});
        Tensor target_positions(frame.target_positions.data, DType::I32, {width, batch_size});
        Tensor target_tokens(frame.target_argmax.data, DType::I32, {width, batch_size});
        Tensor target_logits(frame.target_logits.data, DType::BF16,
                             {TextConfig::output_rows, width, batch_size});
        Tensor target_hidden(frame.target_hidden.data, DType::BF16,
                             {TextConfig::hidden, width, batch_size});
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens(frame.licensed_tokens.data, DType::I32, {width, batch_size});
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor candidates;
        Tensor probabilities;
        if constexpr (Variant::DFlashConfig::sampled_dflash2) {
            candidates = Tensor(frame.proposal_candidates.data, DType::I32,
                                {Variant::DFlashConfig::selector_top_k, target_drafts,
                                 batch_size});
            probabilities = Tensor(frame.proposal_probabilities.data, DType::FP32,
                                   {Variant::DFlashConfig::selector_top_k, target_drafts,
                                    batch_size});
        }

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        Tensor pending_features = dflash_state(state).pending_features.slice(1, 0, width);
        ops::prepare_ragged_prefix(pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     lanes, dflash_rows, envelopes.append);

        propose_batch_impl<Variant>(state, frame, batch_size, proposal_k, envelopes);
        if constexpr (Variant::DFlashConfig::sampled_dflash2) {
            Tensor learned_drafts = frame.learned_draft_tokens.slice(1, 0, batch_size);
            Tensor learned_candidates = frame.learned_candidates.slice(2, 0, batch_size);
            Tensor learned_probabilities = frame.learned_probabilities.slice(2, 0, batch_size);
            Tensor context_tokens(frame.context_copy_tokens.data, DType::I32,
                                  {target_drafts, batch_size});
            Tensor match_lengths = frame.context_match_lengths.slice(0, 0, batch_size);
            Tensor context_extents = frame.context_copy_extents.slice(0, 0, batch_size);
            ops::dflash2_fuse_context_copy(
                learned_drafts, learned_candidates, learned_probabilities, context_tokens,
                match_lengths, context_extents,
                static_cast<std::int32_t>(qwen3_6::detail::kContextCopyMinimumMatch), drafts,
                candidates, probabilities, state.execution.device.stream);
        }
        ops::prepare_masked_block(anchors, frontiers, valid_columns,
                                  Variant::DFlashConfig::mask_token, verify_ids,
                                  target_positions, state.execution.device.stream);
        ops::speculative_prepare_verify_ids(anchors, drafts, target_extents, verify_ids,
                                            state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, lanes, valid_columns, width, batch_size);
        TargetVerifyFrameView verify_frame{
            .ids             = verify_ids,
            .cache_positions = target_positions,
            .rope_positions  = target_positions,
            .valid_columns   = valid_columns,
            .kv_table_rows   = text_rows,
            .lanes           = lanes,
            .target_hidden   = target_hidden,
            .target_logits   = target_logits,
            .target_tokens   = target_tokens,
            .drafts          = drafts,
            .current_extents = target_extents,
            .frontiers       = frontiers,
            .anchors         = anchors,
            .licensed_tokens = licensed_tokens,
            .licensed_counts = licensed_counts,
            .accepted_drafts = accepted,
            .selected_hidden = selected_hidden,
            .replay_records  = state.execution.replay_records,
            .sampling        = frame.sampling,
            .feature_sink    = &sink,
        };
        if constexpr (Variant::DFlashConfig::sampled_dflash2) {
            target_verify_accept_dflash2(state.execution, state.continuation_hidden_store, card,
                                          verify_frame, candidates, probabilities,
                                          target_envelope);
        } else {
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_frame, target_envelope);
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t proposal_k, std::uint32_t target_k,
                                 DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, proposal_k, target_k, envelopes,
                                         target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                         std::uint32_t proposal_k, std::uint32_t target_k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, proposal_k, target_k, envelopes,
                                         target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ginfer::targets::qwen3_6::detail::GINFER_QWEN36_RUNTIME_NS::schedule
