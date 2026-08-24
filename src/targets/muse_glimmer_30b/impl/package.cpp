#include <ginfer/targets/muse_glimmer_30b/package.h>

#include "artifact/reader.h"
#include "targets/muse_glimmer_30b/impl/config.h"
#include "targets/muse_glimmer_30b/impl/load/bindings.h"
#include "targets/muse_glimmer_30b/impl/runtime/program.h"
#include "targets/muse_glimmer_30b/impl/runtime/sequence.h"

#include <stdexcept>
#include <utility>

namespace ginfer::targets::muse_glimmer_30b::detail {

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;

} // namespace ginfer::targets::muse_glimmer_30b::detail

namespace ginfer::targets::muse_glimmer_30b {
namespace {

constexpr ModelSamplingDefaults kMuseDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 64,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 1.0F,
                     .top_k             = 64,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kMuseDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int-dflash-q4") {
        return WeightsProfile::GroupwiseIntDFlashQ4;
    }
    if (identity.model_id == model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Nvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

void Package::resolve_startup_options(WeightsProfile, EngineOptions& options) {
    if (options.speculative.backend != SpeculativeBackend::Automatic) { return; }
    options.speculative.backend       = SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 4;
    options.speculative.proposal_head = ProposalHead::Full;
}

GpuRequirements Package::gpu_requirements(WeightsProfile weights_profile,
                                          const EngineOptions&) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntDFlashQ4:
        return {.sm_86 = true, .sm_89 = true, .sm_120 = true, .nvfp4 = false};
    case WeightsProfile::Nvfp4:
        return {.sm_86 = false, .sm_89 = false, .sm_120 = true, .nvfp4 = true};
    }
    throw std::logic_error("invalid Muse Glimmer 30B weight profile");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    if (options.enable_vision) {
        throw std::invalid_argument(
            "muse-glimmer-30b v1 is text-only; disable Vision for this artifact");
    }
    if (options.speculative.backend == SpeculativeBackend::Mtp) {
        throw std::invalid_argument("muse-glimmer-30b does not implement MTP");
    }
    if (options.speculative.backend == SpeculativeBackend::DFlash) {
        if (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 15) {
            throw std::invalid_argument(
                "muse-glimmer-30b --spec dflash requires --draft-tokens in [1,15]");
        }
    }
    if (options.max_context == 0 || options.max_context > detail::kNativeContext) {
        throw std::invalid_argument("muse-glimmer-30b max_context must be in (0, 131072]");
    }
    const bool enable_dflash2 = options.speculative.backend == SpeculativeBackend::DFlash;
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, detail::bind_artifact(binder, weights_profile, enable_dflash2)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (options.enable_vision) {
        throw std::invalid_argument(
            "muse-glimmer-30b v1 is text-only; disable Vision for this artifact");
    }
    return detail::make_frontend(model.impl_->data.frontend, options);
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return detail::make_sequence_planner(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (model.impl_->data.runtime.dflash2_loaded != plan.layout().dflash_enabled) {
        throw std::invalid_argument(
            "muse-glimmer-30b load plan and sequence plan disagree on DFlash selection");
    }
    return std::make_unique<Program>(model.impl_->data.runtime, std::move(plan), device);
}

} // namespace ginfer::targets::muse_glimmer_30b
