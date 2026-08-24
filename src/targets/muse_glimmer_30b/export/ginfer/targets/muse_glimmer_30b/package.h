#pragma once

#include "ginfer/types.h"
#include "core/gpu_capability.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ginfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::muse_glimmer_30b {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t { GroupwiseInt, GroupwiseIntDFlashQ4, Nvfp4 };

struct FrontendResources;
struct RuntimeModel;
class Program;

class PublishedOutput {
public:
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }
    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }
    [[nodiscard]] iterator end() noexcept { return values_.begin() + size_; }
    [[nodiscard]] const_iterator end() const noexcept { return values_.begin() + size_; }
    [[nodiscard]] OutputDelta& back() noexcept { return values_[size_ - 1]; }
    [[nodiscard]] const OutputDelta& back() const noexcept { return values_[size_ - 1]; }
    void clear() noexcept;
    void push_back(OutputDelta value);

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    PromptPreparationStats prepare;
    bool parse_atem_output = false;
    bool prefix_identity_reusable = true;
};

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;
    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;
    friend class Frontend;
    friend class Program;
};

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;
    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview(std::span<const TokenId> tokens,
                                                  std::uint32_t budget_remaining,
                                                  FinishReason limit_reason);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview() noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;

private:
    class Impl;
    explicit OutputSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend class Frontend;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const noexcept;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
                                                    const StopPolicy& caller_stop,
                                                    const OutputOptions& output = {}) const;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;
    friend struct muse_glimmer_30b::Package;
    friend Frontend make_frontend(const FrontendResources& resources,
                                  const EngineOptions& options);
};

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct muse_glimmer_30b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct muse_glimmer_30b::Package;
};

struct SequenceLayout {
    WeightsProfile weights_profile              = WeightsProfile::GroupwiseInt;
    std::uint32_t capacity                      = 0;
    std::uint32_t kv_capacity                   = 0;
    std::uint32_t max_concurrency               = 1;
    std::uint32_t main_page_groups              = 0;
    std::uint32_t max_main_page_groups          = 0;
    KvCacheStorage kv_cache                     = KvCacheStorage::Int8Group64;
    std::size_t full_kv_bytes                   = 0;
    std::size_t sliding_kv_bytes                = 0;
    std::size_t program_storage_bytes           = 0;
    std::size_t persistent_bytes                = 0;
    std::size_t workspace_bytes                 = 0;
    std::size_t request_transient_bytes         = 0;
    std::size_t graph_allowance_bytes           = 0;
    std::size_t device_reservation_bytes        = 0;
    std::size_t bytes_per_additional_page_group = 0;
    std::uint32_t prefill_chunk                 = 1024;
    bool dflash_enabled                         = false;
    std::uint32_t dflash_draft_tokens            = 0;
};

class SequencePlan {
public:
    SequencePlan() = default;
    explicit SequencePlan(SequenceLayout layout) : layout_(layout) {}

    [[nodiscard]] std::uint32_t capacity() const noexcept { return layout_.capacity; }
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept { return layout_.kv_capacity; }
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept { return layout_.max_concurrency; }
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept {
        return layout_.device_reservation_bytes;
    }
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept {
        return layout_.workspace_bytes;
    }
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept {
        return layout_.request_transient_bytes;
    }
    [[nodiscard]] const SequenceLayout& layout() const noexcept { return layout_; }

private:
    SequenceLayout layout_{};
};

class SequencePlanner {
public:
    SequencePlanner(EngineOptions options, WeightsProfile weights_profile, SequenceLayout minimum,
                    runtime::SequenceCapacityCurve curve, OccupancyPolicy occupancy)
        : options_(std::move(options)), weights_profile_(weights_profile),
          minimum_(std::move(minimum)), curve_(curve), occupancy_(occupancy) {}

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept {
        return curve_;
    }
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

private:
    EngineOptions options_{};
    WeightsProfile weights_profile_ = WeightsProfile::GroupwiseInt;
    SequenceLayout minimum_{};
    runtime::SequenceCapacityCurve curve_{};
    OccupancyPolicy occupancy_{};
};

class RequestBasePlan {
public:
    runtime::RequestPlanSummary summary_value;
    ResolvedSamplingParameters sampling{};
    bool allow_prefix_reuse = true;
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return summary_value;
    }
};

class RequestPlan {
public:
    runtime::RequestPlanSummary summary_value;
    ResolvedSamplingParameters sampling{};
    std::uint32_t reuse_base = 0;
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return summary_value;
    }
};

class Program {
public:
    Program(const RuntimeModel& model, SequencePlan plan, DeviceContext& device);
    ~Program();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    [[nodiscard]] RequestBasePlan plan_request_base(const PreparedPrompt& prompt,
                                                    const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPrompt& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "muse-glimmer-30b";
    static constexpr std::string_view target_key = "muse_glimmer_30b";

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = detail::Frontend;
    using PreparedPrompt  = detail::PreparedPrompt;
    using OutputSession   = detail::OutputSession;
    using SequencePlanner = detail::SequencePlanner;
    using SequencePlan    = detail::SequencePlan;
    using RequestBasePlan = detail::RequestBasePlan;
    using RequestPlan     = detail::RequestPlan;
    using Program         = detail::Program;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    static void resolve_startup_options(WeightsProfile weights_profile, EngineOptions& options);
    [[nodiscard]] static GpuRequirements gpu_requirements(WeightsProfile weights_profile,
                                                          const EngineOptions& options);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::muse_glimmer_30b
} // namespace ginfer
