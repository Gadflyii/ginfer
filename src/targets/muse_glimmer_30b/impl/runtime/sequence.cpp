#include "targets/muse_glimmer_30b/impl/runtime/sequence.h"

#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ginfer/ops/dflash2_select.h"
#include "ginfer/ops/gqa_attention.h"
#include "ginfer/ops/sampling.h"
#include "ginfer/ops/speculative_round.h"
#include "ginfer/ops/swa.h"
#include "targets/muse_glimmer_30b/impl/config.h"
#include "targets/muse_glimmer_30b/impl/runtime/program_storage.h"
#include "targets/muse_glimmer_30b/impl/runtime/workspace_recipe.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ginfer::targets::muse_glimmer_30b::detail {
namespace {

constexpr std::size_t kCudaGraphAllowancePerExactBatch = 12ULL << 20;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::uint32_t page_count(std::uint32_t tokens) {
    if (tokens == 0) { throw std::invalid_argument("muse-glimmer-30b capacity must be positive"); }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

void add_scratch(WorkspaceLayoutBuilder& layout, std::size_t bytes) {
    auto scope = layout.scope();
    (void)layout.alloc_bytes(bytes);
}

void plan_text_layers(WorkspaceLayoutBuilder& layout, const SequenceLayout& sequence,
                      std::int32_t width, std::int32_t batch,
                      const OccupancyPolicy& occupancy) {
    auto scope = layout.scope();
    (void)allocate_text_layers_workspace(
        layout, width, batch, sequence.weights_profile == WeightsProfile::Nvfp4);
    const DType cache_dtype = sequence.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16
                                                                            : DType::I8;
    add_scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, cache_dtype, {1, sequence.capacity}, batch, 1,
                            width, occupancy));
}

void plan_sampling(WorkspaceLayoutBuilder& layout, std::int32_t batch) {
    auto scope = layout.scope();
    (void)layout.alloc(DType::BF16, {TextConfig::vocab, batch});
    (void)layout.alloc(DType::BF16, {TextConfig::vocab, batch});
    add_scratch(layout, ops::sampling_workspace_capacity_bytes(TextConfig::vocab, 1, batch));
}

void plan_dflash_projection(WorkspaceLayoutBuilder& layout, std::int32_t width,
                            std::int32_t batch = 1) {
    auto scope = layout.scope();
    (void)allocate_dflash_projection_workspace(layout, width, batch);
}

void plan_dflash_layers(WorkspaceLayoutBuilder& layout, const SequenceLayout& sequence,
                        std::int32_t width, std::int32_t batch = 1) {
    auto scope = layout.scope();
    (void)allocate_dflash_layers_workspace(layout, width, batch);
    add_scratch(layout,
                ops::swa_workspace_capacity_bytes({0, sequence.capacity}, 1, width, batch,
                                                  DFlash2Config::cyclic_capacity));
}

std::size_t plan_prefill_workspace(const SequenceLayout& sequence,
                                   const OccupancyPolicy& occupancy) {
    const auto tokens = static_cast<std::int32_t>(sequence.prefill_chunk);
    WorkspaceLayoutBuilder layout;
    auto phase = layout.scope();
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, tokens});
    (void)layout.alloc(DType::I32, {tokens, 1});
    if (sequence.dflash_enabled) {
        (void)layout.alloc(DType::BF16, {DFlash2Config::captured_rows, tokens});
    }
    {
        auto embed = layout.scope();
        (void)layout.alloc(DType::I32, {tokens});
        (void)layout.alloc(DType::BF16, {TextConfig::hidden, tokens});
    }
    plan_text_layers(layout, sequence, tokens, 1, occupancy);
    if (sequence.dflash_enabled) {
        auto append = layout.scope();
        (void)layout.alloc(DType::I32, {1});
        (void)layout.alloc(DType::I32, {1});
        plan_dflash_projection(layout, tokens);
    }
    plan_sampling(layout, 1);
    return layout.peak_bytes(256);
}

std::size_t plan_dflash_round_workspace(const SequenceLayout& sequence,
                                        const OccupancyPolicy& occupancy) {
    const auto drafts = static_cast<std::int32_t>(sequence.dflash_draft_tokens);
    const auto width  = drafts + 1;
    const auto batch  = static_cast<std::int32_t>(sequence.max_concurrency);
    const auto columns = width * batch;
    WorkspaceLayoutBuilder layout;
    auto phase = layout.scope();
    (void)layout.alloc(DType::I32, {batch}); // anchors
    (void)layout.alloc(DType::I32, {batch}); // lengths
    (void)layout.alloc(DType::I32, {batch}); // valid columns
    (void)layout.alloc(DType::I32, {batch}); // draft extents
    (void)layout.alloc(DType::I32, {batch}); // cyclic lanes
    (void)layout.alloc(DType::I32, {width, batch});
    (void)layout.alloc(DType::I32, {width, batch});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, width, batch});
    plan_dflash_layers(layout, sequence, width, batch);

    (void)layout.alloc(DType::BF16, {TextConfig::hidden, drafts, batch});
    (void)layout.alloc(DType::BF16, {TextConfig::vocab, drafts, batch});
    (void)layout.alloc(DType::BF16, {TextConfig::vocab, drafts, batch});
    (void)layout.alloc(DType::I32, {drafts, batch});
    (void)layout.alloc(DType::I32, {DFlash2Config::selector_top_k, drafts, batch});
    (void)layout.alloc(DType::FP32, {DFlash2Config::selector_top_k, drafts, batch});
    add_scratch(layout, ops::dflash2_select_path_workspace_capacity_bytes(
                            TextConfig::vocab, DFlash2Config::selector_rank,
                            DFlash2Config::selector_top_k, drafts, drafts, batch, batch));
    (void)layout.alloc(DType::I32, {width, batch});
    (void)layout.alloc(DType::I32, {width, batch});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, columns});
    (void)layout.alloc(DType::BF16, {DFlash2Config::captured_rows, width, batch});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, columns});
    plan_text_layers(layout, sequence, width, batch, occupancy);
    (void)layout.alloc(DType::BF16, {TextConfig::vocab, width, batch});
    (void)layout.alloc(DType::BF16, {TextConfig::vocab, width, batch});
    (void)layout.alloc(DType::I32, {width, batch});
    (void)layout.alloc(DType::I32, {batch});
    (void)layout.alloc(DType::I32, {batch});
    add_scratch(layout, ops::dflash2_accept_workspace_capacity_bytes(
                            TextConfig::vocab, drafts, drafts, 1, batch));
    plan_dflash_projection(layout, width, batch);
    return layout.peak_bytes(256);
}

std::size_t plan_workspace(const SequenceLayout& sequence, const OccupancyPolicy& occupancy) {
    std::size_t capacity = plan_prefill_workspace(sequence, occupancy);
    {
        WorkspaceLayoutBuilder ordinary;
        plan_text_layers(ordinary, sequence, 1,
                         static_cast<std::int32_t>(sequence.max_concurrency), occupancy);
        plan_sampling(ordinary, static_cast<std::int32_t>(sequence.max_concurrency));
        capacity = std::max(capacity, ordinary.peak_bytes(256));
    }
    if (sequence.dflash_enabled) {
        capacity = std::max(capacity, plan_dflash_round_workspace(sequence, occupancy));
    }
    return capacity;
}

SequenceLayout build_layout(const EngineOptions& options, WeightsProfile weights_profile,
                            std::uint32_t main_page_groups, const OccupancyPolicy& occupancy) {
    SequenceLayout layout;
    layout.weights_profile      = weights_profile;
    layout.capacity             = options.max_context;
    layout.max_concurrency      = options.max_concurrency;
    layout.main_page_groups     = main_page_groups;
    layout.kv_capacity          = main_page_groups * static_cast<std::uint32_t>(kPagedKVPageSize);
    layout.kv_cache             = options.kv_cache;
    layout.max_main_page_groups = options.max_concurrency * page_count(options.max_context);
    layout.prefill_chunk        = std::min(options.prefill_chunk, options.max_context);
    layout.dflash_enabled       = options.speculative.backend == SpeculativeBackend::DFlash;
    layout.dflash_draft_tokens  = options.speculative.draft_tokens;
    if (layout.dflash_enabled) {
        layout.prefill_chunk = std::min(
            layout.prefill_chunk, static_cast<std::uint32_t>(DFlash2Config::cyclic_capacity));
    }

    LayoutBuilder kv_builder;
    const PagedKVPoolLayout kv_layout = plan_paged_kv_pool(kv_builder, make_text_kv_spec(layout));
    layout.full_kv_bytes              = kv_builder.finish(256, "muse kv");
    layout.sliding_kv_bytes           = 0;
    if (layout.dflash_enabled) {
        LayoutBuilder cyclic_builder;
        (void)plan_cyclic_kv_cache(cyclic_builder, static_cast<std::uint32_t>(DFlash2Config::layers),
                                   static_cast<std::uint32_t>(DFlash2Config::cyclic_capacity),
                                   DFlash2Config::kv_heads, TextConfig::head_dim,
                                   static_cast<std::int32_t>(options.max_concurrency));
        layout.sliding_kv_bytes = cyclic_builder.finish(256, "muse dflash2 cyclic");
    }
    const ProgramStorageLayout program_storage =
        plan_program_storage(layout.max_concurrency);
    layout.program_storage_bytes = program_storage.bytes;
    layout.persistent_bytes = checked_add(
        checked_add(layout.full_kv_bytes, layout.sliding_kv_bytes, "Muse persistent KV"),
        layout.program_storage_bytes, "Muse Program storage");
    layout.workspace_bytes         = plan_workspace(layout, occupancy);
    layout.request_transient_bytes = 0;
    layout.graph_allowance_bytes =
        options.use_cuda_graph && !layout.dflash_enabled
            ? checked_mul(kCudaGraphAllowancePerExactBatch, layout.max_concurrency,
                          "Muse exact-batch CUDA Graph allowance")
            : 0;
    (void)kv_layout;

    layout.device_reservation_bytes = layout.persistent_bytes;
    layout.device_reservation_bytes =
        checked_add(layout.device_reservation_bytes, layout.workspace_bytes, "workspace");
    layout.device_reservation_bytes = checked_add(layout.device_reservation_bytes,
                                                  layout.request_transient_bytes, "transient");
    layout.device_reservation_bytes =
        checked_add(layout.device_reservation_bytes, layout.graph_allowance_bytes, "graphs");
    if (layout.device_reservation_bytes == 0) {
        throw std::logic_error("muse-glimmer-30b sequence reservation is empty");
    }
    return layout;
}

} // namespace

PagedKVPoolSpec make_text_kv_spec(const SequenceLayout& layout) {
    PagedKVPoolSpec spec;
    spec.page_group_count      = layout.main_page_groups;
    spec.logical_page_capacity = page_count(layout.capacity);
    spec.table_rows            = static_cast<std::int32_t>(layout.max_concurrency);
    spec.plane_order           = PagedKVPlaneOrder::PageMajor;
    const bool quantized       = layout.kv_cache == KvCacheStorage::Int8Group64;
    const DType code_dtype     = quantized ? DType::I8 : DType::BF16;
    spec.planes.reserve(static_cast<std::size_t>(TextConfig::layers) * (quantized ? 4ULL : 2ULL));
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        spec.planes.push_back({code_dtype, TextConfig::head_dim, TextConfig::kv_heads, 256});
        spec.planes.push_back({code_dtype, TextConfig::head_dim, TextConfig::kv_heads, 256});
        if (quantized) {
            spec.planes.push_back(
                {DType::FP16, TextConfig::head_dim / static_cast<int>(kKvQuantGroup),
                 TextConfig::kv_heads, 256});
            spec.planes.push_back(
                {DType::FP16, TextConfig::head_dim / static_cast<int>(kKvQuantGroup),
                 TextConfig::kv_heads, 256});
        }
    }
    return spec;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    const std::size_t expected = curve_.reservation_bytes(main_page_groups);
    SequenceLayout layout      = main_page_groups == minimum_.main_page_groups
                                     ? minimum_
                                     : build_layout(options_, weights_profile_, main_page_groups,
                                                    occupancy_);
    if (layout.device_reservation_bytes != expected) {
        throw std::logic_error(
            "muse-glimmer-30b sequence layout is not affine in Main KV page capacity");
    }
    return SequencePlan(layout);
}

SequencePlanner make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                                      WeightsProfile weights_profile) {
    if (options.max_context == 0 || options.max_context > kNativeContext) {
        throw std::invalid_argument("muse-glimmer-30b max_context must be in (0, 131072]");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("muse-glimmer-30b max_concurrency must be in [1,8]");
    }
    if (options.enable_vision) {
        throw std::invalid_argument(
            "muse-glimmer-30b v1 is text-only; disable Vision for this artifact");
    }

    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    SequenceLayout minimum = build_layout(options, weights_profile, minimum_pages,
                                          device.occupancy());
    runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
        .minimum_main_page_groups             = minimum_pages,
        .maximum_main_page_groups             = maximum_pages,
        .minimum_device_reservation_bytes     = minimum.device_reservation_bytes,
        .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        SequenceLayout adjacent = build_layout(options, weights_profile, minimum_pages + 1U,
                                               device.occupancy());
        if (adjacent.device_reservation_bytes <= minimum.device_reservation_bytes) {
            throw std::logic_error("muse-glimmer-30b sequence layout has a nonpositive KV stride");
        }
        curve.bytes_per_additional_main_page_group =
            adjacent.device_reservation_bytes - minimum.device_reservation_bytes;
    }
    return SequencePlanner(options, weights_profile, std::move(minimum), curve,
                           device.occupancy());
}

} // namespace ginfer::targets::muse_glimmer_30b::detail
