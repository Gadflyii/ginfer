#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include "ginfer/ops/rmsnorm.h"
#include "ops/occupancy.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ginfer::ops::detail {
namespace {

inline constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();
inline constexpr std::int32_t kLastCooperativeCols = 4096;

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Bf16GdnGatingScheduleId schedule;
};

constexpr std::array<RouteSpec, 6> k27Routes{{
    {{1, 1}, Bf16GdnGatingScheduleId::GemvPairedRows},
    {{2, 8}, Bf16GdnGatingScheduleId::SmallTSplit10},
    // As token tiles double, halve SplitK so the catalog stays near 192 CTAs on 170 SMs.
    // Resolve falls back when the exact kernel grid exceeds resident capacity on this SM image.
    {{9, 1024}, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {{1025, 2048}, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {{2049, 4096}, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {{4097, kAnyCols}, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

constexpr std::array<RouteSpec, 5> k35Routes{{
    // The same progression keeps the long-range cooperative routes near 256 CTAs.
    {{1, 127}, Bf16GdnGatingScheduleId::MmaCooperativeSplit16},
    {{128, 1024}, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {{1025, 2048}, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {{2049, 4096}, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {{4097, kAnyCols}, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes,
                                 std::int32_t last) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == last && expected == static_cast<std::int64_t>(last) + 1;
}

static_assert(catalog_is_closed(k27Routes, kAnyCols));
static_assert(catalog_is_closed(k35Routes, kAnyCols));

bool is_27(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.heads == 48 && problem.input_rows == 5120;
}

bool is_35(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.heads == 32 && problem.input_rows == 2048;
}

bool schedule_uses_mma(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return true;
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SmallTSplit10:
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return false;
    }
    return false;
}

std::int32_t mma_tile_cols(const Bf16GdnGatingProblem& problem) noexcept {
    return is_35(problem) ? 64 : 128;
}

std::int32_t schedule_split_k(Bf16GdnGatingScheduleId schedule) {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return 10;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        return 32;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        return 16;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return 8;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return 4;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return 2;
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return 1;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

bool cooperative_grid_is_resident(Bf16GdnGatingScheduleId schedule, std::int32_t cols,
                                  std::int32_t tile_cols, std::int32_t row_tiles,
                                  std::int32_t resident_ctas) noexcept {
    const std::int64_t column_tiles = (static_cast<std::int64_t>(cols) + tile_cols - 1) / tile_cols;
    const std::int64_t grid_ctas =
        column_tiles * row_tiles * static_cast<std::int64_t>(schedule_split_k(schedule));
    return grid_ctas <= resident_ctas;
}

std::int32_t cooperative_blocks_per_sm(const Bf16GdnGatingProblem& problem,
                                       Bf16GdnGatingScheduleId schedule,
                                       SmArch sm_arch) noexcept {
    // These are kernel-image facts, not generic device occupancy multipliers. CUDA 13.1 ptxas
    // emits the SM8x 27B Split8 kernel at 256 threads/65 registers and Split4/2 at 512
    // threads/74 registers; all use 40 KiB dynamic shared memory. The 35B kernels use 256
    // threads and 24 KiB: Split32 is register-limited to two CTAs, Split16 admits four, and
    // Split8/4/2 use 74 registers on SM8x (three CTAs) versus 62 on SM120 (four CTAs).
    if (is_27(problem)) {
        switch (sm_arch) {
        case SmArch::Ampere86:
        case SmArch::Ada89:
            return schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit8 ? 2 : 1;
        case SmArch::Blackwell120:
            return 2;
        }
        return 0;
    }

    if (schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit32) { return 2; }
    switch (sm_arch) {
    case SmArch::Ampere86:
    case SmArch::Ada89:
        return schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit16 ? 4 : 3;
    case SmArch::Blackwell120:
        return 4;
    }
    return 0;
}

bool cooperative_27_grid_is_resident(Bf16GdnGatingScheduleId schedule, std::int32_t cols,
                                     const OccupancyPolicy& occupancy) noexcept {
    const std::int32_t resident_ctas =
        occupancy.sm_count * cooperative_blocks_per_sm({48, 5120, cols}, schedule,
                                                       occupancy.sm_arch);
    return cooperative_grid_is_resident(schedule, cols, 128, 3, resident_ctas);
}

bool cooperative_35_grid_is_resident(Bf16GdnGatingScheduleId schedule, std::int32_t cols,
                                     const OccupancyPolicy& occupancy) noexcept {
    const std::int32_t resident_ctas =
        occupancy.sm_count * cooperative_blocks_per_sm({32, 2048, cols}, schedule,
                                                       occupancy.sm_arch);
    return cooperative_grid_is_resident(schedule, cols, 64, 2, resident_ctas);
}

bool candidate_is_legal(Bf16GdnGatingScheduleId schedule, const Bf16GdnGatingProblem& problem,
                        const OccupancyPolicy& occupancy) noexcept {
    if (!bf16_gdn_gating_admits(problem)) { return false; }
    if (is_27(problem)) {
        switch (schedule) {
        case Bf16GdnGatingScheduleId::GemvPairedRows:
            return problem.cols == 1;
        case Bf16GdnGatingScheduleId::SmallTSplit10:
            return problem.cols >= 2 && problem.cols <= 8;
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
            return cooperative_27_grid_is_resident(schedule, problem.cols, occupancy);
        case Bf16GdnGatingScheduleId::MmaUnsplit:
            return true;
        case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
            return false;
        }
    }

    switch (schedule) {
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        return problem.cols <= 4 * 65'535;
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return problem.cols <= 8 * 65'535;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return true;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return cooperative_35_grid_is_resident(schedule, problem.cols, occupancy);
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return false;
    }
    return false;
}

std::size_t checked_partial_bytes(std::int32_t heads, std::int32_t split_k, std::int32_t cols) {
    const std::size_t logical_rows = static_cast<std::size_t>(2 * heads);
    const std::size_t split        = static_cast<std::size_t>(split_k);
    const std::size_t tokens       = static_cast<std::size_t>(cols);
    if (tokens > std::numeric_limits<std::size_t>::max() / logical_rows ||
        split > std::numeric_limits<std::size_t>::max() / (tokens * logical_rows)) {
        throw std::overflow_error("BF16 GDN gating workspace element count overflows size_t");
    }
    const std::size_t elements = split * tokens * logical_rows;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("BF16 GDN gating workspace byte count overflows size_t");
    }
    return elements * sizeof(float);
}

void execute_resolved(const Bf16GdnGatingPlan& plan, const Bf16GdnGatingProblem& problem,
                      const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                      const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                      Tensor& beta, cudaStream_t stream) {
    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }

    switch (plan.schedule) {
    case Bf16GdnGatingScheduleId::GemvPairedRows:
        bf16_gdn_gating_proj_gemv_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        bf16_gdn_gating_proj_small_t_split10_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                    scratch.data, scratch.bytes, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        bf16_gdn_gating_proj_35_simt_c4_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               stream);
        return;
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        bf16_gdn_gating_proj_35_simt_c8_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        bf16_gdn_gating_proj_35_mma_split32_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        bf16_gdn_gating_proj_35_mma_split16_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_split8_launch(plan.token_variant, x, a_weight, b_weight,
                                                      A_log, dt_bias, scratch.data, g, beta,
                                                      stream);
        } else {
            bf16_gdn_gating_proj_mma_split8_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        }
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_split4_launch(plan.token_variant, x, a_weight, b_weight,
                                                      A_log, dt_bias, scratch.data, g, beta,
                                                      stream);
        } else {
            bf16_gdn_gating_proj_mma_split4_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        }
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_split2_launch(plan.token_variant, x, a_weight, b_weight,
                                                      A_log, dt_bias, scratch.data, g, beta,
                                                      stream);
        } else {
            bf16_gdn_gating_proj_mma_split2_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        }
        return;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight,
                                                       A_log, dt_bias, g, beta, stream);
        } else {
            bf16_gdn_gating_proj_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight,
                                                    A_log, dt_bias, g, beta, stream);
        }
        return;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

template <std::size_t N>
Bf16GdnGatingPlan resolve_with_fallback(Bf16GdnGatingScheduleId catalog_schedule,
                                        const std::array<Bf16GdnGatingScheduleId, N>& by_split,
                                        const Bf16GdnGatingProblem& problem,
                                        const OccupancyPolicy& occupancy) {
    if (candidate_is_legal(catalog_schedule, problem, occupancy)) {
        return bf16_gdn_gating_resolve_candidate(catalog_schedule, problem, occupancy);
    }
    bool passed_catalog = false;
    for (const Bf16GdnGatingScheduleId schedule : by_split) {
        if (schedule == catalog_schedule) {
            passed_catalog = true;
            continue;
        }
        if (passed_catalog && candidate_is_legal(schedule, problem, occupancy)) {
            return bf16_gdn_gating_resolve_candidate(schedule, problem, occupancy);
        }
    }
    throw std::invalid_argument("BF16 GDN gating: no resident fallback for exact problem");
}

std::size_t interval_capacity(const Bf16GdnGatingProblem& base, std::int32_t min_cols,
                              std::int32_t max_cols, const OccupancyPolicy& occupancy) {
    std::size_t maximum = 0;
    const auto include = [&](std::int32_t cols) {
        maximum = std::max(maximum, bf16_gdn_gating_resolve_plan(
                                        {base.heads, base.input_rows, cols}, occupancy)
                                        .workspace_bytes);
    };
    include(min_cols);
    include(max_cols);

    // Every workspace-bearing route is in T=1..4096. Exhausting this small, fixed admission
    // domain is an exact capacity proof across catalog and architecture-residency transitions;
    // the unbounded tail is the zero-workspace unsplit route.
    const std::int32_t first = std::max<std::int32_t>(min_cols, 1);
    const std::int32_t last  = std::min(max_cols, kLastCooperativeCols);
    for (std::int32_t cols = first; cols <= last; ++cols) { include(cols); }
    return maximum;
}

} // namespace

const char* bf16_gdn_gating_schedule_name(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::GemvPairedRows:
        return "gdn_gating_proj.bf16.gemv.paired_rows";
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return "gdn_gating_proj.bf16.small_t.split10";
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        return "gdn_gating_proj.bf16.simt.warp_row.c4";
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return "gdn_gating_proj.bf16.simt.warp_row.c8";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        return "gdn_gating_proj.bf16.mma.cooperative_split32";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        return "gdn_gating_proj.bf16.mma.cooperative_split16";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return "gdn_gating_proj.bf16.mma.cooperative_split8";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return "gdn_gating_proj.bf16.mma.cooperative_split4";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return "gdn_gating_proj.bf16.mma.cooperative_split2";
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return "gdn_gating_proj.bf16.mma.unsplit";
    }
    return "gdn_gating_proj.bf16.unknown";
}

const char* bf16_gdn_norm_gating_schedule_name(Bf16GdnNormGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnNormGatingScheduleId::Composed:
        return "gdn_norm_gating_proj.bf16.composed";
    case Bf16GdnNormGatingScheduleId::MmaCooperativeSplit32:
        return "gdn_norm_gating_proj.bf16.mma.cooperative_split32";
    }
    return "gdn_norm_gating_proj.bf16.unknown";
}

bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept {
    if (problem.cols < 1) { return false; }
    return is_27(problem) || is_35(problem);
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId schedule,
                                                    const Bf16GdnGatingProblem& problem,
                                                    const OccupancyPolicy& occupancy) {
    if (!candidate_is_legal(schedule, problem, occupancy)) {
        throw std::invalid_argument("BF16 GDN gating: candidate is not legal for exact problem");
    }
    const bool mma                          = schedule_uses_mma(schedule);
    const Bf16GdnGatingTokenVariant variant = !mma ? Bf16GdnGatingTokenVariant::None
                                                   : ((problem.cols % mma_tile_cols(problem)) == 0
                                                          ? Bf16GdnGatingTokenVariant::Full
                                                          : Bf16GdnGatingTokenVariant::Predicated);
    const std::int32_t split_k              = schedule_split_k(schedule);
    const std::size_t workspace =
        split_k > 1 ? checked_partial_bytes(problem.heads, split_k, problem.cols) : 0;
    return {schedule, variant, workspace};
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem,
                                               const OccupancyPolicy& occupancy) {
    if (!bf16_gdn_gating_admits(problem)) {
        throw std::invalid_argument(
            "BF16 GDN gating: exact problem or column count is not admitted");
    }
    static constexpr std::array<Bf16GdnGatingScheduleId, 4> k27CoopBySplit{{
        Bf16GdnGatingScheduleId::MmaCooperativeSplit8,
        Bf16GdnGatingScheduleId::MmaCooperativeSplit4,
        Bf16GdnGatingScheduleId::MmaCooperativeSplit2,
        Bf16GdnGatingScheduleId::MmaUnsplit,
    }};
    static constexpr std::array<Bf16GdnGatingScheduleId, 6> k35CoopBySplit{{
        Bf16GdnGatingScheduleId::MmaCooperativeSplit32,
        Bf16GdnGatingScheduleId::MmaCooperativeSplit16,
        Bf16GdnGatingScheduleId::MmaCooperativeSplit8,
        Bf16GdnGatingScheduleId::MmaCooperativeSplit4,
        Bf16GdnGatingScheduleId::MmaCooperativeSplit2,
        Bf16GdnGatingScheduleId::MmaUnsplit,
    }};
    if (is_27(problem)) {
        for (const RouteSpec& route : k27Routes) {
            if (route.cols.contains(problem.cols)) {
                return resolve_with_fallback(route.schedule, k27CoopBySplit, problem, occupancy);
            }
        }
    } else {
        for (const RouteSpec& route : k35Routes) {
            if (route.cols.contains(problem.cols)) {
                return resolve_with_fallback(route.schedule, k35CoopBySplit, problem, occupancy);
            }
        }
    }
    throw std::logic_error("BF16 GDN gating: admitted problem has no covering route");
}

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_cols, std::int32_t max_cols,
                                                     const OccupancyPolicy& occupancy) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("BF16 GDN gating: invalid column interval");
    }
    const Bf16GdnGatingProblem base{heads, input_rows, 1};
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, min_cols}, occupancy);
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, max_cols}, occupancy);
    return interval_capacity(base, min_cols, max_cols, occupancy);
}

Bf16GdnNormGatingPlan bf16_gdn_norm_gating_resolve_plan(const Bf16GdnGatingProblem& problem,
                                                        const OccupancyPolicy& occupancy) {
    Bf16GdnGatingPlan control            = bf16_gdn_gating_resolve_plan(problem, occupancy);
    Bf16GdnNormGatingScheduleId schedule = Bf16GdnNormGatingScheduleId::Composed;
    std::int32_t norm_splits             = 0;
    if (is_35(problem) && problem.cols <= 16) {
        control  = bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId::MmaCooperativeSplit32,
                                                     problem, occupancy);
        schedule = Bf16GdnNormGatingScheduleId::MmaCooperativeSplit32;
        norm_splits = 32;
    }
    const std::size_t norm_partial_bytes =
        static_cast<std::size_t>(norm_splits) * problem.cols * sizeof(float);
    return {schedule, control, control.workspace_bytes + norm_partial_bytes};
}

std::size_t bf16_gdn_norm_gating_capacity_workspace_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_cols,
                                                          std::int32_t max_cols,
                                                          const OccupancyPolicy& occupancy) {
    std::size_t maximum = bf16_gdn_gating_capacity_workspace_bytes(heads, input_rows, min_cols,
                                                                  max_cols, occupancy);
    if (heads == 32 && input_rows == 2048 && min_cols <= 16) {
        const std::int32_t fused_cols = std::min<std::int32_t>(max_cols, 16);
        maximum                       = std::max(
            maximum, bf16_gdn_norm_gating_resolve_plan({heads, input_rows, fused_cols}, occupancy)
                         .workspace_bytes);
    }
    return maximum;
}

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, const OccupancyPolicy& occupancy,
                                  cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnGatingPlan resolved = bf16_gdn_gating_resolve_plan(problem, occupancy);
    if (resolved.schedule != plan.schedule || resolved.token_variant != plan.token_variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("BF16 GDN gating: plan does not match the exact problem");
    }
    execute_resolved(plan, problem, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_gating_execute_candidate(Bf16GdnGatingScheduleId schedule, const Tensor& x,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias,
                                       WorkspaceArena& ws, Tensor& g, Tensor& beta,
                                       const OccupancyPolicy& occupancy, cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_candidate(schedule, problem, occupancy);
    execute_resolved(plan, problem, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, const OccupancyPolicy& occupancy,
                              cudaStream_t stream) {
    const Bf16GdnGatingPlan plan =
        bf16_gdn_gating_resolve_plan({g.ne[0], x.ne[0], x.ne[1]}, occupancy);
    bf16_gdn_gating_execute_plan(plan, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, occupancy,
                                 stream);
}

void bf16_gdn_norm_gating_dispatch(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                                   const Weight& a_weight, const Weight& b_weight,
                                   const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                   Tensor& g, Tensor& beta, const OccupancyPolicy& occupancy,
                                   cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnNormGatingPlan plan = bf16_gdn_norm_gating_resolve_plan(problem, occupancy);
    if (plan.schedule == Bf16GdnNormGatingScheduleId::Composed) {
        rmsnorm(x, norm_weight, eps, true, h, stream);
        execute_resolved(plan.control, problem, h, a_weight, b_weight, A_log, dt_bias, ws, g, beta,
                         stream);
        return;
    }

    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }
    bf16_gdn_norm_gating_proj_35_mma_split32_launch(plan.control.token_variant, x, norm_weight, eps,
                                                    h, a_weight, b_weight, A_log, dt_bias,
                                                    scratch.data, g, beta, stream);
}

} // namespace ginfer::ops::detail
