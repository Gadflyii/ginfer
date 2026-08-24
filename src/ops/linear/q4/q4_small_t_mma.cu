#include "ops/linear/q4/q4_launch.h"

#include "core/device.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ginfer::ops::detail {
namespace {

constexpr int kFirstSmallT    = 2;
constexpr int kLastFullT      = 8;
constexpr int kLastOptimizedT = 20;
using FullGeometry            = Q4DraftHeadGeometry<5120>;
using OptimizedGeometry       = Q4DraftHeadGeometry<2048>;

#if defined(GINFER_COMPILED_CC) && GINFER_COMPILED_CC >= 120
template <int InputRows>
struct Q4ClosedSmallTGeometry {
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
    static constexpr int kOutputRows   = 0;
};

template <int RowsPerCta>
struct Q4ClosedSmallTRows {
    static constexpr int kOutputRowsPerCta = RowsPerCta;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }

    __device__ __forceinline__ int output_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

// This is the sm_120a-only closed small-T route used by the registered Muse and Qwen DFlash2
// matrices. Thirty-two rows amortize the activation tile while retaining a four-block launch
// bound with the 8/16-column shared-memory footprint.
using Q4ClosedSmallTSchedule = Q4SmallTMmaSchedule<32, 4>;
using Q4ClosedSmallTRows32   = Q4ClosedSmallTRows<Q4ClosedSmallTSchedule::kRowsPerCta>;
#endif

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4DraftSmallTSchedule;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    q4_small_t_mma_kernel<Geometry, TileTokens, ActiveTokens>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, ((First + static_cast<int>(Offsets) + 7) / 8) * 8,
                      First + static_cast<int>(Offsets)>...};
}

constexpr auto kFullLaunchers = make_launchers<FullGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastFullT - kFirstSmallT + 1>{});
constexpr auto kOptimizedLaunchers = make_launchers<OptimizedGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastOptimizedT - kFirstSmallT + 1>{});

template <class Geometry>
bool matches(const Tensor& x, const Weight& weight) {
    return weight.n == Geometry::kOutputRows && weight.k == Geometry::kInputRows &&
           weight.padded_shape[1] == Geometry::kInputRows && x.ne[1] >= kFirstSmallT;
}

#if defined(GINFER_COMPILED_CC) && GINFER_COMPILED_CC >= 120
template <int InputRows, int TileCols>
void launch_closed_tile(const Tensor& x, const Weight& weight, Tensor& out,
                        cudaStream_t stream) {
    using Geometry = Q4ClosedSmallTGeometry<InputRows>;
    const int rows = out.ne[0];
    const int cols = x.ne[1];
    if ((rows % Q4ClosedSmallTSchedule::kRowsPerCta) != 0) {
        throw std::invalid_argument(
            "Q4 Linear closed small-T: output rows do not cover exact CTAs");
    }

    const int blocks = rows / Q4ClosedSmallTSchedule::kRowsPerCta;
    q4_small_t_mma_kernel<Geometry, TileCols, 0, Q4SmallTMmaStoreEpilogue,
                          Q4ClosedSmallTRows32, Q4ClosedSmallTSchedule>
        <<<blocks, Q4ClosedSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            Q4SmallTMmaStoreEpilogue{}, Q4ClosedSmallTRows32{}, cols, rows);
    CUDA_CHECK(cudaGetLastError());
}

template <int InputRows>
void launch_closed_k(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 8) {
        launch_closed_tile<InputRows, 8>(x, weight, out, stream);
    } else {
        launch_closed_tile<InputRows, 16>(x, weight, out, stream);
    }
}

bool closed_problem(std::int32_t n, std::int32_t k) {
    switch (k) {
    case 4096:
        return n == 5120 || n == 6656;
    case 5120:
        return n == 4096 || n == 17408;
    case 6656:
        return n == 256 || n == 1024 || n == 1664 || n == 4096 || n == 6656 || n == 19968;
    case 17408:
    case 25600:
        return n == 5120;
    case 19968:
    case 33280:
        return n == 6656;
    default:
        return false;
    }
}
#endif

} // namespace

void launch_q4_draft_head_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    if (matches<FullGeometry>(x, weight) && x.ne[1] <= kLastFullT) {
        kFullLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out, stream);
        return;
    }
    if (matches<OptimizedGeometry>(x, weight) && x.ne[1] <= kLastOptimizedT) {
        kOptimizedLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out,
                                                                              stream);
        return;
    }
    throw std::invalid_argument("Q4 Linear draft-head small-T: unsupported exact problem");
}

void launch_q4_closed_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
#if defined(GINFER_COMPILED_CC) && GINFER_COMPILED_CC >= 120
    if (x.ne[1] < 2 || x.ne[1] > 16 || weight.padded_shape[1] != weight.k ||
        !closed_problem(weight.n, weight.k)) {
        throw std::invalid_argument("Q4 Linear closed small-T: unsupported exact problem");
    }
    switch (weight.k) {
    case 4096:
        launch_closed_k<4096>(x, weight, out, stream);
        return;
    case 5120:
        launch_closed_k<5120>(x, weight, out, stream);
        return;
    case 6656:
        launch_closed_k<6656>(x, weight, out, stream);
        return;
    case 17408:
        launch_closed_k<17408>(x, weight, out, stream);
        return;
    case 19968:
        launch_closed_k<19968>(x, weight, out, stream);
        return;
    case 25600:
        launch_closed_k<25600>(x, weight, out, stream);
        return;
    case 33280:
        launch_closed_k<33280>(x, weight, out, stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("Q4 Linear closed small-T: unsupported input width");
#else
    (void)x;
    (void)weight;
    (void)out;
    (void)stream;
    throw std::invalid_argument("Q4 Linear closed small-T requires sm_120a");
#endif
}

} // namespace ginfer::ops::detail
