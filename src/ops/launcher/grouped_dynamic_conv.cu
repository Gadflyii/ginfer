#include "ops/launcher/grouped_dynamic_conv.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"
#include "ginfer/ops/linear.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ginfer::ops::detail {
namespace {

#if GINFER_COMPILED_CC >= 120
constexpr int kMuseHidden      = 6656;
constexpr int kMuseDynamicRows = 1664;
constexpr int kMuseMaxColumns  = 128;

struct MuseDynamicProjectionGeometry {
    static constexpr int kInputRows  = kMuseHidden;
    static constexpr int kOutputRows = kMuseDynamicRows;
};

// DFlash has at most 128 compact columns. A 16-row tile gives the B=1 draft path enough
// independent CTAs to cover most of a 5090 while eight warps reuse each 16xK weight tile across
// 64 columns. Larger compact batches retain the 16-row tile and expose all 104 output-row CTAs;
// the compact activation remains L2-resident while each CTA streams a disjoint weight tile.
using MuseNarrowSchedule =
    Bf16MmaSchedule<16, 64, 64, 16, 8, 3, 1, Cache::cg, Cache::cg,
                    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast>;
using MuseWideSchedule =
    Bf16MmaSchedule<16, 128, 64, 16, 16, 2, 1, Cache::cg, Cache::cg,
                    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast>;

static_assert(kMuseHidden * static_cast<int>(sizeof(__nv_bfloat16)) ==
              2 * kMuseDynamicRows * static_cast<int>(sizeof(float)));

template <class Schedule, bool FullColumns>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) void
muse_projection_splitk2_kernel(const __nv_bfloat16* __restrict__ hidden,
                               const __nv_bfloat16* __restrict__ projection,
                               float* __restrict__ partial, int columns) {
    constexpr int BM             = Schedule::kBlockRows;
    constexpr int BN             = Schedule::kBlockCols;
    constexpr int BK             = Schedule::kBlockK;
    constexpr int WM             = Schedule::kWarpRows;
    constexpr int WN             = Schedule::kWarpCols;
    constexpr int MT             = Schedule::kMmaRows;
    constexpr int NT             = Schedule::kMmaCols;
    constexpr int KSUB           = Schedule::kMmaK;
    constexpr int S              = Schedule::kPipelineStages;
    constexpr int WARPS_N        = Schedule::kWarpsN;
    constexpr int THREADS        = Schedule::kThreads;
    constexpr int kSplitK        = 2;
    constexpr int kKPerSplit     = kMuseHidden / kSplitK;
    constexpr int kTilesPerSplit = kKPerSplit / BK;
    static_assert(kMuseDynamicRows % BM == 0);
    static_assert(kMuseHidden % kSplitK == 0 && kKPerSplit % BK == 0);
    static_assert(kTilesPerSplit >= S);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* Bs = As + S * BM * BK;

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int wm     = warp / WARPS_N;
    const int wn     = warp - wm * WARPS_N;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int split    = static_cast<int>(blockIdx.x) & 1;
    const int tile_m   = static_cast<int>(blockIdx.x) >> 1;
    const int m0       = tile_m * BM;
    const int split_k0 = split * kKPerSplit;

    float accum[MT][NT][4] = {};

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    auto stage_inputs = [&](int stage, int local_k_tile) {
        const int k0  = split_k0 + local_k_tile * BK;
        auto* a_stage = As + stage * BM * BK;
        auto* b_stage = Bs + stage * BN * BK;

#pragma unroll 1
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row = item / (BK / 8);
            const int k8  = item - row * (BK / 8);
            const int kk  = k8 * 8;
            cp_async<16, Schedule::kWeightCache>(
                &a_stage[row * BK + bf16_mma_shared_col<Schedule>(row, kk)],
                &projection[static_cast<std::int64_t>(m0 + row) * kMuseHidden + k0 + kk]);
        }

#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int token = item / (BK / 8);
            const int k8    = item - token * (BK / 8);
            const int kk    = k8 * 8;
            auto* dst       = &b_stage[token * BK + bf16_mma_shared_col<Schedule>(token, kk)];
            if constexpr (FullColumns) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &hidden[static_cast<std::int64_t>(token) * kMuseHidden + k0 + kk]);
            } else {
                const bool valid = token < columns;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst,
                    &hidden[static_cast<std::int64_t>(valid ? token : 0) * kMuseHidden + k0 + kk],
                    valid ? 16 : 0);
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

#pragma unroll 1
    for (int k_tile = 0; k_tile < kTilesPerSplit; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= kTilesPerSplit) {
            cp_wait<S - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_frag)[MT][4],
                                  unsigned(&b_frag)[NT][2]) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + a_row_offset;
                const int col = k_step * 16 + a_col_offset;
                ldmatrix_x4(
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    smem_addr(
                        &As[stage * BM * BK + row * BK + bf16_mma_shared_col<Schedule>(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = wn * WN + ni * 8 + b_inner_row;
                const int col = k_step * 16 + b_k_offset;
                ldmatrix_x2(
                    b_frag[ni][0], b_frag[ni][1],
                    smem_addr(
                        &Bs[stage * BN * BK + row * BK + bf16_mma_shared_col<Schedule>(row, col)]));
            }
        };

        unsigned a_frag[2][MT][4];
        unsigned b_frag[2][NT][2];
        load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < KSUB) {
                load_fragments(k_step + 1, a_frag[slot ^ 1], b_frag[slot ^ 1]);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                             accum[mi][ni][3], a_frag[slot][mi][0], a_frag[slot][mi][1],
                             a_frag[slot][mi][2], a_frag[slot][mi][3], b_frag[slot][ni][0],
                             b_frag[slot][ni][1]);
                }
            }
        }

        __syncthreads();
        const int next = k_tile + S;
        if (next < kTilesPerSplit) {
            stage_inputs(stage, next);
            cp_commit();
        }
    }

    const std::int64_t split_offset =
        static_cast<std::int64_t>(split) * columns * kMuseDynamicRows;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row0 = m0 + wm * WM + mi * 16 + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int token0   = wn * WN + ni * 8 + 2 * lid;
            const int token1   = token0 + 1;
            const float* value = accum[mi][ni];
            auto store = [&](int row, int token, float result) {
                partial[split_offset + static_cast<std::int64_t>(token) * kMuseDynamicRows + row] =
                    result;
            };
            if constexpr (FullColumns) {
                store(row0, token0, value[0]);
                store(row0, token1, value[1]);
                store(row1, token0, value[2]);
                store(row1, token1, value[3]);
            } else {
                if (token0 < columns) {
                    store(row0, token0, value[0]);
                    store(row1, token0, value[2]);
                }
                if (token1 < columns) {
                    store(row0, token1, value[1]);
                    store(row1, token1, value[3]);
                }
            }
        }
    }
}

__global__ void muse_projection_splitk2_reduce_kernel(const float* __restrict__ partial,
                                                       __nv_bfloat16* __restrict__ dynamic,
                                                       int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) {
        dynamic[index] = __float2bfloat16_rn(partial[index] + partial[elements + index]);
    }
}

template <class Schedule, bool FullColumns>
void launch_muse_projection_splitk2(const __nv_bfloat16* hidden,
                                    const __nv_bfloat16* projection, __nv_bfloat16* dynamic,
                                    __nv_bfloat16* output_scratch, int columns,
                                    cudaStream_t stream) {
    constexpr int kRowTiles = kMuseDynamicRows / Schedule::kBlockRows;
    muse_projection_splitk2_kernel<Schedule, FullColumns>
        <<<2 * kRowTiles, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            hidden, projection, reinterpret_cast<float*>(output_scratch), columns);
    CUDA_CHECK(cudaGetLastError());
    const int elements = kMuseDynamicRows * columns;
    constexpr int kThreads = 256;
    muse_projection_splitk2_reduce_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                                             stream>>>(
        reinterpret_cast<const float*>(output_scratch), dynamic, elements);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool FullColumns>
void launch_muse_projection_variant(const __nv_bfloat16* hidden, const __nv_bfloat16* projection,
                                    __nv_bfloat16* dynamic, int columns, cudaStream_t stream) {
    constexpr int kRowTiles = kMuseDynamicRows / Schedule::kBlockRows;
    const int column_tiles = (columns + Schedule::kBlockCols - 1) / Schedule::kBlockCols;
    const Bf16MmaContiguousOutput output{dynamic, kMuseDynamicRows};
    bf16_gemm_mma_kernel<MuseDynamicProjectionGeometry, Schedule, FullColumns>
        <<<kRowTiles * column_tiles, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            hidden, projection, output, columns);
    CUDA_CHECK(cudaGetLastError());
}

void launch_muse_bf16_projection(const __nv_bfloat16* hidden,
                                 const __nv_bfloat16* projection, __nv_bfloat16* dynamic,
                                 __nv_bfloat16* output_scratch, int columns,
                                 cudaStream_t stream) {
    if (columns <= MuseNarrowSchedule::kBlockCols) {
        if (columns == MuseNarrowSchedule::kBlockCols) {
            launch_muse_projection_variant<MuseNarrowSchedule, true>(hidden, projection, dynamic,
                                                                      columns, stream);
        } else {
            launch_muse_projection_variant<MuseNarrowSchedule, false>(hidden, projection, dynamic,
                                                                       columns, stream);
        }
    } else if (columns == MuseWideSchedule::kBlockCols) {
        launch_muse_projection_splitk2<MuseWideSchedule, true>(
            hidden, projection, dynamic, output_scratch, columns, stream);
    } else {
        launch_muse_projection_splitk2<MuseWideSchedule, false>(
            hidden, projection, dynamic, output_scratch, columns, stream);
    }
}

bool is_aligned_16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}
#endif

__global__ void gemm_pt_a_ginfer_b_kernel(const __nv_bfloat16* __restrict__ a_nk,
                                          const __nv_bfloat16* __restrict__ b_kt,
                                          __nv_bfloat16* __restrict__ c_nt, int n, int k, int t) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    if (row >= n || col >= t) { return; }
    float acc = 0.0f;
    const __nv_bfloat16* a_row = a_nk + static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* b_col = b_kt + static_cast<std::int64_t>(col) * k;
    for (int i = 0; i < k; ++i) {
        acc += __bfloat162float(a_row[i]) * __bfloat162float(b_col[i]);
    }
    c_nt[row + static_cast<std::int64_t>(col) * n] = __float2bfloat16(acc);
}

__global__ void grouped_dyn_conv_kernel(const __nv_bfloat16* __restrict__ hidden,
                                        const __nv_bfloat16* __restrict__ dynamic,
                                        const __nv_bfloat16* __restrict__ base,
                                        __nv_bfloat16* __restrict__ out, int hidden_size,
                                        int width, int group_size, int stage) {
    const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    if (h >= hidden_size || column >= width) { return; }
    constexpr int kKernel = 2;
    const int groups      = hidden_size / group_size;
    const int g           = h / group_size;
    const int pack        = kKernel * groups;
    float acc             = 0.0f;
    for (int off = 0; off < kKernel; ++off) {
        float x = 0.0f;
        if (column >= off) {
            const std::int64_t source_column =
                static_cast<std::int64_t>(batch) * width + column - off;
            x = __bfloat162float(hidden[h + source_column * hidden_size]);
        }
        const float bw = __bfloat162float(
            base[(static_cast<std::int64_t>(stage) * kKernel + off) * hidden_size + h]);
        const int dyn_row = stage * pack + off * groups + g;
        const std::int64_t dynamic_column = static_cast<std::int64_t>(batch) * width + column;
        const float dw = __bfloat162float(dynamic[dyn_row + dynamic_column * (2 * pack)]);
        acc += (bw + dw) * x;
    }
    const std::int64_t output_column = static_cast<std::int64_t>(batch) * width + column;
    out[h + output_column * hidden_size] = __float2bfloat16(acc);
}

void launch_gemm(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n, int k,
                 int t, cudaStream_t stream) {
    const dim3 block(32, 1, 1);
    const dim3 grid(static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>(t), 1u);
    gemm_pt_a_ginfer_b_kernel<<<grid, block, 0, stream>>>(a, b, c, n, k, t);
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv(const __nv_bfloat16* hidden, const __nv_bfloat16* dynamic,
                 const __nv_bfloat16* base, __nv_bfloat16* out, int hidden_size, int width,
                 int batch, int group_size, int stage, cudaStream_t stream) {
    const dim3 block(128, 1, 1);
    const dim3 grid(static_cast<unsigned>((hidden_size + 127) / 128),
                    static_cast<unsigned>(width), static_cast<unsigned>(batch));
    grouped_dyn_conv_kernel<<<grid, block, 0, stream>>>(hidden, dynamic, base, out, hidden_size,
                                                        width, group_size, stage);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& proj,
                                         const Tensor& base, Tensor& dynamic, Tensor& out,
                                         std::int32_t group_size, cudaStream_t stream) {
    const int h = hidden.ne[0];
    const int width = hidden.ne[1];
    const int batch = hidden.ne[2];
    const int columns = width * batch;
    const int n = dynamic.ne[0];
#if GINFER_COMPILED_CC >= 120
    if (h == kMuseHidden && n == kMuseDynamicRows && columns <= kMuseMaxColumns &&
        is_aligned_16(hidden.data) && is_aligned_16(proj.data) && is_aligned_16(dynamic.data) &&
        is_aligned_16(out.data)) {
        launch_muse_bf16_projection(static_cast<const __nv_bfloat16*>(hidden.data),
                                    static_cast<const __nv_bfloat16*>(proj.data),
                                    static_cast<__nv_bfloat16*>(dynamic.data),
                                    static_cast<__nv_bfloat16*>(out.data), columns, stream);
    } else
#endif
    {
        launch_gemm(static_cast<const __nv_bfloat16*>(proj.data),
                    static_cast<const __nv_bfloat16*>(hidden.data),
                    static_cast<__nv_bfloat16*>(dynamic.data), n, h, columns, stream);
    }
    launch_conv(static_cast<const __nv_bfloat16*>(hidden.data),
                static_cast<const __nv_bfloat16*>(dynamic.data),
                static_cast<const __nv_bfloat16*>(base.data), static_cast<__nv_bfloat16*>(out.data),
                h, width, batch, group_size, 0, stream);
}

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Weight& projection,
                                         const Tensor& base, Tensor& dynamic, Tensor& out,
                                         std::int32_t group_size, cudaStream_t stream) {
    const int h = hidden.ne[0];
    const int width = hidden.ne[1];
    const int batch = hidden.ne[2];
    const int columns = width * batch;
    Tensor hidden_matrix = hidden.reshape({h, columns});
    Tensor dynamic_matrix = dynamic.reshape({dynamic.ne[0], columns});
    ginfer::ops::linear(hidden_matrix, projection, dynamic_matrix, stream);
    launch_conv(static_cast<const __nv_bfloat16*>(hidden.data),
                static_cast<const __nv_bfloat16*>(dynamic.data),
                static_cast<const __nv_bfloat16*>(base.data),
                static_cast<__nv_bfloat16*>(out.data), h, width, batch, group_size, 0, stream);
}

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& dynamic,
                                        const Tensor& base, Tensor& out, std::int32_t group_size,
                                        cudaStream_t stream) {
    const int h = hidden.ne[0];
    const int width = hidden.ne[1];
    const int batch = hidden.ne[2];
    launch_conv(static_cast<const __nv_bfloat16*>(hidden.data),
                static_cast<const __nv_bfloat16*>(dynamic.data),
                static_cast<const __nv_bfloat16*>(base.data), static_cast<__nv_bfloat16*>(out.data),
                h, width, batch, group_size, 1, stream);
}

} // namespace ginfer::ops::detail
