#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstdint>

namespace ginfer::ops::detail {
namespace {

template <class Schedule>
void launch_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    const std::size_t dynamic_shared_bytes =
        Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK
            ? static_cast<std::size_t>(k) * sizeof(__nv_bfloat16)
            : 0u;

    if (dynamic_shared_bytes > 0) {
        CUDA_CHECK(cudaFuncSetAttribute(
            q4_rowsplit_gemv_kernel<Schedule>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dynamic_shared_bytes)));
    }
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, dynamic_shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), nullptr,
        rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_gemv<Q4GemvR4W1DirectSchedule>(x, w, out, stream);
}

void launch_q4_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_gemv<Q4GemvR1W8DirectSchedule>(x, w, out, stream);
}

template <int GroupsPerRow, int WarpsPerRow = 8>
void launch_warp_split(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const dim3 grid(static_cast<unsigned>(rows), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(WarpsPerRow * 32), 1u, 1u);
    q4_gemv_warp_split_kernel<GroupsPerRow, WarpsPerRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_gemv_r1_w8_async(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    switch (k) {
    case 4096:
        launch_warp_split<64>(x, w, out, stream);
        return;
    case 6656:
        launch_warp_split<104>(x, w, out, stream);
        return;
    case 19968:
        launch_warp_split<312>(x, w, out, stream);
        return;
    case 33280:
        launch_warp_split<520>(x, w, out, stream);
        return;
    default:
        launch_gemv<Q4GemvR1W8AsyncDirectSchedule>(x, w, out, stream);
        return;
    }
}

} // namespace ginfer::ops::detail
