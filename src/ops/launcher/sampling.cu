// Implements: include/ginfer/ops/sampling.h
// Match: validated contiguous BF16/I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use the same layout authority, so exactly one
// finite route owns each shape. Muse token_domain 202048 uses the hierarchical top-64 route;
// Qwen vocabs keep the 20-candidate multiblock geometry.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

namespace ginfer::ops::detail {

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    const WideSamplingWorkspaceLayout wide =
        make_wide_sampling_workspace_layout(token_domain, columns);
    if (wide.multiblock) { return wide.bytes; }
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t batch             = logits.ne[1];
    const auto* positions                = static_cast<const std::int32_t*>(logical_positions.data);
    const WideSamplingWorkspaceLayout wide_layout =
        make_wide_sampling_workspace_layout(token_domain, batch);
    if (wide_layout.multiblock) {
        const std::int32_t partial_blocks = sampler_wide_partial_count(token_domain);
        const std::int32_t groups         = sampler_wide_group_count(partial_blocks);
        const WideSamplingWorkspace scratch = wide_layout.bind(workspace);
        const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                                static_cast<unsigned int>(batch));
        // Full radix sort wins at one or two rows; at three or more rows the exact radix-select
        // partials avoid enough exchange work to offset their shared-histogram synchronization.
        const bool radix_select = batch > 2;
        if (radix_select) {
            sampling_wide_partial_topk_kernel<true><<<partial_grid, kSamplerBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain,
                physical_rows, scratch);
        } else {
            sampling_wide_partial_topk_kernel<false><<<partial_grid, kSamplerBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain,
                physical_rows, scratch);
        }
        CUDA_CHECK(cudaGetLastError());
        const dim3 group_grid(static_cast<unsigned int>(groups),
                              static_cast<unsigned int>(batch));
        if (radix_select) {
            sampling_wide_group_finalize_sample_kernel<true>
                <<<group_grid, kSamplerGroupBlock, 0, stream>>>(
                    static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
                    partial_blocks, groups, scratch);
        } else {
            sampling_wide_group_finalize_sample_kernel<false>
                <<<group_grid, kSamplerGroupBlock, 0, stream>>>(
                    static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
                    partial_blocks, groups, scratch);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    if (!layout.multiblock) {
        sample_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain, physical_rows,
        scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
        partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ginfer::ops::detail
