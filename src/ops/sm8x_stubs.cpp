#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <stdexcept>

namespace ginfer::ops::detail {
namespace {

[[noreturn]] void reject_nvfp4() {
    throw std::runtime_error("NVFP4 execution requires an sm_120a GInfer image");
}

[[noreturn]] void reject_fp8_kind() {
    throw std::runtime_error("FP8 kind::f8f6f4 execution requires an sm_120a GInfer image");
}

} // namespace

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight&, const char*) { reject_nvfp4(); }

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                  std::int32_t, std::int32_t) {
    reject_nvfp4();
}

void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena*,
                    cudaStream_t) {
    reject_nvfp4();
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}

void nvfp4_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                               LinearPolicy, WorkspaceArena*, cudaStream_t) {
    reject_nvfp4();
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                      std::int32_t, std::int32_t) {
    reject_nvfp4();
}

void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                               WorkspaceArena&, cudaStream_t) {
    reject_nvfp4();
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}

void nvfp4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                  WorkspaceArena&, cudaStream_t) {
    reject_nvfp4();
}

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}

void nvfp4_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                              WorkspaceArena*, cudaStream_t) {
    reject_nvfp4();
}

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    reject_nvfp4();
}

void nvfp4_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                 Tensor&, Tensor&, LinearPolicy, WorkspaceArena&, cudaStream_t) {
    reject_nvfp4();
}

void nvfp4_gdn_record_small_t_launch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}

void nvfp4_gdn_record_post_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4();
}

void launch_fp8_a8_quantize(const Tensor&, const Weight&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_kind();
}

void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_kind();
}

void fp8_attn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Fp8A8Workspace, cudaStream_t) {
    reject_fp8_kind();
}

void fp8_gdn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    reject_fp8_kind();
}

void fp8_linear_add_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                              cudaStream_t) {
    reject_fp8_kind();
}

void fp8_linear_swiglu_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t) {
    reject_fp8_kind();
}

} // namespace ginfer::ops::detail
