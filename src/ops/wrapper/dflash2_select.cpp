#include "ginfer/ops/dflash2_select.h"

#include "core/layout.h"
#include "ops/common/dflash2_accept_workspace.h"
#include "ops/common/dflash2_selector_workspace.h"
#include "ops/launcher/dflash2_select.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ginfer::ops {
namespace {

void require_contiguous(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.dtype != dtype || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("dflash2: invalid ") + label);
    }
}

void require_vector(const Tensor& tensor, DType dtype, std::int32_t count, const char* label) {
    require_contiguous(tensor, dtype, label);
    if (tensor.ne[0] != count || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("dflash2: invalid ") + label + " shape");
    }
}

void require_shape(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                   std::int32_t n2, const char* label) {
    require_contiguous(tensor, dtype, label);
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("dflash2: invalid ") + label + " shape");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_selector_weight(const Weight& weight, std::int32_t rows, std::int32_t columns,
                             const char* label) {
    if (weight.ndim != 2 || weight.n != rows || weight.k != columns ||
        weight.shape[0] != rows || weight.shape[1] != columns || weight.qdata == nullptr) {
        throw std::invalid_argument(std::string("dflash2: invalid ") + label + " shape");
    }
    if (weight.qtype == QType::BF16_CTRL) {
        const std::uint64_t required = static_cast<std::uint64_t>(rows) * columns * 2ULL;
        if (weight.layout != QuantLayout::Contiguous || weight.qhigh != nullptr ||
            weight.scales != nullptr || weight.payload_bytes < required) {
            throw std::invalid_argument(std::string("dflash2: invalid BF16 ") + label);
        }
        return;
    }
    if (weight.qtype == QType::Q4G64_F16S) {
        const std::int32_t padded_columns = (columns + 127) & ~127;
        const std::uint64_t groups = static_cast<std::uint64_t>(rows) * padded_columns / 64ULL;
        const std::uint64_t code_bytes = groups * 32ULL;
        const std::uint64_t scale_offset = (code_bytes + 255ULL) & ~255ULL;
        const std::uint64_t required = scale_offset + groups * 2ULL;
        const auto* payload = static_cast<const std::byte*>(weight.payload);
        if (weight.layout != QuantLayout::RowSplit || weight.group != 64 ||
            weight.group_size != 64 || weight.scale_dtype != DType::FP16 ||
            weight.padded_shape[0] != rows || weight.padded_shape[1] != padded_columns ||
            weight.qhigh != nullptr || weight.high_plane_bytes != 0 || weight.scales == nullptr ||
            weight.payload == nullptr || weight.qdata != weight.payload ||
            static_cast<const std::byte*>(weight.scales) != payload + scale_offset ||
            weight.payload_bytes < required) {
            throw std::invalid_argument(std::string("dflash2: invalid Q4 ") + label);
        }
        return;
    }
    if (weight.qtype == QType::W8G32_F16S) {
        const std::int32_t padded_columns = (columns + 127) & ~127;
        const std::uint64_t groups = static_cast<std::uint64_t>(rows) * padded_columns / 32ULL;
        const std::uint64_t code_bytes = groups * 32ULL;
        const std::uint64_t scale_offset = (code_bytes + 255ULL) & ~255ULL;
        const std::uint64_t required = scale_offset + groups * 2ULL;
        const auto* payload = static_cast<const std::byte*>(weight.payload);
        if (weight.layout != QuantLayout::RowSplit || weight.group != 32 ||
            weight.group_size != 32 || weight.scale_dtype != DType::FP16 ||
            weight.padded_shape[0] != rows || weight.padded_shape[1] != padded_columns ||
            weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
            weight.scales == nullptr || weight.payload == nullptr ||
            weight.qdata != weight.payload ||
            static_cast<const std::byte*>(weight.scales) != payload + scale_offset ||
            weight.payload_bytes < required) {
            throw std::invalid_argument(std::string("dflash2: invalid W8 ") + label);
        }
        return;
    }
    throw std::invalid_argument(std::string("dflash2: unsupported ") + label + " format");
}

} // namespace

void dflash2_fuse_context_copy(const Tensor& learned_drafts,
                               const Tensor& learned_candidates,
                               const Tensor& learned_q_probs,
                               const Tensor& context_tokens,
                               const Tensor& context_match_lengths,
                               const Tensor& context_extents,
                               std::int32_t minimum_match,
                               Tensor& drafts,
                               Tensor& candidates,
                               Tensor& q_probs,
                               cudaStream_t stream) {
    const std::int32_t learned = learned_drafts.ne[0];
    const std::int32_t batch = learned_drafts.ne[1];
    const std::int32_t target = drafts.ne[0];
    const std::int32_t top_k = learned_candidates.ne[0];
    if (learned < 1 || learned > target || target > 15 || batch < 1 || batch > 8 ||
        top_k < 1 || top_k > 16 || minimum_match < 1) {
        throw std::invalid_argument("dflash2_fuse_context_copy: invalid L/K/B/top_k/minimum");
    }
    require_shape(learned_drafts, DType::I32, learned, batch, 1, "learned drafts");
    require_shape(learned_candidates, DType::I32, top_k, learned, batch,
                  "learned candidates");
    require_shape(learned_q_probs, DType::FP32, top_k, learned, batch,
                  "learned probabilities");
    require_shape(context_tokens, DType::I32, target, batch, 1, "context tokens");
    require_vector(context_match_lengths, DType::I32, batch, "context match lengths");
    require_vector(context_extents, DType::I32, batch, "context extents");
    require_shape(drafts, DType::I32, target, batch, 1, "fused drafts");
    require_shape(candidates, DType::I32, top_k, target, batch, "fused candidates");
    require_shape(q_probs, DType::FP32, top_k, target, batch, "fused probabilities");

    const std::array<const Tensor*, 9> operands = {
        &learned_drafts, &learned_candidates, &learned_q_probs, &context_tokens,
        &context_match_lengths, &context_extents, &drafts, &candidates, &q_probs};
    for (std::size_t lhs = 0; lhs < operands.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < operands.size(); ++rhs) {
            if (overlaps(*operands[lhs], *operands[rhs])) {
                throw std::invalid_argument(
                    "dflash2_fuse_context_copy: tensor storage must not overlap");
            }
        }
    }

    detail::dflash2_fuse_context_copy_launch(
        learned_drafts, learned_candidates, learned_q_probs, context_tokens,
        context_match_lengths, context_extents, minimum_match, drafts, candidates, q_probs,
        stream);
}

std::size_t dflash2_accept_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch) {
    if (token_domain <= 0 || min_drafts < 1 || max_drafts < min_drafts ||
        max_drafts > kDFlash2AcceptMaxDrafts || min_batch < 1 || max_batch < min_batch ||
        max_batch > kDFlash2AcceptMaxBatch) {
        throw std::invalid_argument("dflash2 accept workspace: invalid profile or interval");
    }
    const std::int32_t columns = (max_drafts + 1) * max_batch;
    if (!dflash2_accept_hierarchy_ok(token_domain, columns)) { return 0; }
    const std::int32_t partial_stride = dflash2_accept_partial_stride(token_domain);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I64, {kSamplerWideCandidates, partial_stride, columns});
    (void)layout.alloc(DType::I32, {columns});
    (void)layout.alloc(DType::I32, {kSamplerWideCandidates, columns});
    (void)layout.alloc(DType::FP32, {kSamplerWideCandidates, columns});
    (void)layout.alloc(DType::I32, {columns});
    return layout.peak_bytes(256);
}

std::size_t dflash2_select_path_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t rank, std::int32_t top_k,
    std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch) {
    if (token_domain <= 0 || rank <= 0 || rank > 256 || top_k <= 0 ||
        top_k > kDFlash2SelectorMaximumTopK || top_k > token_domain || min_drafts <= 0 ||
        max_drafts < min_drafts || max_drafts > kDFlash2SelectorMaximumDrafts ||
        min_batch <= 0 || max_batch < min_batch ||
        max_batch > kDFlash2SelectorMaximumBatch) {
        throw std::invalid_argument("dflash2 selector workspace: invalid profile or interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {top_k, max_drafts, max_batch});
    (void)layout.alloc(DType::BF16, {rank, max_drafts, max_batch});
    const std::int32_t columns = max_drafts * max_batch;
    if (dflash2_selector_hierarchy_ok(token_domain, top_k, columns)) {
        const std::int32_t partial_stride = dflash2_selector_partial_stride(token_domain);
        (void)layout.alloc(DType::I64, {top_k, partial_stride, columns});
        (void)layout.alloc(DType::I32, {columns});
    }
    return layout.peak_bytes(256);
}

void dflash2_select_path(const Tensor& logits, const Tensor& hidden, const Weight& proj,
                         const Weight& predecessor_codebook, const Weight& successor_codebook,
                         const Tensor& anchors, const SamplingConfig* configs,
                         const Tensor& lengths, const Tensor& current_extents,
                         std::int32_t token_domain, std::int32_t top_k,
                         WorkspaceArena& workspace, Tensor& drafts, Tensor& candidates,
                         Tensor& q_probs, cudaStream_t stream) {
    if (configs == nullptr) {
        throw std::invalid_argument("dflash2_select_path: configs must be non-null");
    }
    require_contiguous(logits, DType::BF16, "selector logits");
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t k     = logits.ne[1];
    const std::int32_t batch = logits.ne[2];
    if (physical_rows <= 0 || token_domain <= 0 || token_domain > physical_rows || k <= 0 ||
        k > kDFlash2SelectorMaximumDrafts || batch <= 0 ||
        batch > kDFlash2SelectorMaximumBatch || logits.ne[3] != 1 || top_k <= 0 ||
        top_k > kDFlash2SelectorMaximumTopK || top_k > token_domain) {
        throw std::invalid_argument("dflash2_select_path: invalid V/K/B/top_k");
    }
    const std::int32_t h = hidden.ne[0];
    require_shape(hidden, DType::BF16, h, k, batch, "selector hidden");
    const std::int32_t rank = proj.n;
    if (h <= 0 || rank <= 0 || rank > 256) {
        throw std::invalid_argument("dflash2_select_path: invalid projection shape");
    }
    require_selector_weight(proj, rank, h, "selector projection");
    require_selector_weight(predecessor_codebook, physical_rows, rank, "predecessor codebook");
    require_selector_weight(successor_codebook, physical_rows, rank, "successor codebook");
    if (predecessor_codebook.qtype != proj.qtype || successor_codebook.qtype != proj.qtype) {
        throw std::invalid_argument("dflash2_select_path: selector weights must share one profile");
    }
    require_vector(anchors, DType::I32, batch, "selector anchors");
    require_vector(lengths, DType::I32, batch, "selector lengths");
    require_vector(current_extents, DType::I32, batch, "selector extents");
    require_shape(drafts, DType::I32, k, batch, 1, "selector drafts");
    require_shape(candidates, DType::I32, top_k, k, batch, "selector candidates");
    require_shape(q_probs, DType::FP32, top_k, k, batch, "selector probabilities");

    auto scratch_scope = workspace.scope();
    Tensor unary_scratch = workspace.alloc(DType::FP32, {top_k, k, batch});
    Tensor hidden_scratch = workspace.alloc(DType::BF16, {rank, k, batch});
    Tensor partial_keys;
    Tensor group_done;
    const std::int32_t columns = k * batch;
    if (dflash2_selector_hierarchy_ok(token_domain, top_k, columns)) {
        const std::int32_t partial_stride = dflash2_selector_partial_stride(token_domain);
        partial_keys = workspace.alloc(DType::I64, {top_k, partial_stride, columns});
        group_done   = workspace.alloc(DType::I32, {columns});
    }
    detail::dflash2_select_path_launch(
        logits, hidden, proj, predecessor_codebook, successor_codebook, anchors, configs, lengths,
        current_extents, token_domain, top_k, unary_scratch, hidden_scratch, drafts, candidates,
        q_probs, partial_keys, group_done, stream);
}

void dflash2_accept(const Tensor& logits, const Tensor& drafts, const Tensor& candidates,
                    const Tensor& q_probs, const Tensor& current_extents,
                    const SamplingConfig* configs, std::int32_t token_domain, Tensor& lengths,
                    Tensor& anchors, Tensor& licensed, Tensor& licensed_counts, Tensor& accepted,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (configs == nullptr) {
        throw std::invalid_argument("dflash2_accept: configs must be non-null");
    }
    require_contiguous(logits, DType::BF16, "target logits");
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t k             = drafts.ne[0];
    const std::int32_t batch         = drafts.ne[1];
    if (k < 1 || k > 15 || batch < 1 || batch > 8 || logits.ne[1] != k + 1 ||
        logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument("dflash2_accept: invalid K/B/logit shape");
    }
    require_shape(drafts, DType::I32, k, batch, 1, "drafts");
    const std::int32_t selector_top_k = candidates.ne[0];
    if (selector_top_k < 1 || selector_top_k > 16) {
        throw std::invalid_argument("dflash2_accept: invalid selector top_k");
    }
    require_shape(candidates, DType::I32, selector_top_k, k, batch, "candidates");
    require_shape(q_probs, DType::FP32, selector_top_k, k, batch, "q probabilities");
    require_vector(current_extents, DType::I32, batch, "current extents");
    require_vector(lengths, DType::I32, batch, "lengths");
    require_vector(anchors, DType::I32, batch, "anchors");
    require_shape(licensed, DType::I32, k + 1, batch, 1, "licensed tokens");
    require_vector(licensed_counts, DType::I32, batch, "licensed counts");
    require_vector(accepted, DType::I32, batch, "accepted counts");
    if (token_domain < 1 || token_domain > physical_rows) {
        throw std::invalid_argument("dflash2_accept: invalid token domain");
    }

    auto scratch_scope = workspace.scope();
    Tensor partial_keys;
    Tensor group_done;
    Tensor dist_indices;
    Tensor dist_probabilities;
    Tensor dist_support;
    const std::int32_t columns = (k + 1) * batch;
    if (dflash2_accept_hierarchy_ok(token_domain, columns)) {
        const std::int32_t partial_stride = dflash2_accept_partial_stride(token_domain);
        partial_keys = workspace.alloc(
            DType::I64, {kSamplerWideCandidates, partial_stride, columns});
        group_done = workspace.alloc(DType::I32, {columns});
        dist_indices = workspace.alloc(DType::I32, {kSamplerWideCandidates, columns});
        dist_probabilities = workspace.alloc(DType::FP32, {kSamplerWideCandidates, columns});
        dist_support = workspace.alloc(DType::I32, {columns});
    }
    detail::dflash2_accept_launch(logits, drafts, candidates, q_probs, current_extents, configs,
                                  token_domain, lengths, anchors, licensed, licensed_counts,
                                  accepted, partial_keys, group_done, dist_indices,
                                  dist_probabilities, dist_support, stream);
}

} // namespace ginfer::ops
