#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ginfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ginfer::ops {

/**
 * DFlash2 candidate path selector.
 *
 * logits: BF16 [physical_rows, K, B], with token_domain logical vocabulary rows
 * hidden: BF16 [H, K, B]
 * proj: BF16_CTRL, Q4G64_F16S, or W8G32_F16S Weight [rank, H]
 * predecessor/successor codebooks: same profile as proj, [vocab, rank]
 * anchors/lengths/current_extents: device I32 [B]
 * configs: device SamplingConfig[B]
 * drafts: I32 [K,B] chosen path
 * candidates: I32 [top_k,K,B] vocab ids from the unary top-k
 * q_probs: FP32 [top_k,K,B] selector distribution over those candidates
 *
 * Row b produces current_extents[b] sequential drafts and zeroes its remaining draft/q tail.
 * configs[b].temperature<=0 selects an argmax path; positive temperature applies softmax to the
 * selector scores and inverse-CDF sampling (temperature affects selector scores, not vocab
 * logits). RNG key is (configs[b].seed,lengths[b]+1+t,kSamplePurposeDFlashSelect). Inputs are
 * unchanged and all outputs are overwritten. The Op uses only caller-owned transient storage.
 */
void dflash2_select_path(const Tensor& logits, const Tensor& hidden, const Weight& proj,
                         const Weight& predecessor_codebook, const Weight& successor_codebook,
                         const Tensor& anchors, const SamplingConfig* configs,
                         const Tensor& lengths, const Tensor& current_extents,
                         std::int32_t token_domain, std::int32_t top_k,
                         WorkspaceArena& workspace, Tensor& drafts, Tensor& candidates,
                         Tensor& q_probs, cudaStream_t stream);

/**
 * Lossless context-copy composition for a DFlash2 proposal.
 *
 * learned_drafts: I32 [L,B]
 * learned_candidates/learned_q_probs: I32/FP32 [top_k,L,B]
 * context_tokens: I32 [K,B]
 * context_match_lengths/context_extents: I32 [B]
 * drafts: I32 [K,B]
 * candidates/q_probs: I32/FP32 [top_k,K,B]
 *
 * A row whose context_match_lengths[b] is at least minimum_match uses its first
 * context_extents[b] context tokens as point-mass proposals: candidate rank zero is the context
 * token with probability one and every other rank is zero. Otherwise, its first L columns are
 * copied exactly from the learned proposal. Every remaining output value is zero. L is in [1,K],
 * K is in [1,15], B is in [1,8], top_k is in [1,16], and context_extents are in [0,K]. Inputs are
 * unchanged, outputs are fully overwritten, and all input/output storage is mutually
 * non-overlapping. The point-mass q distribution makes ordinary rejection sampling lossless.
 */
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
                               cudaStream_t stream);

/**
 * Caller-owned selector scratch for every K/B pair in the inclusive domains. token_domain is
 * the logical logits vocabulary; registered large domains use the exact hierarchical top-k route.
 */
[[nodiscard]] std::size_t dflash2_select_path_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t rank, std::int32_t top_k,
    std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

/**
 * DFlash2 sampled rejection against target verify logits.
 *
 * logits: BF16 [vocab,K+1,B], drafts: I32 [K,B], candidates/q_probs: [top_k,K,B],
 * current_extents/lengths/anchors/licensed_counts/accepted: I32 [B], and licensed is
 * I32 [K+1,B]. Row b consumes only current_extents[b] drafts; its target bonus column is therefore
 * column current_extents[b], not necessarily physical column K.
 *
 * For a positive-temperature row, accepts draft i when u*q_i < p_i(draft_i) under the ordinary
 * Muse target distribution: target top_k in [1,63] keeps that many candidates, while top_k<=0 or
 * top_k>=64 keeps min(64,token_domain). On first rejection it samples from normalized
 * clamp(p-scatter(q),0); when every draft accepts it samples the row's bonus column. For a
 * nonpositive-temperature row it accepts the longest exact target-argmax prefix and emits the
 * first mismatch or bonus. Greedy and stochastic rows may coexist in one batch.
 *
 * The Op overwrites every licensed slot (zeroing the unused tail), updates lengths/anchors/counts,
 * and increments each stochastic row's non-null token_counts for all licensed tokens. Greedy rows
 * do not update token_counts, matching sample(). Other inputs are unchanged. Large token domains
 * use caller-owned transient storage to construct every exact target top-64 distribution in
 * parallel; the state transition remains sequential within each row.
 */
void dflash2_accept(const Tensor& logits, const Tensor& drafts, const Tensor& candidates,
                    const Tensor& q_probs, const Tensor& current_extents,
                    const SamplingConfig* configs, std::int32_t token_domain, Tensor& lengths,
                    Tensor& anchors, Tensor& licensed, Tensor& licensed_counts, Tensor& accepted,
                    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Caller-owned acceptance scratch for every K/B pair in the inclusive domains. K is the physical
 * draft extent, so each row has K+1 target verification columns. The capacity is zero only when
 * every shape in the interval uses the bounded single-CTA route (currently token domains <=256 or
 * domains too large for the closed exact hierarchy). The registered V=202048, K<=15, B<=8 route
 * is hierarchical and fully represented by this query.
 */
[[nodiscard]] std::size_t dflash2_accept_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

} // namespace ginfer::ops
