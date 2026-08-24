#pragma once

#include "core/tensor.h"
#include "ginfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ginfer::ops::detail {

void dflash2_fuse_context_copy_launch(const Tensor& learned_drafts,
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

void dflash2_select_path_launch(const Tensor& logits, const Tensor& hidden, const Weight& proj,
                                const Weight& predecessor_codebook,
                                const Weight& successor_codebook, const Tensor& anchors,
                                const SamplingConfig* configs, const Tensor& lengths,
                                const Tensor& current_extents, std::int32_t token_domain,
                                std::int32_t top_k,
                                Tensor& unary_scratch, Tensor& hidden_scratch, Tensor& drafts,
                                Tensor& candidates, Tensor& q_probs, const Tensor& partial_keys,
                                const Tensor& group_done, cudaStream_t stream);

void dflash2_accept_launch(const Tensor& logits, const Tensor& drafts, const Tensor& candidates,
                           const Tensor& q_probs, const Tensor& current_extents,
                           const SamplingConfig* configs, std::int32_t token_domain,
                           Tensor& lengths, Tensor& anchors, Tensor& licensed,
                           Tensor& licensed_counts, Tensor& accepted,
                           const Tensor& partial_keys, const Tensor& group_done,
                           const Tensor& dist_indices, const Tensor& dist_probabilities,
                           const Tensor& dist_support, cudaStream_t stream);

} // namespace ginfer::ops::detail
