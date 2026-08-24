#pragma once

#include "ginfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace ginfer::targets::qwen3_6::detail {

inline constexpr std::uint32_t kContextCopyMinimumMatch = 6;
inline constexpr std::uint32_t kContextCopyMaximumMatch = 12;
inline constexpr std::uint32_t kContextCopyMaximumDrafts = 15;

struct ContextCopyMatch {
    std::array<TokenId, kContextCopyMaximumDrafts> tokens{};
    std::uint32_t match_length = 0;
    std::uint32_t valid_tokens = 0;

    [[nodiscard]] bool available() const noexcept { return valid_tokens != 0; }
};

[[nodiscard]] inline bool use_wide_context_copy(
    std::span<const ContextCopyMatch> matches,
    std::span<const std::uint32_t> confirmations,
    std::uint32_t learned_drafts,
    std::uint32_t target_drafts,
    bool requires_wide_append) {
    if (matches.empty() || confirmations.size() != matches.size() || learned_drafts == 0 ||
        target_drafts <= learned_drafts) {
        return false;
    }
    if (requires_wide_append) { return true; }
    for (std::size_t row = 0; row < matches.size(); ++row) {
        if (confirmations[row] < 2U || matches[row].valid_tokens <= learned_drafts) {
            return false;
        }
    }
    return true;
}

// Exact request-local suffix lookup for lossless speculative proposals. The index stores the
// ends of six-token windows. Hash collisions are resolved by comparing represented TokenIds, and
// candidates are extended backwards through twelve tokens before choosing longest-match then
// most-recent. The continuation may overlap the current suffix, which preserves periodic copies.
class ContextCopyIndex {
public:
    void clear() noexcept {
        ends_.clear();
        indexed_tokens_ = 0;
    }

    void rebuild(std::span<const TokenId> tokens) {
        clear();
        ends_.reserve(tokens.size());
        append(tokens);
    }

    void append(std::span<const TokenId> tokens) {
        if (tokens.size() < indexed_tokens_) {
            rebuild(tokens);
            return;
        }
        const std::size_t first_end =
            std::max<std::size_t>(indexed_tokens_, kContextCopyMinimumMatch - 1U);
        for (std::size_t end = first_end; end < tokens.size(); ++end) {
            ends_[window_hash(tokens, end)].push_back(static_cast<std::uint32_t>(end));
        }
        indexed_tokens_ = tokens.size();
    }

    [[nodiscard]] ContextCopyMatch find(std::span<const TokenId> tokens,
                                        std::uint32_t maximum_drafts) const {
        ContextCopyMatch out;
        maximum_drafts = std::min(maximum_drafts, kContextCopyMaximumDrafts);
        if (maximum_drafts == 0 || tokens.size() <= kContextCopyMinimumMatch) { return out; }

        const std::size_t suffix_end = tokens.size() - 1U;
        const auto found             = ends_.find(window_hash(tokens, suffix_end));
        if (found == ends_.end()) { return out; }

        std::uint32_t best_length = 0;
        std::uint32_t best_end    = 0;
        for (auto it = found->second.rbegin(); it != found->second.rend(); ++it) {
            const std::uint32_t candidate_end = *it;
            if (candidate_end >= suffix_end ||
                !same_window(tokens, candidate_end, suffix_end)) {
                continue;
            }
            std::uint32_t length = kContextCopyMinimumMatch;
            while (length < kContextCopyMaximumMatch && candidate_end >= length &&
                   suffix_end >= length &&
                   tokens[candidate_end - length] == tokens[suffix_end - length]) {
                ++length;
            }
            if (length > best_length || (length == best_length && candidate_end > best_end)) {
                best_length = length;
                best_end    = candidate_end;
            }
        }
        if (best_length == 0) { return out; }

        out.match_length = best_length;
        out.valid_tokens = std::min<std::uint32_t>(
            maximum_drafts, static_cast<std::uint32_t>(suffix_end - best_end));
        for (std::uint32_t i = 0; i < out.valid_tokens; ++i) {
            out.tokens[i] = tokens[static_cast<std::size_t>(best_end) + 1U + i];
        }
        return out;
    }

private:
    [[nodiscard]] static std::uint64_t mix(std::uint64_t hash, TokenId token) noexcept {
        hash ^= static_cast<std::uint32_t>(token);
        hash *= 1099511628211ULL;
        return hash;
    }

    [[nodiscard]] static std::uint64_t window_hash(std::span<const TokenId> tokens,
                                                   std::size_t end) noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        const std::size_t begin = end + 1U - kContextCopyMinimumMatch;
        for (std::size_t i = begin; i <= end; ++i) { hash = mix(hash, tokens[i]); }
        return hash;
    }

    [[nodiscard]] static bool same_window(std::span<const TokenId> tokens,
                                          std::size_t left_end,
                                          std::size_t right_end) noexcept {
        for (std::size_t i = 0; i < kContextCopyMinimumMatch; ++i) {
            if (tokens[left_end - i] != tokens[right_end - i]) { return false; }
        }
        return true;
    }

    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> ends_;
    std::size_t indexed_tokens_ = 0;
};

} // namespace ginfer::targets::qwen3_6::detail
