// Behavioral qualification for Muse DFlash2 sampled rejection.
//
// The expected state transition is computed by a CPU oracle from the BF16 values
// represented at the public input.  The oracle independently constructs the
// documented top-64 Muse distribution and implements the public counter key; it
// does not call another production sampling path or include a private CUDA helper.
#include "ginfer/ops/dflash2_select.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ginfer;
using namespace ginfer::test;

namespace {

constexpr int kMuseTargetCap = 64;

Weight bf16_weight(const GuardedDeviceBuffer& buffer, int rows, int columns) {
    Weight weight{};
    weight.payload          = buffer.data();
    weight.payload_bytes    = static_cast<std::uint64_t>(rows) * columns * 2ULL;
    weight.qtype            = QType::BF16_CTRL;
    weight.layout           = QuantLayout::Contiguous;
    weight.qdata            = buffer.data();
    weight.n                = rows;
    weight.k                = columns;
    weight.ndim             = 2;
    weight.shape[0]         = rows;
    weight.shape[1]         = columns;
    weight.padded_shape[0]  = rows;
    weight.padded_shape[1]  = columns;
    return weight;
}

struct Candidate {
    double logit = 0.0;
    int token    = 0;
};

struct Distribution {
    std::vector<int> tokens;
    std::vector<double> probabilities;
};

struct Expected {
    std::vector<std::int32_t> licensed;
    std::vector<std::int32_t> counts;
    std::int32_t licensed_count = 0;
    std::int32_t accepted       = 0;
    std::int32_t anchor         = 0;
    std::int32_t length         = 0;
};

struct Case {
    std::string label;
    int token_domain = 0;
    int physical_rows = 0;
    int initial_length = 0;
    ops::SamplingConfig config{};
    std::vector<float> logits;
    std::vector<std::int32_t> drafts;
    std::vector<std::int32_t> candidates;
    std::vector<float> q_probs;
    std::vector<std::int32_t> initial_counts;
};

std::uint64_t splitmix64(std::uint64_t x) {
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

// Independent host expression of the counter-based public RNG contract.  Keeping
// the 24-bit numerator available lets the cases cover exact inverse-CDF boundaries.
std::uint32_t uniform_numerator(std::uint64_t seed, std::int32_t position,
                                std::int32_t purpose) {
    std::uint64_t key = splitmix64(
        seed ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                UINT64_C(0xD1B54A32D192ED03)));
    key = splitmix64(
        key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) << 21));
    return static_cast<std::uint32_t>(key >> 40);
}

double uniform(std::uint64_t seed, std::int32_t position, std::int32_t purpose) {
    return static_cast<double>(uniform_numerator(seed, position, purpose)) / 16777216.0;
}

Distribution target_distribution(const Case& c, int column, int overlay_len) {
    std::vector<Candidate> ordered(static_cast<std::size_t>(c.token_domain));
    const std::size_t base = static_cast<std::size_t>(column) * c.physical_rows;
    for (int token = 0; token < c.token_domain; ++token) {
        int count = c.initial_counts[static_cast<std::size_t>(token)];
        for (int j = 0; j < overlay_len; ++j) {
            if (c.drafts[static_cast<std::size_t>(j)] == token) ++count;
        }
        double adjusted = c.logits[base + static_cast<std::size_t>(token)];
        if (count > 0) adjusted -= c.config.presence_penalty;
        adjusted -= static_cast<double>(c.config.frequency_penalty) * count;
        ordered[static_cast<std::size_t>(token)] = {adjusted, token};
    }
    std::sort(ordered.begin(), ordered.end(), [](const Candidate& a, const Candidate& b) {
        return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
    });

    int cap = kMuseTargetCap;
    if (c.config.top_k > 0 && c.config.top_k < cap) cap = c.config.top_k;
    cap = std::min(cap, c.token_domain);
    ordered.resize(static_cast<std::size_t>(cap));

    std::vector<double> weight(static_cast<std::size_t>(cap));
    const double maximum = ordered.front().logit / c.config.temperature;
    double total         = 0.0;
    for (int rank = 0; rank < cap; ++rank) {
        weight[static_cast<std::size_t>(rank)] =
            std::exp(ordered[static_cast<std::size_t>(rank)].logit / c.config.temperature -
                     maximum);
        total += weight[static_cast<std::size_t>(rank)];
    }

    const double minimum = c.config.min_p > 0.0f ? c.config.min_p * weight.front() : -1.0;
    const double top_p_target = static_cast<double>(c.config.top_p) * total;
    double cumulative         = 0.0;
    int support               = 0;
    for (int rank = 0; rank < cap; ++rank) {
        if (minimum >= 0.0 && weight[static_cast<std::size_t>(rank)] < minimum) break;
        cumulative += weight[static_cast<std::size_t>(rank)];
        support = rank + 1;
        if (c.config.top_p < 1.0f && cumulative >= top_p_target) break;
    }
    support = std::max(support, 1);

    double kept = 0.0;
    for (int rank = 0; rank < support; ++rank) kept += weight[static_cast<std::size_t>(rank)];
    Distribution distribution;
    distribution.tokens.reserve(static_cast<std::size_t>(support));
    distribution.probabilities.reserve(static_cast<std::size_t>(support));
    for (int rank = 0; rank < support; ++rank) {
        distribution.tokens.push_back(ordered[static_cast<std::size_t>(rank)].token);
        distribution.probabilities.push_back(weight[static_cast<std::size_t>(rank)] / kept);
    }
    return distribution;
}

double probability_of(const Distribution& distribution, int token) {
    for (std::size_t i = 0; i < distribution.tokens.size(); ++i) {
        if (distribution.tokens[i] == token) return distribution.probabilities[i];
    }
    return 0.0;
}

int inverse_cdf(const std::vector<int>& tokens, const std::vector<double>& probabilities,
                double u) {
    double cumulative = 0.0;
    int picked         = tokens.back();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        cumulative += probabilities[i];
        picked = tokens[i];
        if (u < cumulative) break; // The boundary belongs to the following interval.
    }
    return picked;
}

Expected oracle(const Case& c) {
    const int k     = static_cast<int>(c.drafts.size());
    const int width = static_cast<int>(c.candidates.size()) / k;
    Expected expected;
    expected.licensed.assign(static_cast<std::size_t>(k + 1), 0);
    expected.counts = c.initial_counts;

    int accepted = 0;
    int terminal = -1;
    for (int column = 0; column <= k; ++column) {
        const Distribution p = target_distribution(c, column, column);
        const int position   = c.initial_length + column + 1;
        if (column == k) {
            terminal = inverse_cdf(
                p.tokens, p.probabilities,
                uniform(c.config.seed, position, ops::kSamplePurposeSpeculativeBonus));
            break;
        }

        const int draft = c.drafts[static_cast<std::size_t>(column)];
        double q_draft  = 0.0;
        for (int j = 0; j < width; ++j) {
            const std::size_t at = static_cast<std::size_t>(column * width + j);
            if (c.candidates[at] == draft) {
                q_draft = c.q_probs[at];
                break;
            }
        }
        const double accept_u =
            uniform(c.config.seed, position, ops::kSamplePurposeSpeculativeAccept);
        if (accept_u * q_draft < probability_of(p, draft)) {
            ++accepted;
            continue;
        }

        std::vector<double> residual = p.probabilities;
        double residual_mass         = 0.0;
        for (std::size_t rank = 0; rank < p.tokens.size(); ++rank) {
            for (int j = 0; j < width; ++j) {
                const std::size_t at = static_cast<std::size_t>(column * width + j);
                if (p.tokens[rank] == c.candidates[at]) residual[rank] -= c.q_probs[at];
            }
            residual[rank] = std::max(residual[rank], 0.0);
            residual_mass += residual[rank];
        }
        if (residual_mass == 0.0) residual = p.probabilities;
        else
            for (double& value : residual) value /= residual_mass;
        terminal = inverse_cdf(
            p.tokens, residual,
            uniform(c.config.seed, position, ops::kSamplePurposeSpeculativeCorrection));
        break;
    }

    for (int i = 0; i < accepted; ++i) {
        expected.licensed[static_cast<std::size_t>(i)] = c.drafts[static_cast<std::size_t>(i)];
    }
    expected.licensed[static_cast<std::size_t>(accepted)] = terminal;
    expected.licensed_count = accepted + 1;
    expected.accepted       = accepted;
    expected.anchor         = terminal;
    expected.length         = c.initial_length + expected.licensed_count;
    for (int i = 0; i < expected.licensed_count; ++i) {
        ++expected.counts[static_cast<std::size_t>(expected.licensed[static_cast<std::size_t>(i)])];
    }
    return expected;
}

Expected greedy_oracle(const Case& c) {
    const int k = static_cast<int>(c.drafts.size());
    Expected expected;
    expected.licensed.assign(static_cast<std::size_t>(k + 1), 0);
    expected.counts = c.initial_counts;

    int accepted = 0;
    int terminal = 0;
    for (int column = 0; column <= k; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * c.physical_rows;
        int target             = 0;
        for (int token = 1; token < c.token_domain; ++token) {
            const float value = c.logits[base + static_cast<std::size_t>(token)];
            const float best  = c.logits[base + static_cast<std::size_t>(target)];
            if (value > best || (value == best && token < target)) { target = token; }
        }
        if (column < k && target == c.drafts[static_cast<std::size_t>(column)]) {
            ++accepted;
            continue;
        }
        terminal = target;
        break;
    }

    for (int i = 0; i < accepted; ++i) {
        expected.licensed[static_cast<std::size_t>(i)] = c.drafts[static_cast<std::size_t>(i)];
    }
    expected.licensed[static_cast<std::size_t>(accepted)] = terminal;
    expected.licensed_count = accepted + 1;
    expected.accepted       = accepted;
    expected.anchor         = terminal;
    expected.length         = c.initial_length + expected.licensed_count;
    return expected;
}

std::vector<std::uint16_t> bf16_bits(std::vector<float>& values) {
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<float> uniform_top_logits(int rows, int columns, int high_tokens = 80) {
    std::vector<float> logits(static_cast<std::size_t>(rows) * columns, -8.0f);
    for (int column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * rows;
        for (int token = 0; token < std::min(rows, high_tokens); ++token) {
            logits[base + static_cast<std::size_t>(token)] = 0.0f;
        }
    }
    return logits;
}

Case base_case(std::string label, int domain, int k) {
    Case c;
    c.label         = std::move(label);
    c.token_domain  = domain;
    c.physical_rows = domain;
    c.config.temperature = 1.0f;
    c.config.top_k       = 64;
    c.config.top_p       = 1.0f;
    c.logits             = uniform_top_logits(domain, k + 1);
    c.initial_counts.assign(static_cast<std::size_t>(domain), 0);
    return c;
}

int run_case(Case c) {
    round_to_bf16(c.logits);
    const Expected expected = oracle(c);
    const int k              = static_cast<int>(c.drafts.size());
    const int selector_width = static_cast<int>(c.candidates.size()) / k;

    std::vector<std::uint16_t> logits_bits(c.logits.size());
    for (std::size_t i = 0; i < c.logits.size(); ++i) logits_bits[i] = f32_to_bf16(c.logits[i]);
    DeviceBuffer d_logits     = to_device(logits_bits);
    DeviceBuffer d_drafts     = to_device(c.drafts);
    DeviceBuffer d_candidates = to_device(c.candidates);
    DeviceBuffer d_q_probs    = to_device(c.q_probs);
    GuardedDeviceBuffer d_counts(c.initial_counts.size() * sizeof(std::int32_t));
    d_counts.copy_from_host(c.initial_counts.data(), d_counts.bytes());
    c.config.token_counts = static_cast<std::int32_t*>(d_counts.data());
    DeviceBuffer d_config = to_device(std::vector<ops::SamplingConfig>{c.config});

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_extent(sizeof(std::int32_t));
    GuardedDeviceBuffer d_anchor(sizeof(std::int32_t));
    GuardedDeviceBuffer d_licensed(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_licensed_count(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    d_length.copy_from_host(&c.initial_length, sizeof(c.initial_length));
    d_extent.copy_from_host(&k, sizeof(k));
    d_anchor.fill(0xcd);
    d_licensed.fill(0xcd);
    d_licensed_count.fill(0xcd);
    d_accepted.fill(0xcd);

    Tensor logits(d_logits.p, DType::BF16, {c.physical_rows, k + 1, 1});
    Tensor drafts(d_drafts.p, DType::I32, {k, 1});
    Tensor candidates(d_candidates.p, DType::I32, {selector_width, k, 1});
    Tensor q_probs(d_q_probs.p, DType::FP32, {selector_width, k, 1});
    Tensor extent(d_extent.data(), DType::I32, {1});
    Tensor length(d_length.data(), DType::I32, {1});
    Tensor anchor(d_anchor.data(), DType::I32, {1});
    Tensor licensed(d_licensed.data(), DType::I32, {k + 1});
    Tensor licensed_count(d_licensed_count.data(), DType::I32, {1});
    Tensor accepted(d_accepted.data(), DType::I32, {1});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_accept_workspace_capacity_bytes(c.token_domain, k, k, 1, 1)));
    ops::dflash2_accept(logits, drafts, candidates, q_probs, extent,
                        static_cast<const ops::SamplingConfig*>(d_config.p), c.token_domain,
                        length, anchor, licensed, licensed_count, accepted, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact(
        (c.label + " licensed").c_str(),
        from_device<std::int32_t>(d_licensed.data(), static_cast<std::size_t>(k + 1)),
        expected.licensed);
    failures += verify_exact((c.label + " licensed count").c_str(),
                             from_device<std::int32_t>(d_licensed_count.data(), 1),
                             {expected.licensed_count});
    failures += verify_exact((c.label + " accepted").c_str(), from_device<std::int32_t>(d_accepted.data(), 1),
                             {expected.accepted});
    failures += verify_exact((c.label + " anchor").c_str(), from_device<std::int32_t>(d_anchor.data(), 1),
                             {expected.anchor});
    failures += verify_exact((c.label + " length").c_str(), from_device<std::int32_t>(d_length.data(), 1),
                             {expected.length});
    failures += verify_exact((c.label + " token counts").c_str(),
                             from_device<std::int32_t>(d_counts.data(), expected.counts.size()),
                             expected.counts);
    failures += verify_exact((c.label + " logits unchanged").c_str(),
                             from_device<std::uint16_t>(d_logits, logits_bits.size()), logits_bits);
    failures += verify_exact((c.label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, c.drafts.size()), c.drafts);
    failures += verify_exact((c.label + " extent unchanged").c_str(),
                             from_device<std::int32_t>(d_extent.data(), 1), {k});
    failures += verify_exact((c.label + " candidates unchanged").c_str(),
                             from_device<std::int32_t>(d_candidates, c.candidates.size()),
                             c.candidates);
    failures += verify_exact((c.label + " q probabilities unchanged").c_str(),
                             from_device<float>(d_q_probs, c.q_probs.size()), c.q_probs);
    failures += d_counts.verify_guards(c.label + " count guards");
    failures += d_length.verify_guards(c.label + " length guards");
    failures += d_extent.verify_guards(c.label + " extent guards");
    failures += d_anchor.verify_guards(c.label + " anchor guards");
    failures += d_licensed.verify_guards(c.label + " licensed guards");
    failures += d_licensed_count.verify_guards(c.label + " licensed-count guards");
    failures += d_accepted.verify_guards(c.label + " accepted guards");
    return failures;
}

int run_batched_acceptance_case() {
    constexpr int kTokenDomain = 202048;
    constexpr int kPhysicalRows = kTokenDomain;
    constexpr int kDrafts = 4;
    constexpr int kColumns = kDrafts + 1;
    constexpr int kBatch = 2;
    constexpr int kSelectorTopK = 3;
    const std::string label = "DFlash acceptance B=2 mixed modes/extents";

    std::vector<float> logits(static_cast<std::size_t>(kPhysicalRows) * kColumns * kBatch,
                              -6.0f);
    const auto logit = [&](int batch, int column, int token) -> float& {
        return logits[static_cast<std::size_t>(token) +
                      static_cast<std::size_t>(kPhysicalRows) *
                          (column + kColumns * batch)];
    };
    // Greedy row: accept token 2, then reject draft 5 in favor of token 4.
    logit(0, 0, 2) = 5.0f;
    logit(0, 0, 1) = 1.0f;
    logit(0, 1, 4) = 6.0f;
    logit(0, 1, 5) = 2.0f;

    // Stochastic row: each proposal is separated from the rest of the target support, so the
    // independent oracle's three acceptance decisions and extent-local bonus are robust to
    // transcendental implementation details.
    logit(1, 0, 1) = 6.0f;
    logit(1, 0, 2) = 1.0f;
    logit(1, 0, 4) = 0.5f;
    logit(1, 0, 0) = 0.0f;
    logit(1, 1, 3) = 6.0f;
    logit(1, 1, 0) = 1.0f;
    logit(1, 1, 6) = 0.5f;
    logit(1, 1, 2) = 0.0f;
    logit(1, 2, 5) = 6.0f;
    logit(1, 2, 2) = 1.0f;
    logit(1, 2, 7) = 0.5f;
    logit(1, 2, 1) = 0.0f;
    logit(1, 3, 7) = 7.0f; // bonus column is extent 3, not physical column 4
    logit(1, 3, 6) = 1.0f;
    logit(1, 3, 4) = 0.5f;
    logit(1, 3, 0) = 0.0f;
    const std::vector<std::uint16_t> logits_bits = bf16_bits(logits);

    const std::vector<std::int32_t> drafts{
        2, 5, 6, 7, // greedy row
        1, 3, 5, 0, // stochastic row; final physical slot is outside its extent
    };
    std::vector<std::int32_t> candidates(
        static_cast<std::size_t>(kSelectorTopK) * kDrafts * kBatch);
    std::vector<float> q_probs(candidates.size(), 0.0f);
    const auto set_proposal = [&](int batch, int position, std::int32_t a, std::int32_t b,
                                  std::int32_t c) {
        const std::size_t base = static_cast<std::size_t>(kSelectorTopK) *
                                 (position + kDrafts * batch);
        candidates[base]     = a;
        candidates[base + 1] = b;
        candidates[base + 2] = c;
        q_probs[base]        = 0.5f;
        q_probs[base + 1]    = 0.3f;
        q_probs[base + 2]    = 0.2f;
    };
    set_proposal(0, 0, 2, 1, 0);
    set_proposal(0, 1, 5, 4, 3);
    set_proposal(0, 2, 6, 5, 4);
    set_proposal(0, 3, 7, 6, 5);
    set_proposal(1, 0, 1, 2, 4);
    set_proposal(1, 1, 3, 0, 6);
    set_proposal(1, 2, 5, 2, 7);
    set_proposal(1, 3, 0, 1, 2);

    const std::vector<std::int32_t> extents{2, 3};
    const std::vector<std::int32_t> initial_lengths{20, 40};
    const std::vector<std::int32_t> initial_anchors{7, 6};
    std::vector<std::int32_t> initial_counts0(kTokenDomain, 0);
    std::vector<std::int32_t> initial_counts1(kTokenDomain, 0);
    initial_counts0[2] = 3;
    initial_counts0[4] = 1;
    initial_counts1[0] = 2;
    initial_counts1[7] = 1;

    GuardedDeviceBuffer d_token_counts0(initial_counts0.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_token_counts1(initial_counts1.size() * sizeof(std::int32_t));
    d_token_counts0.copy_from_host(initial_counts0.data(), d_token_counts0.bytes());
    d_token_counts1.copy_from_host(initial_counts1.data(), d_token_counts1.bytes());

    std::vector<ops::SamplingConfig> configs(kBatch);
    configs[0].temperature       = 0.0f;
    configs[0].top_k            = 4;
    configs[0].presence_penalty = 100.0f; // Greedy must ignore penalties and occurrence state.
    configs[0].seed             = 111;
    configs[0].token_counts = static_cast<std::int32_t*>(d_token_counts0.data());
    configs[1].temperature       = 1.0f;
    configs[1].top_k            = 4;
    configs[1].top_p            = 1.0f;
    configs[1].presence_penalty = 0.25f;
    configs[1].frequency_penalty = 0.125f;
    configs[1].seed              = 90210;
    configs[1].token_counts = static_cast<std::int32_t*>(d_token_counts1.data());

    const auto extract_row = [&](int batch, int extent,
                                 const std::vector<std::int32_t>& initial_counts) {
        Case row;
        row.label          = label + " row " + std::to_string(batch);
        row.token_domain   = kTokenDomain;
        row.physical_rows  = kPhysicalRows;
        row.initial_length = initial_lengths[static_cast<std::size_t>(batch)];
        row.config         = configs[static_cast<std::size_t>(batch)];
        row.initial_counts = initial_counts;
        row.logits.resize(static_cast<std::size_t>(kPhysicalRows) * (extent + 1));
        for (int column = 0; column <= extent; ++column) {
            const std::size_t source = static_cast<std::size_t>(kPhysicalRows) *
                                       (column + kColumns * batch);
            std::copy_n(logits.begin() + static_cast<std::ptrdiff_t>(source), kPhysicalRows,
                        row.logits.begin() + static_cast<std::ptrdiff_t>(column * kPhysicalRows));
        }
        const std::size_t draft_base = static_cast<std::size_t>(batch) * kDrafts;
        row.drafts.assign(drafts.begin() + static_cast<std::ptrdiff_t>(draft_base),
                          drafts.begin() + static_cast<std::ptrdiff_t>(draft_base + extent));
        row.candidates.resize(static_cast<std::size_t>(kSelectorTopK) * extent);
        row.q_probs.resize(row.candidates.size());
        const std::size_t proposal_base =
            static_cast<std::size_t>(batch) * kDrafts * kSelectorTopK;
        std::copy_n(candidates.begin() + static_cast<std::ptrdiff_t>(proposal_base),
                    row.candidates.size(), row.candidates.begin());
        std::copy_n(q_probs.begin() + static_cast<std::ptrdiff_t>(proposal_base),
                    row.q_probs.size(), row.q_probs.begin());
        return row;
    };
    const Expected expected0 = greedy_oracle(extract_row(0, extents[0], initial_counts0));
    const Expected expected1 = oracle(extract_row(1, extents[1], initial_counts1));

    std::vector<std::int32_t> expected_licensed(static_cast<std::size_t>(kColumns) * kBatch, 0);
    std::copy(expected0.licensed.begin(), expected0.licensed.end(), expected_licensed.begin());
    std::copy(expected1.licensed.begin(), expected1.licensed.end(),
              expected_licensed.begin() + kColumns);
    const std::vector<std::int32_t> expected_licensed_counts{expected0.licensed_count,
                                                             expected1.licensed_count};
    const std::vector<std::int32_t> expected_accepted{expected0.accepted, expected1.accepted};
    const std::vector<std::int32_t> expected_anchors{expected0.anchor, expected1.anchor};
    const std::vector<std::int32_t> expected_lengths{expected0.length, expected1.length};

    GuardedDeviceBuffer d_logits(logits_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_drafts(drafts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_candidates(candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_q_probs(q_probs.size() * sizeof(float));
    GuardedDeviceBuffer d_extents(extents.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_configs(configs.size() * sizeof(ops::SamplingConfig));
    GuardedDeviceBuffer d_lengths(initial_lengths.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_anchors(initial_anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_licensed(expected_licensed.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_licensed_counts(kBatch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(kBatch * sizeof(std::int32_t));
    d_logits.copy_from_host(logits_bits.data(), d_logits.bytes());
    d_drafts.copy_from_host(drafts.data(), d_drafts.bytes());
    d_candidates.copy_from_host(candidates.data(), d_candidates.bytes());
    d_q_probs.copy_from_host(q_probs.data(), d_q_probs.bytes());
    d_extents.copy_from_host(extents.data(), d_extents.bytes());
    d_configs.copy_from_host(configs.data(), d_configs.bytes());
    d_lengths.copy_from_host(initial_lengths.data(), d_lengths.bytes());
    d_anchors.copy_from_host(initial_anchors.data(), d_anchors.bytes());
    d_licensed.fill(0xcd);
    d_licensed_counts.fill(0xcd);
    d_accepted.fill(0xcd);
    std::vector<std::uint8_t> config_bytes(d_configs.bytes());
    std::memcpy(config_bytes.data(), configs.data(), config_bytes.size());

    Tensor logits_tensor(d_logits.data(), DType::BF16, {kPhysicalRows, kColumns, kBatch});
    Tensor drafts_tensor(d_drafts.data(), DType::I32, {kDrafts, kBatch});
    Tensor candidates_tensor(d_candidates.data(), DType::I32,
                             {kSelectorTopK, kDrafts, kBatch});
    Tensor q_probs_tensor(d_q_probs.data(), DType::FP32,
                          {kSelectorTopK, kDrafts, kBatch});
    Tensor extents_tensor(d_extents.data(), DType::I32, {kBatch});
    Tensor lengths_tensor(d_lengths.data(), DType::I32, {kBatch});
    Tensor anchors_tensor(d_anchors.data(), DType::I32, {kBatch});
    Tensor licensed_tensor(d_licensed.data(), DType::I32, {kColumns, kBatch});
    Tensor licensed_counts_tensor(d_licensed_counts.data(), DType::I32, {kBatch});
    Tensor accepted_tensor(d_accepted.data(), DType::I32, {kBatch});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_accept_workspace_capacity_bytes(
                 kTokenDomain, kDrafts, kDrafts, kBatch, kBatch)));
    ops::dflash2_accept(logits_tensor, drafts_tensor, candidates_tensor, q_probs_tensor,
                        extents_tensor,
                        static_cast<const ops::SamplingConfig*>(d_configs.data()), kTokenDomain,
                        lengths_tensor, anchors_tensor, licensed_tensor, licensed_counts_tensor,
                        accepted_tensor, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact((label + " licensed and zero tail").c_str(),
                                from_device<std::int32_t>(d_licensed.data(),
                                                          expected_licensed.size()),
                                expected_licensed);
    failures += verify_exact((label + " licensed counts").c_str(),
                             from_device<std::int32_t>(d_licensed_counts.data(), kBatch),
                             expected_licensed_counts);
    failures += verify_exact((label + " accepted counts").c_str(),
                             from_device<std::int32_t>(d_accepted.data(), kBatch),
                             expected_accepted);
    failures += verify_exact((label + " anchors").c_str(),
                             from_device<std::int32_t>(d_anchors.data(), kBatch),
                             expected_anchors);
    failures += verify_exact((label + " lengths").c_str(),
                             from_device<std::int32_t>(d_lengths.data(), kBatch),
                             expected_lengths);
    failures += verify_exact((label + " greedy token counts").c_str(),
                             from_device<std::int32_t>(d_token_counts0.data(), kTokenDomain),
                             expected0.counts);
    failures += verify_exact((label + " stochastic token counts").c_str(),
                             from_device<std::int32_t>(d_token_counts1.data(), kTokenDomain),
                             expected1.counts);

    failures += verify_exact((label + " logits immutable").c_str(),
                             from_device<std::uint16_t>(d_logits.data(), logits_bits.size()),
                             logits_bits);
    failures += verify_exact((label + " drafts immutable").c_str(),
                             from_device<std::int32_t>(d_drafts.data(), drafts.size()), drafts);
    failures += verify_exact((label + " candidates immutable").c_str(),
                             from_device<std::int32_t>(d_candidates.data(), candidates.size()),
                             candidates);
    failures += verify_exact((label + " q probabilities immutable").c_str(),
                             from_device<float>(d_q_probs.data(), q_probs.size()), q_probs);
    failures += verify_exact((label + " extents immutable").c_str(),
                             from_device<std::int32_t>(d_extents.data(), extents.size()), extents);
    failures += verify_exact((label + " configs immutable").c_str(),
                             from_device<std::uint8_t>(d_configs.data(), config_bytes.size()),
                             config_bytes);

    failures += d_logits.verify_guards(label + " logits guards");
    failures += d_drafts.verify_guards(label + " drafts guards");
    failures += d_candidates.verify_guards(label + " candidates guards");
    failures += d_q_probs.verify_guards(label + " q-probability guards");
    failures += d_extents.verify_guards(label + " extent guards");
    failures += d_configs.verify_guards(label + " config guards");
    failures += d_lengths.verify_guards(label + " length guards");
    failures += d_anchors.verify_guards(label + " anchor guards");
    failures += d_licensed.verify_guards(label + " licensed guards");
    failures += d_licensed_counts.verify_guards(label + " licensed-count guards");
    failures += d_accepted.verify_guards(label + " accepted guards");
    failures += d_token_counts0.verify_guards(label + " greedy token-count guards");
    failures += d_token_counts1.verify_guards(label + " stochastic token-count guards");
    return failures;
}

int run_batched_selector_case() {
    constexpr int kVocab = 6;
    constexpr int kHidden = 2;
    constexpr int kRank = 2;
    constexpr int kDrafts = 3;
    constexpr int kBatch = 2;
    constexpr int kTopK = 3;
    const std::string label = "DFlash selector B=2 extent-zero/tie ordering";

    std::vector<float> logits(static_cast<std::size_t>(kVocab) * kDrafts * kBatch, -5.0f);
    const auto logit = [&](int batch, int position, int token) -> float& {
        return logits[static_cast<std::size_t>(token) +
                      static_cast<std::size_t>(kVocab) * (position + kDrafts * batch)];
    };
    logit(0, 0, 0) = 2.0f;
    logit(0, 0, 2) = 2.0f;
    logit(0, 0, 4) = 1.0f;
    logit(0, 1, 1) = 3.0f;
    logit(0, 1, 3) = 3.0f;
    logit(0, 1, 5) = 2.0f;
    logit(0, 2, 0) = 4.0f;
    logit(0, 2, 2) = 4.0f;
    logit(0, 2, 4) = 1.0f;
    logit(1, 0, 4) = 6.0f;
    logit(1, 0, 5) = 6.0f;
    logit(1, 0, 1) = 2.0f;
    logit(1, 1, 2) = 7.0f;
    logit(1, 1, 3) = 7.0f;
    logit(1, 1, 0) = 1.0f;
    logit(1, 2, 0) = 8.0f;
    logit(1, 2, 1) = 8.0f;
    logit(1, 2, 5) = 3.0f;

    // All numerical reference inputs are first represented as BF16, exactly as at the public Op
    // boundary. Identity projection makes the mathematical host projection exact without copying
    // the production kernel's staging arithmetic.
    std::vector<float> hidden{
        1.0f, 0.5f, 0.5f, 1.0f, 2.0f, -1.0f,
        -4.0f, 3.0f, 5.0f, -2.0f, -3.0f, -6.0f,
    };
    std::vector<float> projection{1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> predecessor(static_cast<std::size_t>(kVocab) * kRank);
    std::vector<float> successor(static_cast<std::size_t>(kVocab) * kRank);
    for (int token = 0; token < kVocab; ++token) {
        predecessor[static_cast<std::size_t>(token) * kRank] =
            0.25f * static_cast<float>(token + 1);
        predecessor[static_cast<std::size_t>(token) * kRank + 1] =
            0.125f * static_cast<float>(kVocab - token);
        successor[static_cast<std::size_t>(token) * kRank] =
            0.25f * static_cast<float>((token % 3) - 1);
        successor[static_cast<std::size_t>(token) * kRank + 1] =
            token % 2 == 0 ? -0.5f : 0.5f;
    }
    const std::vector<std::uint16_t> logits_bits = bf16_bits(logits);
    const std::vector<std::uint16_t> hidden_bits = bf16_bits(hidden);
    const std::vector<std::uint16_t> projection_bits = bf16_bits(projection);
    const std::vector<std::uint16_t> predecessor_bits = bf16_bits(predecessor);
    const std::vector<std::uint16_t> successor_bits = bf16_bits(successor);

    const std::vector<std::int32_t> anchors{0, 5};
    const std::vector<std::int32_t> lengths{100, 900};
    const std::vector<std::int32_t> extents{2, 0};
    std::vector<ops::SamplingConfig> configs(kBatch);
    configs[0].temperature = 0.75f;
    configs[0].seed        = 17;
    configs[1].temperature = 0.0f;
    configs[1].seed        = 999;

    std::vector<std::int32_t> expected_candidates(
        static_cast<std::size_t>(kTopK) * kDrafts * kBatch);
    std::vector<std::int32_t> expected_drafts(static_cast<std::size_t>(kDrafts) * kBatch, 0);
    std::vector<double> expected_q(static_cast<std::size_t>(kTopK) * kDrafts * kBatch, 0.0);

    for (int batch = 0; batch < kBatch; ++batch) {
        for (int position = 0; position < kDrafts; ++position) {
            std::vector<Candidate> ordered(kVocab);
            for (int token = 0; token < kVocab; ++token) {
                ordered[static_cast<std::size_t>(token)] = {
                    static_cast<double>(logit(batch, position, token)), token};
            }
            std::sort(ordered.begin(), ordered.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
                      });
            const std::size_t base = static_cast<std::size_t>(kTopK) *
                                     (position + kDrafts * batch);
            for (int rank = 0; rank < kTopK; ++rank) {
                expected_candidates[base + static_cast<std::size_t>(rank)] =
                    ordered[static_cast<std::size_t>(rank)].token;
            }
        }
    }

    for (int batch = 0; batch < kBatch; ++batch) {
        int previous = anchors[static_cast<std::size_t>(batch)];
        for (int position = 0; position < extents[static_cast<std::size_t>(batch)]; ++position) {
            double projected[kRank]{};
            for (int rank = 0; rank < kRank; ++rank) {
                for (int component = 0; component < kHidden; ++component) {
                    const double p = projection[static_cast<std::size_t>(rank) * kHidden +
                                                component];
                    const double h = hidden[static_cast<std::size_t>(component) +
                                            static_cast<std::size_t>(kHidden) *
                                                (position + kDrafts * batch)];
                    projected[rank] += p * h;
                }
            }
            const std::size_t base = static_cast<std::size_t>(kTopK) *
                                     (position + kDrafts * batch);
            std::vector<double> score(kTopK);
            for (int choice = 0; choice < kTopK; ++choice) {
                const int candidate = expected_candidates[base + static_cast<std::size_t>(choice)];
                double value        = logit(batch, position, candidate);
                for (int rank = 0; rank < kRank; ++rank) {
                    value += predecessor[static_cast<std::size_t>(previous) * kRank + rank] *
                             projected[rank] *
                             successor[static_cast<std::size_t>(candidate) * kRank + rank];
                }
                score[static_cast<std::size_t>(choice)] = value;
            }

            int pick = 0;
            if (!(configs[static_cast<std::size_t>(batch)].temperature > 0.0f)) {
                for (int choice = 1; choice < kTopK; ++choice) {
                    const int candidate =
                        expected_candidates[base + static_cast<std::size_t>(choice)];
                    const int best = expected_candidates[base + static_cast<std::size_t>(pick)];
                    if (score[static_cast<std::size_t>(choice)] >
                            score[static_cast<std::size_t>(pick)] ||
                        (score[static_cast<std::size_t>(choice)] ==
                             score[static_cast<std::size_t>(pick)] &&
                         candidate < best)) {
                        pick = choice;
                    }
                }
                expected_q[base + static_cast<std::size_t>(pick)] = 1.0;
            } else {
                const double peak = *std::max_element(score.begin(), score.end());
                double mass       = 0.0;
                for (int choice = 0; choice < kTopK; ++choice) {
                    const double probability = std::exp(
                        (score[static_cast<std::size_t>(choice)] - peak) /
                        configs[static_cast<std::size_t>(batch)].temperature);
                    expected_q[base + static_cast<std::size_t>(choice)] = probability;
                    mass += probability;
                }
                for (int choice = 0; choice < kTopK; ++choice) {
                    expected_q[base + static_cast<std::size_t>(choice)] /= mass;
                }
                const double u = uniform(configs[static_cast<std::size_t>(batch)].seed,
                                         lengths[static_cast<std::size_t>(batch)] + 1 + position,
                                         ops::kSamplePurposeDFlashSelect);
                double cumulative = 0.0;
                pick              = kTopK - 1;
                for (int choice = 0; choice < kTopK; ++choice) {
                    cumulative += expected_q[base + static_cast<std::size_t>(choice)];
                    if (u < cumulative) {
                        pick = choice;
                        break;
                    }
                }
            }
            previous = expected_candidates[base + static_cast<std::size_t>(pick)];
            expected_drafts[static_cast<std::size_t>(position) +
                            static_cast<std::size_t>(kDrafts) * batch] = previous;
        }
    }

    GuardedDeviceBuffer d_logits(logits_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_projection(projection_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_predecessor(predecessor_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_successor(successor_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_lengths(lengths.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_extents(extents.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_configs(configs.size() * sizeof(ops::SamplingConfig));
    GuardedDeviceBuffer d_drafts(expected_drafts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_candidates(expected_candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_q(expected_q.size() * sizeof(float));
    d_logits.copy_from_host(logits_bits.data(), d_logits.bytes());
    d_hidden.copy_from_host(hidden_bits.data(), d_hidden.bytes());
    d_projection.copy_from_host(projection_bits.data(), d_projection.bytes());
    d_predecessor.copy_from_host(predecessor_bits.data(), d_predecessor.bytes());
    d_successor.copy_from_host(successor_bits.data(), d_successor.bytes());
    d_anchors.copy_from_host(anchors.data(), d_anchors.bytes());
    d_lengths.copy_from_host(lengths.data(), d_lengths.bytes());
    d_extents.copy_from_host(extents.data(), d_extents.bytes());
    d_configs.copy_from_host(configs.data(), d_configs.bytes());
    d_drafts.fill(0xcd);
    d_candidates.fill(0xcd);
    d_q.fill(0xcd);
    std::vector<std::uint8_t> config_bytes(d_configs.bytes());
    std::memcpy(config_bytes.data(), configs.data(), config_bytes.size());

    const std::size_t workspace_capacity = ops::dflash2_select_path_workspace_capacity_bytes(
        kVocab, kRank, kTopK, kDrafts, kDrafts, kBatch, kBatch);
    GuardedDeviceBuffer d_workspace(workspace_capacity);
    d_workspace.fill(0x5a);
    WorkspaceArena workspace(DeviceSpan{d_workspace.data(), d_workspace.bytes()});

    Tensor logits_tensor(d_logits.data(), DType::BF16, {kVocab, kDrafts, kBatch});
    Tensor hidden_tensor(d_hidden.data(), DType::BF16, {kHidden, kDrafts, kBatch});
    const Weight projection_weight = bf16_weight(d_projection, kRank, kHidden);
    const Weight predecessor_weight = bf16_weight(d_predecessor, kVocab, kRank);
    const Weight successor_weight = bf16_weight(d_successor, kVocab, kRank);
    Tensor anchors_tensor(d_anchors.data(), DType::I32, {kBatch});
    Tensor lengths_tensor(d_lengths.data(), DType::I32, {kBatch});
    Tensor extents_tensor(d_extents.data(), DType::I32, {kBatch});
    Tensor drafts_tensor(d_drafts.data(), DType::I32, {kDrafts, kBatch});
    Tensor candidates_tensor(d_candidates.data(), DType::I32, {kTopK, kDrafts, kBatch});
    Tensor q_tensor(d_q.data(), DType::FP32, {kTopK, kDrafts, kBatch});
    ops::dflash2_select_path(
        logits_tensor, hidden_tensor, projection_weight, predecessor_weight, successor_weight,
        anchors_tensor, static_cast<const ops::SamplingConfig*>(d_configs.data()), lengths_tensor,
        extents_tensor, kVocab, kTopK, workspace, drafts_tensor, candidates_tensor, q_tensor,
        nullptr);
    cuda_synchronize();

    int failures = verify_exact((label + " candidates/tie order").c_str(),
                                from_device<std::int32_t>(d_candidates.data(),
                                                          expected_candidates.size()),
                                expected_candidates);
    failures += verify_exact((label + " drafts/zero tail").c_str(),
                             from_device<std::int32_t>(d_drafts.data(), expected_drafts.size()),
                             expected_drafts);
    const std::vector<float> actual_q =
        from_device<float>(d_q.data(), expected_q.size());
    std::vector<double> actual_q_double(actual_q.begin(), actual_q.end());
    failures += verify_pointwise(label + " probabilities/zero tail", actual_q_double, expected_q,
                                 PointwiseCriterion{3.0e-5, 3.0e-5});

    failures += verify_exact((label + " logits immutable").c_str(),
                             from_device<std::uint16_t>(d_logits.data(), logits_bits.size()),
                             logits_bits);
    failures += verify_exact((label + " hidden immutable").c_str(),
                             from_device<std::uint16_t>(d_hidden.data(), hidden_bits.size()),
                             hidden_bits);
    failures += verify_exact((label + " projection immutable").c_str(),
                             from_device<std::uint16_t>(d_projection.data(),
                                                        projection_bits.size()),
                             projection_bits);
    failures += verify_exact((label + " predecessor immutable").c_str(),
                             from_device<std::uint16_t>(d_predecessor.data(),
                                                        predecessor_bits.size()),
                             predecessor_bits);
    failures += verify_exact((label + " successor immutable").c_str(),
                             from_device<std::uint16_t>(d_successor.data(), successor_bits.size()),
                             successor_bits);
    failures += verify_exact((label + " anchors immutable").c_str(),
                             from_device<std::int32_t>(d_anchors.data(), anchors.size()), anchors);
    failures += verify_exact((label + " lengths immutable").c_str(),
                             from_device<std::int32_t>(d_lengths.data(), lengths.size()), lengths);
    failures += verify_exact((label + " extents immutable").c_str(),
                             from_device<std::int32_t>(d_extents.data(), extents.size()), extents);
    failures += verify_exact((label + " configs immutable").c_str(),
                             from_device<std::uint8_t>(d_configs.data(), config_bytes.size()),
                             config_bytes);

    const std::size_t unary_bytes =
        static_cast<std::size_t>(kTopK) * kDrafts * kBatch * sizeof(float);
    const std::size_t hidden_offset = (unary_bytes + 255U) & ~std::size_t{255U};
    const std::size_t expected_high_water =
        hidden_offset +
        static_cast<std::size_t>(kRank) * kDrafts * kBatch * sizeof(std::uint16_t);
    if (workspace.capacity() != workspace_capacity || workspace.used() != 0 ||
        workspace.peak_used() != expected_high_water) {
        std::cerr << label << ": caller workspace capacity/high-water mismatch capacity="
                  << workspace.capacity() << " expected_capacity=" << workspace_capacity
                  << " used=" << workspace.used() << " peak=" << workspace.peak_used()
                  << " expected_peak=" << expected_high_water << '\n';
        ++failures;
    }

    failures += d_logits.verify_guards(label + " logits guards");
    failures += d_hidden.verify_guards(label + " hidden guards");
    failures += d_projection.verify_guards(label + " projection guards");
    failures += d_predecessor.verify_guards(label + " predecessor guards");
    failures += d_successor.verify_guards(label + " successor guards");
    failures += d_anchors.verify_guards(label + " anchor guards");
    failures += d_lengths.verify_guards(label + " length guards");
    failures += d_extents.verify_guards(label + " extent guards");
    failures += d_configs.verify_guards(label + " config guards");
    failures += d_drafts.verify_guards(label + " draft guards");
    failures += d_candidates.verify_guards(label + " candidate guards");
    failures += d_q.verify_guards(label + " probability guards");
    failures += d_workspace.verify_guards(label + " workspace guards");
    return failures;
}

int run_real_shape_quantized_selector_case(QType qtype, int physical_vocab, int token_domain,
                                            int hidden_size, int drafts, std::string label,
                                            std::uint32_t seed) {
    constexpr int kRank = 256;
    constexpr int kTopK = 16;
    constexpr int kAnchor = 4321;
    constexpr int kCandidateBase = 1000;
    if (physical_vocab < token_domain || kCandidateBase + kTopK > token_domain || drafts <= 0) {
        throw std::invalid_argument(label + ": invalid real-shape selector case");
    }

    quantized_weight::PatternedWeightOptions options;
    options.row_split_scale = quantized_weight::RowSplitScalePattern::Unit;
    options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
    auto projection = quantized_weight::make_patterned_weight(
        qtype, kRank, hidden_size, seed, options);
    auto predecessor = quantized_weight::make_patterned_weight(
        qtype, physical_vocab, kRank, seed + 12U, options);
    auto successor = quantized_weight::make_patterned_weight(
        qtype, physical_vocab, kRank, seed + 24U, options);

    std::vector<float> hidden(static_cast<std::size_t>(hidden_size) * drafts);
    fill_uniform(hidden, seed + 32U, -0.01F, 0.01F);
    round_to_bf16(hidden);
    std::vector<float> logits(static_cast<std::size_t>(physical_vocab) * drafts, -1000.0F);
    for (int draft = 0; draft < drafts; ++draft) {
        for (int candidate = 0; candidate < kTopK; ++candidate) {
            logits[kCandidateBase + candidate +
                   static_cast<std::size_t>(draft) * physical_vocab] = 0.0F;
        }
        // The Qwen output matrix has physical padding. If the selector accidentally uses the
        // physical row count as the token domain, these poisoned rows displace every legal
        // candidate and the exact candidate assertion below fails.
        for (int token = token_domain; token < physical_vocab; ++token) {
            logits[token + static_cast<std::size_t>(draft) * physical_vocab] = 1000.0F;
        }
    }
    round_to_bf16(logits);

    std::vector<int> expected_choices(drafts);
    std::vector<double> projected(kRank, 0.0);
    std::vector<double> scores(kTopK, 0.0);
    std::vector<int> order(kTopK);
    int predecessor_token = kAnchor;
    for (int draft = 0; draft < drafts; ++draft) {
        const float* hidden_column =
            hidden.data() + static_cast<std::size_t>(draft) * hidden_size;
        for (int rank = 0; rank < kRank; ++rank) {
            projected[static_cast<std::size_t>(rank)] =
                quantized_weight::dot_fp64(projection, rank, hidden_column, hidden_size);
        }
        for (int choice = 0; choice < kTopK; ++choice) {
            const int token = kCandidateBase + choice;
            double score = 0.0;
            for (int rank = 0; rank < kRank; ++rank) {
                score += quantized_weight::logical_weight_fp64(
                             predecessor, predecessor_token, rank) *
                         projected[static_cast<std::size_t>(rank)] *
                         quantized_weight::logical_weight_fp64(successor, token, rank);
            }
            scores[static_cast<std::size_t>(choice)] = score;
            order[static_cast<std::size_t>(choice)] = choice;
        }
        std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
            return scores[static_cast<std::size_t>(lhs)] != scores[static_cast<std::size_t>(rhs)]
                       ? scores[static_cast<std::size_t>(lhs)] >
                             scores[static_cast<std::size_t>(rhs)]
                       : lhs < rhs;
        });
        const double margin = scores[static_cast<std::size_t>(order[0])] -
                              scores[static_cast<std::size_t>(order[1])];
        if (!(margin > 0.1)) {
            std::cerr << label << ": oracle margin is too small at draft " << draft
                      << ": " << margin << '\n';
            return 1;
        }
        expected_choices[static_cast<std::size_t>(draft)] = order[0];
        predecessor_token = kCandidateBase + order[0];
    }

    const std::vector<std::uint16_t> hidden_bits = bf16_bits(hidden);
    const std::vector<std::uint16_t> logits_bits = bf16_bits(logits);
    DeviceBuffer d_projection(projection.payload.size());
    DeviceBuffer d_predecessor(predecessor.payload.size());
    DeviceBuffer d_successor(successor.payload.size());
    DeviceBuffer d_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    DeviceBuffer d_logits(logits_bits.size() * sizeof(std::uint16_t));
    DeviceBuffer d_anchor(sizeof(std::int32_t));
    DeviceBuffer d_length(sizeof(std::int32_t));
    DeviceBuffer d_extent(sizeof(std::int32_t));
    DeviceBuffer d_config(sizeof(ops::SamplingConfig));
    GuardedDeviceBuffer d_drafts(static_cast<std::size_t>(drafts) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_candidates(static_cast<std::size_t>(kTopK) * drafts *
                                     sizeof(std::int32_t));
    GuardedDeviceBuffer d_q(static_cast<std::size_t>(kTopK) * drafts * sizeof(float));
    d_projection.copy_from_host(projection.payload.data(), projection.payload.size());
    d_predecessor.copy_from_host(predecessor.payload.data(), predecessor.payload.size());
    d_successor.copy_from_host(successor.payload.data(), successor.payload.size());
    d_hidden.copy_from_host(hidden_bits.data(), d_hidden.bytes);
    d_logits.copy_from_host(logits_bits.data(), d_logits.bytes);
    const std::int32_t anchor = kAnchor;
    const std::int32_t length = 77;
    const std::int32_t extent = drafts;
    ops::SamplingConfig config{};
    config.temperature = 0.0F;
    d_anchor.copy_from_host(&anchor, sizeof(anchor));
    d_length.copy_from_host(&length, sizeof(length));
    d_extent.copy_from_host(&extent, sizeof(extent));
    d_config.copy_from_host(&config, sizeof(config));
    d_drafts.fill(0xcd);
    d_candidates.fill(0xcd);
    d_q.fill(0xcd);

    const std::size_t workspace_capacity = ops::dflash2_select_path_workspace_capacity_bytes(
        token_domain, kRank, kTopK, drafts, drafts, 1, 1);
    WorkspaceArena workspace(workspace_capacity);
    Tensor logits_tensor(d_logits.p, DType::BF16, {physical_vocab, drafts, 1});
    Tensor hidden_tensor(d_hidden.p, DType::BF16, {hidden_size, drafts, 1});
    Tensor anchor_tensor(d_anchor.p, DType::I32, {1});
    Tensor length_tensor(d_length.p, DType::I32, {1});
    Tensor extent_tensor(d_extent.p, DType::I32, {1});
    Tensor draft_tensor(d_drafts.data(), DType::I32, {drafts, 1});
    Tensor candidate_tensor(d_candidates.data(), DType::I32, {kTopK, drafts, 1});
    Tensor q_tensor(d_q.data(), DType::FP32, {kTopK, drafts, 1});
    const Weight projection_weight = projection.device_weight(d_projection.p);
    const Weight predecessor_weight = predecessor.device_weight(d_predecessor.p);
    const Weight successor_weight = successor.device_weight(d_successor.p);
    ops::dflash2_select_path(
        logits_tensor, hidden_tensor, projection_weight, predecessor_weight, successor_weight,
        anchor_tensor, static_cast<const ops::SamplingConfig*>(d_config.p), length_tensor,
        extent_tensor, token_domain, kTopK, workspace, draft_tensor, candidate_tensor, q_tensor,
        nullptr);
    cuda_synchronize();

    std::vector<std::int32_t> expected_candidates(
        static_cast<std::size_t>(kTopK) * drafts);
    std::vector<float> expected_q(static_cast<std::size_t>(kTopK) * drafts, 0.0F);
    for (int draft = 0; draft < drafts; ++draft) {
        for (int choice = 0; choice < kTopK; ++choice) {
            expected_candidates[static_cast<std::size_t>(choice + kTopK * draft)] =
                kCandidateBase + choice;
        }
    }
    std::vector<std::int32_t> expected_drafts(drafts, 0);
    for (int draft = 0; draft < drafts; ++draft) {
        const int expected_choice = expected_choices[static_cast<std::size_t>(draft)];
        expected_q[static_cast<std::size_t>(expected_choice + kTopK * draft)] = 1.0F;
        expected_drafts[static_cast<std::size_t>(draft)] =
            kCandidateBase + expected_choice;
    }
    int failures = verify_exact((label + " candidates").c_str(),
                                from_device<std::int32_t>(d_candidates.data(),
                                                          expected_candidates.size()),
                                expected_candidates);
    failures += verify_exact((label + " selected draft").c_str(),
                             from_device<std::int32_t>(d_drafts.data(), expected_drafts.size()),
                             expected_drafts);
    failures += verify_exact((label + " greedy probabilities").c_str(),
                             from_device<float>(d_q.data(), expected_q.size()), expected_q);
    failures += d_drafts.verify_guards(label + " draft guards");
    failures += d_candidates.verify_guards(label + " candidate guards");
    failures += d_q.verify_guards(label + " probability guards");
    if (workspace.used() != 0 || workspace.peak_used() > workspace_capacity ||
        workspace_capacity - workspace.peak_used() >= 256) {
        std::cerr << label << ": caller workspace mismatch used=" << workspace.used()
                  << " peak=" << workspace.peak_used()
                  << " capacity=" << workspace_capacity << '\n';
        ++failures;
    }
    return failures;
}

int run_context_copy_fusion_case() {
    constexpr int kLearned = 3;
    constexpr int kTarget = 15;
    constexpr int kBatch = 2;
    constexpr int kTopK = 4;
    constexpr int kMinimumMatch = 6;

    std::vector<std::int32_t> learned_drafts(kLearned * kBatch);
    std::vector<std::int32_t> learned_candidates(kTopK * kLearned * kBatch);
    std::vector<float> learned_q(kTopK * kLearned * kBatch);
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int column = 0; column < kLearned; ++column) {
            learned_drafts[column + kLearned * batch] = 1000 + 100 * batch + column;
            for (int rank = 0; rank < kTopK; ++rank) {
                const std::size_t at = static_cast<std::size_t>(
                    rank + kTopK * (column + kLearned * batch));
                learned_candidates[at] = 2000 + 100 * batch + 10 * column + rank;
                learned_q[at] = 0.01F * static_cast<float>(1 + rank + 4 * column + 16 * batch);
            }
        }
    }
    std::vector<std::int32_t> context_tokens(kTarget * kBatch);
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int column = 0; column < kTarget; ++column) {
            context_tokens[column + kTarget * batch] = 3000 + 100 * batch + column;
        }
    }
    const std::vector<std::int32_t> match_lengths = {6, 5};
    const std::vector<std::int32_t> context_extents = {15, 12};

    std::vector<std::int32_t> expected_drafts(kTarget * kBatch, 0);
    std::vector<std::int32_t> expected_candidates(kTopK * kTarget * kBatch, 0);
    std::vector<float> expected_q(kTopK * kTarget * kBatch, 0.0F);
    for (int column = 0; column < context_extents[0]; ++column) {
        const int token = context_tokens[column];
        expected_drafts[column] = token;
        expected_candidates[kTopK * column] = token;
        expected_q[kTopK * column] = 1.0F;
    }
    for (int column = 0; column < kLearned; ++column) {
        expected_drafts[column + kTarget] = learned_drafts[column + kLearned];
        for (int rank = 0; rank < kTopK; ++rank) {
            const std::size_t learned_at = static_cast<std::size_t>(
                rank + kTopK * (column + kLearned));
            const std::size_t target_at = static_cast<std::size_t>(
                rank + kTopK * (column + kTarget));
            expected_candidates[target_at] = learned_candidates[learned_at];
            expected_q[target_at] = learned_q[learned_at];
        }
    }

    GuardedDeviceBuffer d_learned_drafts(learned_drafts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_learned_candidates(learned_candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_learned_q(learned_q.size() * sizeof(float));
    GuardedDeviceBuffer d_context_tokens(context_tokens.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_match_lengths(match_lengths.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_context_extents(context_extents.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_drafts(expected_drafts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_candidates(expected_candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_q(expected_q.size() * sizeof(float));
    d_learned_drafts.copy_from_host(learned_drafts.data(),
                                    learned_drafts.size() * sizeof(std::int32_t));
    d_learned_candidates.copy_from_host(learned_candidates.data(),
                                        learned_candidates.size() * sizeof(std::int32_t));
    d_learned_q.copy_from_host(learned_q.data(), learned_q.size() * sizeof(float));
    d_context_tokens.copy_from_host(context_tokens.data(),
                                    context_tokens.size() * sizeof(std::int32_t));
    d_match_lengths.copy_from_host(match_lengths.data(),
                                   match_lengths.size() * sizeof(std::int32_t));
    d_context_extents.copy_from_host(context_extents.data(),
                                     context_extents.size() * sizeof(std::int32_t));
    d_drafts.fill(0xcd);
    d_candidates.fill(0xcd);
    d_q.fill(0xcd);

    Tensor learned_draft_tensor(d_learned_drafts.data(), DType::I32, {kLearned, kBatch});
    Tensor learned_candidate_tensor(d_learned_candidates.data(), DType::I32,
                                    {kTopK, kLearned, kBatch});
    Tensor learned_q_tensor(d_learned_q.data(), DType::FP32, {kTopK, kLearned, kBatch});
    Tensor context_token_tensor(d_context_tokens.data(), DType::I32, {kTarget, kBatch});
    Tensor match_length_tensor(d_match_lengths.data(), DType::I32, {kBatch});
    Tensor context_extent_tensor(d_context_extents.data(), DType::I32, {kBatch});
    Tensor draft_tensor(d_drafts.data(), DType::I32, {kTarget, kBatch});
    Tensor candidate_tensor(d_candidates.data(), DType::I32, {kTopK, kTarget, kBatch});
    Tensor q_tensor(d_q.data(), DType::FP32, {kTopK, kTarget, kBatch});
    ops::dflash2_fuse_context_copy(
        learned_draft_tensor, learned_candidate_tensor, learned_q_tensor, context_token_tensor,
        match_length_tensor, context_extent_tensor, kMinimumMatch, draft_tensor,
        candidate_tensor, q_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_exact(
        "DFlash context-copy fused drafts",
        from_device<std::int32_t>(d_drafts.data(), expected_drafts.size()), expected_drafts);
    failures += verify_exact(
        "DFlash context-copy fused candidates",
        from_device<std::int32_t>(d_candidates.data(), expected_candidates.size()),
        expected_candidates);
    failures += verify_exact(
        "DFlash context-copy fused probabilities",
        from_device<float>(d_q.data(), expected_q.size()), expected_q);
    failures += verify_exact(
        "DFlash context-copy learned drafts immutable",
        from_device<std::int32_t>(d_learned_drafts.data(), learned_drafts.size()),
        learned_drafts);
    failures += verify_exact(
        "DFlash context-copy learned candidates immutable",
        from_device<std::int32_t>(d_learned_candidates.data(), learned_candidates.size()),
        learned_candidates);
    failures += verify_exact(
        "DFlash context-copy learned probabilities immutable",
        from_device<float>(d_learned_q.data(), learned_q.size()), learned_q);
    failures += verify_exact(
        "DFlash context-copy tokens immutable",
        from_device<std::int32_t>(d_context_tokens.data(), context_tokens.size()),
        context_tokens);
    failures += verify_exact(
        "DFlash context-copy match lengths immutable",
        from_device<std::int32_t>(d_match_lengths.data(), match_lengths.size()), match_lengths);
    failures += verify_exact(
        "DFlash context-copy extents immutable",
        from_device<std::int32_t>(d_context_extents.data(), context_extents.size()),
        context_extents);
    failures += d_learned_drafts.verify_guards("DFlash context-copy learned draft guards");
    failures += d_learned_candidates.verify_guards(
        "DFlash context-copy learned candidate guards");
    failures += d_learned_q.verify_guards("DFlash context-copy learned probability guards");
    failures += d_context_tokens.verify_guards("DFlash context-copy token guards");
    failures += d_match_lengths.verify_guards("DFlash context-copy match guards");
    failures += d_context_extents.verify_guards("DFlash context-copy extent guards");
    failures += d_drafts.verify_guards("DFlash context-copy fused draft guards");
    failures += d_candidates.verify_guards("DFlash context-copy fused candidate guards");
    failures += d_q.verify_guards("DFlash context-copy fused probability guards");
    return failures;
}

Case bonus_boundary_case() {
    Case c = base_case("DFlash top-64 accept/bonus boundary", 202048, 1);
    c.initial_length = 16; // bonus position 18
    c.config.seed    = 34861; // bonus numerator=60/64 exactly
    c.drafts         = {0};
    c.candidates.resize(16);
    c.q_probs.assign(16, (1.0f - 1.0f / 64.0f) / 15.0f);
    for (int i = 0; i < 16; ++i) c.candidates[static_cast<std::size_t>(i)] = i;
    c.q_probs[0] = 1.0f / 64.0f; // p(draft)==q(draft), hence every u<1 accepts.
    return c;
}

Case correction_boundary_case() {
    Case c = base_case("DFlash top-64 rejection/residual boundary", 202048, 1);
    c.initial_length = 17; // correction position 18
    c.config.seed    = 24837; // correction numerator=62/64 exactly
    c.drafts         = {100};
    c.candidates.resize(16);
    c.q_probs.assign(16, 0.0f);
    c.candidates[0] = 100;
    c.q_probs[0]    = 1.0f;
    for (int i = 1; i < 16; ++i) c.candidates[static_cast<std::size_t>(i)] = 80 + i;
    return c;
}

Case accept_boundary_case() {
    Case c = base_case("DFlash strict acceptance boundary", 96, 1);
    c.initial_length = 17;
    c.config.seed    = 10864166; // accept u=1/4 exactly at position 18
    c.drafts         = {0};
    c.candidates.resize(16);
    c.q_probs.assign(16, 1.0f / 16.0f);
    for (int i = 0; i < 16; ++i) c.candidates[static_cast<std::size_t>(i)] = i;
    // u*q == (1/4)*(1/16) == p(draft)==1/64.  The strict contract rejects.
    return c;
}

Case smaller_top_k_case() {
    Case c = base_case("DFlash smaller top-k and filters", 96, 1);
    c.initial_length = 33;
    c.config.seed    = std::numeric_limits<std::uint64_t>::max();
    c.config.top_k   = 7;
    c.config.top_p   = 0.5f; // Equal weights retain the first four of seven.
    c.config.min_p   = 0.5f;
    c.drafts         = {90};
    c.candidates.resize(16);
    c.q_probs.assign(16, 0.0f);
    c.candidates[0] = 90;
    c.q_probs[0]    = 1.0f;
    for (int i = 1; i < 16; ++i) c.candidates[static_cast<std::size_t>(i)] = 80 + i;
    return c;
}

Case penalty_overlay_case() {
    Case c = base_case("DFlash count and accepted-draft penalties", 96, 2);
    c.config.seed              = 0;
    c.config.presence_penalty  = 1.0f;
    c.config.frequency_penalty = 0.5f;
    c.initial_counts[1]        = 2;
    c.drafts                   = {0, 90};
    c.candidates.resize(32);
    c.q_probs.assign(32, 0.0f);
    c.candidates[0] = 0;
    c.q_probs[0]    = 1.0f / 64.0f;
    for (int i = 1; i < 16; ++i) c.candidates[static_cast<std::size_t>(i)] = 64 + i;
    c.candidates[16] = 90;
    c.q_probs[16]    = 1.0f;
    for (int i = 1; i < 16; ++i) c.candidates[static_cast<std::size_t>(16 + i)] = 64 + i;
    return c;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: CUDA device unavailable\n";
            return 0;
        }
        int failures = 0;
        failures += run_case(bonus_boundary_case());
        failures += run_case(correction_boundary_case());
        failures += run_case(accept_boundary_case());
        failures += run_case(smaller_top_k_case());
        failures += run_case(penalty_overlay_case());
        failures += run_batched_acceptance_case();
        failures += run_batched_selector_case();
        failures += run_context_copy_fusion_case();
        failures += run_real_shape_quantized_selector_case(
            QType::Q4G64_F16S, 202048, 202048, 6656, 15,
            "DFlash Q4 selector real Muse shapes", 1709U);
        failures += run_real_shape_quantized_selector_case(
            QType::Q4G64_F16S, 248320, 248077, 5120, 4,
            "DFlash Q4 selector real Qwen3.8 padded shapes", 1801U);
        failures += run_real_shape_quantized_selector_case(
            QType::W8G32_F16S, 248320, 248077, 5120, 4,
            "DFlash W8 selector real Qwen3.8 padded shapes", 1901U);
        if (failures != 0) {
            std::cerr << "dflash2_select test failures: " << failures << '\n';
            return 1;
        }
        std::cout << "dflash2_select tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash2_select test exception: " << error.what() << '\n';
        return 1;
    }
}
