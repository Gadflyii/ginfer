#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ginfer/ops/gqa_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ginfer;
using namespace ginfer::test;

namespace {

constexpr std::int32_t kHeadDim  = 128;
constexpr std::int32_t kQHeads   = 32;
constexpr std::int32_t kKvHeads  = 2;
constexpr std::int32_t kQuantGroup = 64;
constexpr std::int32_t kQuantGroups = kHeadDim / kQuantGroup;
constexpr float kScale = 3.87f / 11.313708498984761f; // 3.87 / sqrt(128)
constexpr std::uint16_t kCacheCanary = 0x40e0u; // BF16 7.0, distinct from fixture values.
constexpr std::uint16_t kOutputCanary = 0x7fc1u;

constexpr ReductionCriterion kMuseBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::size_t q_index(std::int32_t head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(head);
}

std::size_t prompt_q_index(std::int32_t token, std::int32_t head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token));
}

std::size_t kv_index(std::int32_t head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(head);
}

std::size_t history_index(std::int32_t position, std::int32_t head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKvHeads) * static_cast<std::size_t>(position));
}

std::size_t paged_index(std::int32_t physical_page, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kKvHeads) *
                         static_cast<std::size_t>(physical_page)));
}

std::vector<double> cached_decode_oracle(const std::vector<float>& q,
                                         const std::vector<float>& history_k,
                                         const std::vector<float>& history_v,
                                         const std::vector<float>& new_k,
                                         const std::vector<float>& new_v,
                                         std::int32_t position) {
    constexpr std::int32_t kGroup = kQHeads / kKvHeads;
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) * kQHeads);
    std::vector<double> scores(static_cast<std::size_t>(position) + 1);
    std::vector<double> probabilities(scores.size());

    for (std::int32_t q_head = 0; q_head < kQHeads; ++q_head) {
        const std::int32_t kv_head = q_head / kGroup;
        double maximum             = -std::numeric_limits<double>::infinity();
        for (std::int32_t key_position = 0; key_position <= position; ++key_position) {
            double dot = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const float key = key_position == position
                                      ? new_k[kv_index(kv_head, d)]
                                      : history_k[history_index(key_position, kv_head, d)];
                dot += static_cast<double>(q[q_index(q_head, d)]) *
                       static_cast<double>(key);
            }
            const double score = static_cast<double>(kScale) * dot;
            scores[static_cast<std::size_t>(key_position)] = score;
            maximum = std::max(maximum, score);
        }

        double denominator = 0.0;
        for (std::int32_t key_position = 0; key_position <= position; ++key_position) {
            const double probability =
                std::exp(scores[static_cast<std::size_t>(key_position)] - maximum);
            probabilities[static_cast<std::size_t>(key_position)] = probability;
            denominator += probability;
        }

        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            double value = 0.0;
            for (std::int32_t key_position = 0; key_position <= position; ++key_position) {
                const float cached_value =
                    key_position == position
                        ? new_v[kv_index(kv_head, d)]
                        : history_v[history_index(key_position, kv_head, d)];
                value += probabilities[static_cast<std::size_t>(key_position)] /
                         denominator * static_cast<double>(cached_value);
            }
            output[q_index(q_head, d)] = value;
        }
    }
    return output;
}

std::vector<double> sliding_prompt_oracle(const std::vector<float>& q,
                                          const std::vector<float>& history_k,
                                          const std::vector<float>& history_v,
                                          const std::vector<float>& new_k,
                                          const std::vector<float>& new_v,
                                          std::int32_t base_position, std::int32_t width,
                                          std::int32_t window_size) {
    constexpr std::int32_t kGroup = kQHeads / kKvHeads;
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) * kQHeads *
                               static_cast<std::size_t>(width));
    std::vector<double> scores(static_cast<std::size_t>(window_size));
    std::vector<double> probabilities(scores.size());

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int32_t query_position = base_position + token;
        const std::int32_t key_begin      = std::max(0, query_position - window_size + 1);
        const std::int32_t key_count      = query_position - key_begin + 1;
        for (std::int32_t q_head = 0; q_head < kQHeads; ++q_head) {
            const std::int32_t kv_head = q_head / kGroup;
            double maximum             = -std::numeric_limits<double>::infinity();
            for (std::int32_t key_offset = 0; key_offset < key_count; ++key_offset) {
                const std::int32_t key_position = key_begin + key_offset;
                const bool cached              = key_position < base_position;
                const std::int32_t source_position =
                    cached ? key_position : key_position - base_position;
                const std::vector<float>& source_k = cached ? history_k : new_k;
                double dot                         = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[prompt_q_index(token, q_head, d)]) *
                           static_cast<double>(
                               source_k[history_index(source_position, kv_head, d)]);
                }
                const double score = static_cast<double>(kScale) * dot;
                scores[static_cast<std::size_t>(key_offset)] = score;
                maximum = std::max(maximum, score);
            }

            double denominator = 0.0;
            for (std::int32_t key_offset = 0; key_offset < key_count; ++key_offset) {
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(key_offset)] - maximum);
                probabilities[static_cast<std::size_t>(key_offset)] = probability;
                denominator += probability;
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t key_offset = 0; key_offset < key_count; ++key_offset) {
                    const std::int32_t key_position = key_begin + key_offset;
                    const bool cached              = key_position < base_position;
                    const std::int32_t source_position =
                        cached ? key_position : key_position - base_position;
                    const std::vector<float>& source_v = cached ? history_v : new_v;
                    value += probabilities[static_cast<std::size_t>(key_offset)] /
                             denominator *
                             static_cast<double>(
                                 source_v[history_index(source_position, kv_head, d)]);
                }
                output[prompt_q_index(token, q_head, d)] = value;
            }
        }
    }
    return output;
}

int run_bf16_cached_decode_regression() {
    constexpr const char* label          = "muse D=128 BF16 T=1 cached page-boundary decode";
    constexpr std::int32_t position      = kPagedKVPageSize;
    constexpr std::int32_t logical_pages = 2;
    constexpr std::int32_t physical_pages = 3;
    constexpr std::int32_t table_row     = 0;
    const std::vector<std::int32_t> block_table{2, 0};
    const std::vector<std::int32_t> positions{position};

    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) * kQHeads;
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) * kKvHeads;
    const std::size_t history_elements = kv_elements * static_cast<std::size_t>(position);
    const std::size_t cache_elements = static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                       kKvHeads * physical_pages;

    std::vector<float> q(q_elements);
    std::vector<float> new_k(kv_elements);
    std::vector<float> new_v(kv_elements);
    std::vector<float> history_k(history_elements);
    std::vector<float> history_v(history_elements);
    fill_uniform(q, 0x6d01u, -0.25f, 0.25f);
    fill_uniform(new_k, 0x6d02u, -0.25f, 0.25f);
    fill_uniform(new_v, 0x6d03u, -1.0f, 1.0f);
    fill_uniform(history_k, 0x6d04u, -0.25f, 0.25f);
    fill_uniform(history_v, 0x6d05u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(new_k);
    round_to_bf16(new_v);
    round_to_bf16(history_k);
    round_to_bf16(history_v);

    const std::vector<std::uint16_t> q_bits     = to_bf16_bits(q);
    const std::vector<std::uint16_t> new_k_bits = to_bf16_bits(new_k);
    const std::vector<std::uint16_t> new_v_bits = to_bf16_bits(new_v);
    std::vector<std::uint16_t> initial_k(cache_elements, kCacheCanary);
    std::vector<std::uint16_t> initial_v(cache_elements, kCacheCanary);
    for (std::int32_t cached_position = 0; cached_position < position; ++cached_position) {
        const std::int32_t physical_page =
            block_table[static_cast<std::size_t>(cached_position) / kPagedKVPageSize];
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t logical = history_index(cached_position, head, d);
                const std::size_t physical = paged_index(physical_page, head, cached_position, d);
                initial_k[physical] = f32_to_bf16(history_k[logical]);
                initial_v[physical] = f32_to_bf16(history_v[logical]);
            }
        }
    }
    std::vector<std::uint16_t> expected_k = initial_k;
    std::vector<std::uint16_t> expected_v = initial_v;
    const std::int32_t destination_page = block_table[position / kPagedKVPageSize];
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const std::size_t physical = paged_index(destination_page, head, position, d);
            expected_k[physical] = new_k_bits[kv_index(head, d)];
            expected_v[physical] = new_v_bits[kv_index(head, d)];
        }
    }
    const std::vector<double> reference =
        cached_decode_oracle(q, history_k, history_v, new_k, new_v, position);

    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(new_k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(new_v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dblock(block_table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dk_pages(initial_k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv_pages(initial_v.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(new_k_bits.data(), new_k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(new_v_bits.data(), new_v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    dblock.copy_from_host(block_table.data(), block_table.size() * sizeof(std::int32_t));
    dk_pages.copy_from_host(initial_k.data(), initial_k.size() * sizeof(std::uint16_t));
    dv_pages.copy_from_host(initial_v.data(), initial_v.size() * sizeof(std::uint16_t));
    const std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, kQHeads, 1, 1});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, kKvHeads, 1, 1});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, kKvHeads, 1, 1});
    Tensor tp(dp.data(), DType::I32, {1, 1});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, kQHeads, 1, 1});
    PagedKVBatchLayerView cache;
    cache.k_pages = Tensor(dk_pages.data(), DType::BF16,
                           {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages});
    cache.v_pages = Tensor(dv_pages.data(), DType::BF16,
                           {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages});
    cache.block_tables = Tensor(dblock.data(), DType::I32, {logical_pages, 1});
    cache.head_dim      = kHeadDim;
    cache.num_kv_heads = kKvHeads;
    cache.dtype         = DType::BF16;

    const ops::GqaExecutionEnvelope envelope{position + 1u, position + 1u};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, DType::BF16, envelope, 1, 1, 1, test_occupancy());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kScale, cache, envelope, workspace,
                       tout, test_occupancy(), nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(dout.data(), q_bits.size()), reference,
                                    kMuseBf16Criterion);
    failures += verify_exact((std::string(label) + " cache-k").c_str(),
                             from_device<std::uint16_t>(dk_pages.data(), cache_elements),
                             expected_k);
    failures += verify_exact((std::string(label) + " cache-v").c_str(),
                             from_device<std::uint16_t>(dv_pages.data(), cache_elements),
                             expected_v);
    failures += verify_exact((std::string(label) + " q unchanged").c_str(),
                             from_device<std::uint16_t>(dq.data(), q_bits.size()), q_bits);
    failures += verify_exact((std::string(label) + " k unchanged").c_str(),
                             from_device<std::uint16_t>(dk.data(), new_k_bits.size()), new_k_bits);
    failures += verify_exact((std::string(label) + " v unchanged").c_str(),
                             from_device<std::uint16_t>(dv.data(), new_v_bits.size()), new_v_bits);
    failures += verify_exact((std::string(label) + " position unchanged").c_str(),
                             from_device<std::int32_t>(dp.data(), positions.size()), positions);
    failures += verify_exact((std::string(label) + " table row unchanged").c_str(),
                             from_device<std::int32_t>(dtable_row.data(), 1), {table_row});
    failures += verify_exact((std::string(label) + " block table unchanged").c_str(),
                             from_device<std::int32_t>(dblock.data(), block_table.size()),
                             block_table);
    failures += dq.verify_guards(std::string(label) + " q");
    failures += dk.verify_guards(std::string(label) + " k");
    failures += dv.verify_guards(std::string(label) + " v");
    failures += dp.verify_guards(std::string(label) + " position");
    failures += dtable_row.verify_guards(std::string(label) + " table row");
    failures += dblock.verify_guards(std::string(label) + " block table");
    failures += dk_pages.verify_guards(std::string(label) + " cache-k");
    failures += dv_pages.verify_guards(std::string(label) + " cache-v");
    failures += dout.verify_guards(std::string(label) + " output");
    failures += workspace_buffer.verify_guards(std::string(label) + " workspace");
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_bf16_sliding_prompt_empty_first_tile_regression() {
    constexpr const char* label          = "muse D=128 BF16 T=64 sliding prompt empty first tile";
    constexpr std::int32_t width         = kPagedKVPageSize;
    constexpr std::int32_t base_position = 2048;
    constexpr std::int32_t window_size   = 2048;
    constexpr std::int32_t logical_pages  = 33;
    constexpr std::int32_t physical_pages = logical_pages;
    constexpr std::int32_t table_row      = 0;

    // Row 63 admits [64, 2111], so the row-0-selected first key tile [0, 63] is wholly masked.

    const std::size_t q_elements =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(width);
    const std::size_t new_kv_elements =
        static_cast<std::size_t>(kHeadDim) * kKvHeads * static_cast<std::size_t>(width);
    const std::size_t history_elements =
        static_cast<std::size_t>(kHeadDim) * kKvHeads *
        static_cast<std::size_t>(base_position);
    const std::size_t cache_elements = static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                       kKvHeads * physical_pages;

    std::vector<float> q(q_elements);
    std::vector<float> new_k(new_kv_elements);
    std::vector<float> new_v(new_kv_elements);
    std::vector<float> history_k(history_elements);
    std::vector<float> history_v(history_elements);
    fill_uniform(q, 0x7e01u, -0.25f, 0.25f);
    fill_uniform(new_k, 0x7e02u, -0.25f, 0.25f);
    fill_uniform(new_v, 0x7e03u, -1.0f, 1.0f);
    fill_uniform(history_k, 0x7e04u, -0.25f, 0.25f);
    fill_uniform(history_v, 0x7e05u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(new_k);
    round_to_bf16(new_v);
    round_to_bf16(history_k);
    round_to_bf16(history_v);

    const std::vector<std::uint16_t> q_bits     = to_bf16_bits(q);
    const std::vector<std::uint16_t> new_k_bits = to_bf16_bits(new_k);
    const std::vector<std::uint16_t> new_v_bits = to_bf16_bits(new_v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
    fill_iota_i32(positions, base_position);
    std::vector<std::int32_t> block_table(static_cast<std::size_t>(logical_pages));
    fill_iota_i32(block_table);

    std::vector<std::uint16_t> initial_k(cache_elements, kCacheCanary);
    std::vector<std::uint16_t> initial_v(cache_elements, kCacheCanary);
    for (std::int32_t position = 0; position < base_position; ++position) {
        const std::int32_t physical_page =
            block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t logical  = history_index(position, head, d);
                const std::size_t physical = paged_index(physical_page, head, position, d);
                initial_k[physical]         = f32_to_bf16(history_k[logical]);
                initial_v[physical]         = f32_to_bf16(history_v[logical]);
            }
        }
    }
    std::vector<std::uint16_t> expected_k = initial_k;
    std::vector<std::uint16_t> expected_v = initial_v;
    for (std::int32_t token = 0; token < width; ++token) {
        const std::int32_t position = base_position + token;
        const std::int32_t physical_page =
            block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t logical  = history_index(token, head, d);
                const std::size_t physical = paged_index(physical_page, head, position, d);
                expected_k[physical]        = new_k_bits[logical];
                expected_v[physical]        = new_v_bits[logical];
            }
        }
    }
    const std::vector<double> reference = sliding_prompt_oracle(
        q, history_k, history_v, new_k, new_v, base_position, width, window_size);

    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(new_k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(new_v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dblock(block_table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dk_pages(initial_k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv_pages(initial_v.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(new_k_bits.data(), new_k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(new_v_bits.data(), new_v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    dblock.copy_from_host(block_table.data(), block_table.size() * sizeof(std::int32_t));
    dk_pages.copy_from_host(initial_k.data(), initial_k.size() * sizeof(std::uint16_t));
    dv_pages.copy_from_host(initial_v.data(), initial_v.size() * sizeof(std::uint16_t));
    const std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, kQHeads, width, 1});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, kKvHeads, width, 1});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, kKvHeads, width, 1});
    Tensor tp(dp.data(), DType::I32, {width, 1});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, kQHeads, width, 1});
    PagedKVBatchLayerView cache;
    cache.k_pages = Tensor(dk_pages.data(), DType::BF16,
                           {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages});
    cache.v_pages = Tensor(dv_pages.data(), DType::BF16,
                           {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages});
    cache.block_tables = Tensor(dblock.data(), DType::I32, {logical_pages, 1});
    cache.head_dim      = kHeadDim;
    cache.num_kv_heads = kKvHeads;
    cache.dtype         = DType::BF16;

    constexpr ops::GqaExecutionEnvelope envelope{base_position + width,
                                                  base_position + width};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, DType::BF16, envelope, 1, width, width, test_occupancy());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kScale, cache, envelope, workspace,
                       tout, test_occupancy(), nullptr, window_size);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(dout.data(), q_bits.size()), reference,
                                    kMuseBf16Criterion);
    failures += verify_exact((std::string(label) + " cache-k").c_str(),
                             from_device<std::uint16_t>(dk_pages.data(), cache_elements),
                             expected_k);
    failures += verify_exact((std::string(label) + " cache-v").c_str(),
                             from_device<std::uint16_t>(dv_pages.data(), cache_elements),
                             expected_v);
    failures += verify_exact((std::string(label) + " q unchanged").c_str(),
                             from_device<std::uint16_t>(dq.data(), q_bits.size()), q_bits);
    failures += verify_exact((std::string(label) + " k unchanged").c_str(),
                             from_device<std::uint16_t>(dk.data(), new_k_bits.size()), new_k_bits);
    failures += verify_exact((std::string(label) + " v unchanged").c_str(),
                             from_device<std::uint16_t>(dv.data(), new_v_bits.size()), new_v_bits);
    failures += verify_exact((std::string(label) + " positions unchanged").c_str(),
                             from_device<std::int32_t>(dp.data(), positions.size()), positions);
    failures += verify_exact((std::string(label) + " table row unchanged").c_str(),
                             from_device<std::int32_t>(dtable_row.data(), 1), {table_row});
    failures += verify_exact((std::string(label) + " block table unchanged").c_str(),
                             from_device<std::int32_t>(dblock.data(), block_table.size()),
                             block_table);
    failures += dq.verify_guards(std::string(label) + " q");
    failures += dk.verify_guards(std::string(label) + " k");
    failures += dv.verify_guards(std::string(label) + " v");
    failures += dp.verify_guards(std::string(label) + " positions");
    failures += dtable_row.verify_guards(std::string(label) + " table row");
    failures += dblock.verify_guards(std::string(label) + " block table");
    failures += dk_pages.verify_guards(std::string(label) + " cache-k");
    failures += dv_pages.verify_guards(std::string(label) + " cache-v");
    failures += dout.verify_guards(std::string(label) + " output");
    failures += workspace_buffer.verify_guards(std::string(label) + " workspace");
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int count_non_finite(const std::vector<double>& values) {
    int count = 0;
    for (double value : values) {
        if (!std::isfinite(value)) { ++count; }
    }
    return count;
}

int run_muse_case(const char* label, DType dtype, std::int32_t width, std::int32_t window_size) {
    const std::int32_t logical_pages  = 1;
    const std::int32_t physical_pages = 1;
    const std::size_t q_elements =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(width);
    const std::size_t kv_elements =
        static_cast<std::size_t>(kHeadDim) * kKvHeads * static_cast<std::size_t>(width);
    const std::size_t code_elements = static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                      kKvHeads * static_cast<std::size_t>(physical_pages);
    const std::size_t scale_elements = static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                                       kKvHeads * static_cast<std::size_t>(physical_pages);

    std::vector<float> q(q_elements);
    std::vector<float> k(kv_elements);
    std::vector<float> v(kv_elements);
    fill_uniform(q, 0x4d01u, -0.25f, 0.25f);
    fill_uniform(k, 0x4d02u, -0.25f, 0.25f);
    fill_uniform(v, 0x4d03u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
    fill_iota_i32(positions, 0);
    const std::int32_t table_row = 0;

    GuardedDeviceBuffer dq(q_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(kv_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(kv_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk_pages(code_elements *
                                 (dtype == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t)));
    GuardedDeviceBuffer dv_pages(code_elements *
                                 (dtype == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t)));
    GuardedDeviceBuffer dk_scale(dtype == DType::I8 ? scale_elements * sizeof(std::uint16_t) : 1);
    GuardedDeviceBuffer dv_scale(dtype == DType::I8 ? scale_elements * sizeof(std::uint16_t) : 1);
    GuardedDeviceBuffer dblock(logical_pages * sizeof(std::int32_t));

    {
        std::vector<std::uint16_t> qb(q.size()), kb(k.size()), vb(v.size());
        for (std::size_t i = 0; i < q.size(); ++i) qb[i] = f32_to_bf16(q[i]);
        for (std::size_t i = 0; i < k.size(); ++i) kb[i] = f32_to_bf16(k[i]);
        for (std::size_t i = 0; i < v.size(); ++i) vb[i] = f32_to_bf16(v[i]);
        dq.copy_from_host(qb.data(), qb.size() * sizeof(std::uint16_t));
        dk.copy_from_host(kb.data(), kb.size() * sizeof(std::uint16_t));
        dv.copy_from_host(vb.data(), vb.size() * sizeof(std::uint16_t));
    }
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dtable.copy_from_host(&table_row, sizeof(table_row));
    const std::int32_t page0 = 0;
    dblock.copy_from_host(&page0, sizeof(page0));
    dk_pages.fill(0);
    dv_pages.fill(0);
    if (dtype == DType::I8) {
        dk_scale.fill(0);
        dv_scale.fill(0);
    }
    dout.fill(0);

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, kQHeads, width, 1});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, kKvHeads, width, 1});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, kKvHeads, width, 1});
    Tensor tp(dp.data(), DType::I32, {width, 1});
    Tensor ttable(dtable.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, kQHeads, width, 1});

    PagedKVBatchLayerView cache;
    cache.k_pages = Tensor(dk_pages.data(), dtype,
                           {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages});
    cache.v_pages = Tensor(dv_pages.data(), dtype,
                           {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages});
    cache.block_tables = Tensor(dblock.data(), DType::I32, {logical_pages, 1});
    cache.head_dim     = kHeadDim;
    cache.num_kv_heads = kKvHeads;
    cache.dtype        = dtype;
    if (dtype == DType::I8) {
        cache.k_scale_pages = Tensor(dk_scale.data(), DType::FP16,
                                     {kQuantGroups, kPagedKVPageSize, kKvHeads, physical_pages});
        cache.v_scale_pages = Tensor(dv_scale.data(), DType::FP16,
                                     {kQuantGroups, kPagedKVPageSize, kKvHeads, physical_pages});
        cache.quant_group   = kQuantGroup;
    }

    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(width), 64};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, dtype, envelope, 1, width, width, test_occupancy());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable, kScale, cache, envelope, workspace, tout,
                       test_occupancy(), nullptr, window_size);
    cuda_synchronize();

    const std::vector<double> output = from_device_bf16(dout.data(), q_elements);
    const int non_finite             = count_non_finite(output);
    int failures                     = 0;
    if (non_finite != 0) {
        std::cerr << label << ": " << non_finite << " non-finite outputs\n";
        ++failures;
    }
    failures += dout.verify_guards((std::string(label) + " output").c_str());
    failures += workspace_buffer.verify_guards((std::string(label) + " workspace").c_str());
    failures += dk_pages.verify_guards((std::string(label) + " cache-k").c_str());
    failures += dv_pages.verify_guards((std::string(label) + " cache-v").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_bf16_cached_decode_regression();
    failures += run_bf16_sliding_prompt_empty_first_tile_regression();
    failures += run_muse_case("muse W=1 BF16", DType::BF16, 1, 0);
    failures += run_muse_case("muse W=8 BF16", DType::BF16, 8, 0);
    failures += run_muse_case("muse W=1 INT8", DType::I8, 1, 0);
    failures += run_muse_case("muse W=8 window=4", DType::BF16, 8, 4);
    if (failures != 0) {
        std::cerr << failures << " Muse GQA failures\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
