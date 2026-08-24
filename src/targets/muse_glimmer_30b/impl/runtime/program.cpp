#include <ginfer/targets/muse_glimmer_30b/package.h>

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ginfer/ops/dflash2_select.h"
#include "ginfer/ops/embedding.h"
#include "ginfer/ops/gqa_attention.h"
#include "ginfer/ops/grouped_dynamic_conv.h"
#include "ginfer/ops/kv_cache_append_prefix.h"
#include "ginfer/ops/linear.h"
#include "ginfer/ops/prepare_masked_block.h"
#include "ginfer/ops/residual_add.h"
#include "ginfer/ops/rmsnorm.h"
#include "ginfer/ops/rope.h"
#include "ginfer/ops/sampling.h"
#include "ginfer/ops/sigmoid_mul.h"
#include "ginfer/ops/silu_mul.h"
#include "ginfer/ops/speculative_round.h"
#include "ginfer/ops/swa.h"
#include "targets/muse_glimmer_30b/impl/config.h"
#include "targets/muse_glimmer_30b/impl/load/bindings.h"
#include "targets/muse_glimmer_30b/impl/runtime/logit_softcap.h"
#include "targets/muse_glimmer_30b/impl/runtime/program_storage.h"
#include "targets/muse_glimmer_30b/impl/runtime/sequence.h"
#include "targets/muse_glimmer_30b/impl/runtime/workspace_recipe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ginfer::targets::muse_glimmer_30b::detail {
namespace {

using Clock = std::chrono::steady_clock;

constexpr bool kGemmaRms     = true;
constexpr float kRmsEps      = TextConfig::rms_epsilon;
constexpr float kPostNormEps = TextConfig::post_norm_eps;
constexpr float kAttnScale   = kAttentionScale;

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

void fill_bf16_value(void* destination, std::int32_t count, float value) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(count), float_to_bf16(value));
    CUDA_CHECK(cudaMemcpy(destination, host.data(), host.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
}

std::uint32_t pages_for(std::uint32_t tokens) {
    if (tokens == 0) { return 1; }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base,
                                     std::uint32_t prefill_chunk) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk;
    const std::uint64_t decode_units = summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

void copy_d2d(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    if (src.bytes() != dst.bytes()) { throw std::logic_error("muse copy size mismatch"); }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src.data, src.bytes(), cudaMemcpyDeviceToDevice, stream));
}

PagedKVBatchLayerView batch_layer(const PagedKVPool& pool, int layer, DType dtype,
                                  std::int32_t quant_group) {
    const bool quantized     = dtype == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool.plane(base),
        .v_pages       = pool.plane(base + 1),
        .k_scale_pages = quantized ? pool.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool.plane(base + 3) : Tensor(),
        .block_tables  = pool.block_tables(),
        .head_dim      = TextConfig::head_dim,
        .num_kv_heads  = TextConfig::kv_heads,
        .dtype         = dtype,
        .quant_group   = quant_group,
    };
}

Tensor heads_q(const Tensor& packed, std::int32_t width, std::int32_t batch = 1) {
    return packed.view({TextConfig::head_dim, TextConfig::query_heads, width, batch});
}

Tensor heads_kv(const Tensor& packed, std::int32_t width, std::int32_t batch = 1) {
    return packed.view({TextConfig::head_dim, TextConfig::kv_heads, width, batch});
}

int capture_index(int layer) {
    for (int i = 0; i < DFlash2Config::layers; ++i) {
        if (DFlash2Config::target_layer_ids[i] == layer) { return i; }
    }
    return -1;
}

void copy_feature_slice(const Tensor& residual, Tensor& packed, int slot, std::int32_t tokens,
                        cudaStream_t stream) {
    const std::size_t hidden_bytes =
        static_cast<std::size_t>(TextConfig::hidden) * sizeof(std::uint16_t);
    auto* dst = static_cast<std::byte*>(packed.data) + static_cast<std::size_t>(slot) * hidden_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(dst, packed.nb[1], residual.data, residual.nb[1], hidden_bytes,
                                 static_cast<std::size_t>(tokens), cudaMemcpyDeviceToDevice,
                                 stream));
}

Tensor dflash_heads_q(const Tensor& packed, std::int32_t width, std::int32_t batch = 1) {
    return packed.view({TextConfig::head_dim, TextConfig::query_heads, width, batch});
}

Tensor dflash_heads_kv(const Tensor& packed, std::int32_t width, std::int32_t batch = 1) {
    return packed.view({TextConfig::head_dim, DFlash2Config::kv_heads, width, batch});
}

} // namespace

class Program::Impl {
public:
    struct Lane;
    enum class PendingKind : std::uint8_t { None, Begin, Ordinary, Speculative };

    struct PendingRound {
        PendingKind kind       = PendingKind::None;
        std::uint32_t base_seq = 0;
        std::uint32_t produced = 0;
        std::array<TokenId, DFlash2Config::block_size> licensed{};
    };
    struct CudaStream {
        cudaStream_t s = nullptr;
        CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
        ~CudaStream() {
            if (s != nullptr) { (void)cudaStreamDestroy(s); }
        }
        CudaStream(const CudaStream&)            = delete;
        CudaStream& operator=(const CudaStream&) = delete;
    };

    struct CudaEvent {
        cudaEvent_t e = nullptr;
        CudaEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); }
        ~CudaEvent() {
            if (e != nullptr) { (void)cudaEventDestroy(e); }
        }
        CudaEvent(const CudaEvent&)            = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;
    };

    struct DecodeIngress {
        std::array<std::int32_t, kMaximumConcurrency> tokens{};
        std::array<std::int32_t, kMaximumConcurrency> positions{};
        std::array<std::int32_t, kMaximumConcurrency> table_rows{};
        std::array<std::int32_t, kMaximumConcurrency> sample_positions{};
        std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
        std::array<std::int32_t, kMaximumConcurrency> sampled{};
        std::array<std::int32_t, kMaximumConcurrency> extents{};
        std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
        std::array<std::int32_t, kMaximumConcurrency> accepted{};
        std::array<std::int32_t, kMaximumConcurrency * DFlash2Config::block_size> licensed{};
    };

    Impl(const RuntimeModel& model_in, SequencePlan plan_in, DeviceContext& device_in)
        : model(&model_in), plan(std::move(plan_in)), device(&device_in),
          occupancy(device_in.occupancy()), stream(device_in.stream),
          kv_dtype(plan.layout().kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8),
          quant_group(plan.layout().kv_cache == KvCacheStorage::Int8Group64
                          ? static_cast<std::int32_t>(kKvQuantGroup)
                          : 0),
          decode_ingress_host(sizeof(DecodeIngress)) {
        if (model->weights_arena == nullptr) {
            throw std::invalid_argument("Muse model view has no owning weight arena");
        }
        LayoutBuilder kv_builder;
        kv_layout              = plan_paged_kv_pool(kv_builder, make_text_kv_spec(plan.layout()));
        const std::size_t kv_bytes = kv_builder.finish(256, "muse kv");
        if (kv_bytes != plan.layout().full_kv_bytes) {
            throw std::logic_error("muse KV layout does not match the sequence reservation");
        }
        const std::size_t sliding_bytes = plan.layout().sliding_kv_bytes;
        const ProgramStorageLayout program_layout =
            plan_program_storage(plan.max_concurrency());
        if (program_layout.bytes != plan.layout().program_storage_bytes ||
            plan.layout().persistent_bytes !=
                kv_bytes + sliding_bytes + program_layout.bytes) {
            throw std::logic_error("Muse persistent layout does not match the sequence plan");
        }
        sequence_storage = DeviceBuffer(plan.layout().persistent_bytes);
        kv_pool          = std::make_unique<PagedKVPool>(DeviceSpan{sequence_storage.p, kv_bytes},
                                                         kv_layout);
        auto* sliding_ptr = static_cast<std::byte*>(sequence_storage.p) + kv_bytes;
        if (sliding_bytes > 0) {
            LayoutBuilder cyclic_builder;
            dflash_kv_layout = plan_cyclic_kv_cache(
                cyclic_builder, static_cast<std::uint32_t>(DFlash2Config::layers),
                static_cast<std::uint32_t>(DFlash2Config::cyclic_capacity), DFlash2Config::kv_heads,
                TextConfig::head_dim, static_cast<std::int32_t>(plan.max_concurrency()));
            const std::size_t cyclic_bytes = cyclic_builder.finish(256, "muse dflash2 cyclic");
            if (cyclic_bytes != sliding_bytes) {
                throw std::logic_error("muse DFlash2 cyclic layout does not match reservation");
            }
            dflash_kv = std::make_unique<CyclicKVCache>(DeviceSpan{sliding_ptr, sliding_bytes},
                                                        dflash_kv_layout);
        }
        const DeviceSpan program_span{sliding_ptr + sliding_bytes, program_layout.bytes};
        sample_ids         = program_layout.sample_ids.bind(program_span);
        sample_positions   = program_layout.sample_positions.bind(program_span);
        sampling_configs   = program_layout.sampling_configs.bind(program_span);
        token_counts_store = program_layout.token_counts.bind(program_span);
        table_rows_store   = program_layout.table_rows.bind(program_span);
        q_norm_weight      = program_layout.q_norm_weight.bind(program_span);
        k_norm_weight      = program_layout.k_norm_weight.bind(program_span);
        embed_norm_weight  = program_layout.embed_norm_weight.bind(program_span);
        decode_token       = program_layout.decode_token.bind(program_span);
        decode_positions   = program_layout.decode_position.bind(program_span);
        decode_embed_raw   = program_layout.decode_embed_raw.bind(program_span);
        decode_hidden      = program_layout.decode_hidden.bind(program_span);
        workspace          = std::make_unique<DeviceArena>(plan.layout().workspace_bytes);
        dflash_enabled      = plan.layout().dflash_enabled;
        dflash_draft_tokens = static_cast<std::int32_t>(plan.layout().dflash_draft_tokens);

        admission.active_lanes     = plan.max_concurrency();
        admission.main_kv_pages    = plan.layout().main_page_groups;
        admission.backend_kv_pages = 0;

        memory.device                           = device->device;
        memory.max_context                      = plan.capacity();
        memory.kv_capacity                      = plan.kv_capacity();
        memory.kv_capacity_page_groups          = plan.layout().main_page_groups;
        memory.kv_capacity_max_page_groups      = plan.layout().max_main_page_groups;
        memory.kv_cache                         = plan.layout().kv_cache;
        memory.weights = ArenaMemorySummary{model->weights_arena->capacity(),
                                            model->weights_arena->used(),
                                            model->weights_arena->peak_used()};
        memory.sequence.capacity_bytes          = plan.layout().persistent_bytes;
        memory.sequence.used_bytes              = plan.layout().persistent_bytes;
        memory.workspace.capacity_bytes         = plan.workspace_capacity_bytes();
        memory.request_transient.capacity_bytes = plan.request_transient_capacity_bytes();
        memory.runtime_reservation_bytes        = plan.device_reservation_bytes();
        memory.kv_payload_bytes                 = kv_bytes;
        memory.cuda_graph_allowance_bytes       = plan.layout().graph_allowance_bytes;

        CUDA_CHECK(cudaMemset(token_counts_store.data, 0, token_counts_store.bytes()));
        CUDA_CHECK(cudaMemset(table_rows_store.data, 0, table_rows_store.bytes()));

        // llama.cpp synthesizes these at GGUF convert; we keep HF tensors and apply them here.
        fill_bf16_value(q_norm_weight.data, TextConfig::head_dim,
                        TextConfig::qk_scale_factor);
        fill_bf16_value(k_norm_weight.data, TextConfig::head_dim, 1.0F);
        fill_bf16_value(embed_norm_weight.data, TextConfig::hidden, 1.0F);
        if (!dflash_enabled) { prepare_decode_graph(); }
    }

    DecodeIngress& ingress() noexcept {
        return *static_cast<DecodeIngress*>(decode_ingress_host.data());
    }

    void set_table_row(std::int32_t lane) {
        CUDA_CHECK(cudaMemcpyAsync(table_rows_store.data, &lane, sizeof(lane), cudaMemcpyHostToDevice,
                                   stream));
    }

    void decode_ordinary_body(std::int32_t batch) {
        if (batch < 1 || batch > static_cast<std::int32_t>(plan.max_concurrency())) {
            throw std::invalid_argument("Muse ordinary decode batch is out of range");
        }
        DecodeIngress& in = ingress();
        const std::size_t i32_bytes = static_cast<std::size_t>(batch) * sizeof(std::int32_t);
        const std::size_t cfg_bytes =
            static_cast<std::size_t>(batch) * sizeof(ops::SamplingConfig);
        CUDA_CHECK(cudaMemcpyAsync(decode_token.data, in.tokens.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(decode_positions.data, in.positions.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(table_rows_store.data, in.table_rows.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sample_positions.data, in.sample_positions.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sampling_configs.data, in.sampling.data(), cfg_bytes,
                                   cudaMemcpyHostToDevice, stream));
        Tensor tokens = decode_token.slice(0, 0, batch);
        Tensor positions = decode_positions.slice(1, 0, batch);
        Tensor embed_raw = decode_embed_raw.slice(1, 0, batch);
        Tensor hidden = decode_hidden.slice(1, 0, batch);
        ops::embedding(tokens, model->token_embedding, embed_raw, stream);
        ops::rmsnorm(embed_raw, embed_norm_weight, kRmsEps, false, hidden, stream);
        run_layers(hidden, positions, 1, batch, nullptr);

        auto sample_scope = workspace->scope();
        Tensor logits     = workspace->alloc(DType::BF16, {TextConfig::vocab, batch});
        Tensor capped     = workspace->alloc(DType::BF16, {TextConfig::vocab, batch});
        ops::linear(hidden, model->output_head, logits, stream);
        logit_softcap(logits, capped, TextConfig::output_multiplier,
                      TextConfig::final_logit_softcapping, stream);
        Tensor out_ids = sample_ids.slice(0, 0, batch);
        Tensor pos_t   = sample_positions.slice(0, 0, batch);
        ops::sample(capped, out_ids, TextConfig::vocab,
                    static_cast<const ops::SamplingConfig*>(sampling_configs.data), pos_t,
                    ops::kSamplePurposeDecode, *workspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(in.sampled.data(), sample_ids.data, i32_bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }

    void prepare_decode_graph() {
        if (plan.layout().graph_allowance_bytes == 0) { return; }

        const std::uint32_t graph_batches = plan.max_concurrency();
        if (kv_pool->page_group_count() < graph_batches) {
            throw std::invalid_argument(
                "Muse CUDA Graph preparation requires one private KV page per batch row");
        }

        std::vector<PagedKVAllocation> capture_allocations;
        capture_allocations.reserve(graph_batches);
        for (std::uint32_t row = 0; row < graph_batches; ++row) {
            capture_allocations.push_back(kv_pool->reserve(1));
            PagedKVAllocation& allocation = capture_allocations.back();
            allocation.bind_row(static_cast<std::int32_t>(row), stream);
            allocation.materialize_pages(1, stream);

            // A graph representative can exercise any logical context frontier. Repeating a
            // row-private physical page keeps every captured access valid without reserving a
            // complete context solely for startup qualification.
            const std::int32_t page = allocation.page_ids().front();
            std::vector<std::int32_t> repeated(kv_pool->logical_page_capacity(), page);
            Tensor table = kv_pool->block_table_row(static_cast<std::int32_t>(row));
            CUDA_CHECK(cudaMemcpyAsync(table.data, repeated.data(), table.bytes(),
                                       cudaMemcpyHostToDevice, stream));
        }
        ingress() = {};
        for (std::uint32_t row = 0; row < graph_batches; ++row) {
            ingress().table_rows[row] = static_cast<std::int32_t>(row);
        }
        device->synchronize();

        // Warm every exact-B operator route before graph accounting. Batch width selects GEMM and
        // attention launch shapes, so warming only B=1 would charge later code materialization to
        // the graph inventory and would not qualify the B>1 paths before capture.
        for (std::uint32_t batch = 1; batch <= graph_batches; ++batch) {
            workspace->reset();
            decode_ordinary_body(static_cast<std::int32_t>(batch));
            device->synchronize();
        }

        std::size_t free_before = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));

        for (std::uint32_t batch = 1; batch <= graph_batches; ++batch) {
            workspace->reset();
            DecodeGraphDefinition definition;
            definition.capture(stream, [this, batch] {
                decode_ordinary_body(static_cast<std::int32_t>(batch));
            });
            DecodeGraphExecutable& executable = decode_execs[batch - 1U];
            executable.instantiate(definition);
            executable.upload(stream);
            device->synchronize();

            // Execute each exact topology once while its private capture rows are still bound.
            // The definition is then destroyed; only the executable needed by decode remains.
            executable.launch(stream);
            device->synchronize();
        }

        workspace->reset();
        for (PagedKVAllocation& allocation : capture_allocations) { allocation.unbind_row(); }
        capture_allocations.clear();

        std::size_t free_after = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
        const std::size_t consumed = free_before > free_after ? free_before - free_after : 0;
        memory.cuda_graph_observed_bytes = consumed;
        if (consumed > plan.layout().graph_allowance_bytes) {
            throw std::runtime_error(
                "Muse CUDA Graph preparation consumed " + std::to_string(consumed) +
                " bytes, exceeding the planned allowance of " +
                std::to_string(plan.layout().graph_allowance_bytes) + " bytes");
        }
        graphs_ready = true;
    }

    void run_layers(Tensor residual, Tensor positions, std::int32_t width, std::int32_t batch,
                    Tensor* capture_packed, const Tensor& valid_columns = Tensor()) {
        const std::int32_t columns = width * batch;
        auto scope = workspace->scope();
        TextLayersWorkspace storage = allocate_text_layers_workspace(
            *workspace, width, batch, plan.layout().weights_profile == WeightsProfile::Nvfp4);
        Tensor& h                     = storage.h;
        Tensor& alt                   = storage.alt;
        Tensor& q_flat                = storage.q_flat;
        Tensor& k_flat                = storage.k_flat;
        Tensor& q_normed              = storage.q_normed;
        Tensor& k_normed              = storage.k_normed;
        Tensor& v_flat                = storage.v_flat;
        Tensor& gate_flat             = storage.gate_flat;
        Tensor& attn                  = storage.attn;
        Tensor& attn_proj             = storage.attn_proj;
        Tensor& ffn_gate              = storage.ffn_gate;
        Tensor& ffn_up                = storage.ffn_up;
        Tensor& ffn_mid               = storage.ffn_mid;
        Tensor& ffn_down              = storage.ffn_down;
        Tensor table_rows(table_rows_store.data, DType::I32, {batch});

        std::array<std::optional<WorkspaceArena>, 4> linear_arenas;
        for (std::size_t index = 0; index < linear_arenas.size(); ++index) {
            if (storage.linear_scratch[index].bytes != 0) {
                linear_arenas[index].emplace(storage.linear_scratch[index]);
            }
        }
        const auto project = [&](const Tensor& input, const Weight& weight, Tensor& output,
                                 std::size_t arena, cudaStream_t projection_stream) {
            if (weight.qtype != QType::NVFP4) {
                ops::linear(input, weight, output, projection_stream);
            } else if (linear_arenas[arena].has_value()) {
                ops::linear(input, weight, output, ops::LinearPolicy::AllowA4,
                            *linear_arenas[arena], projection_stream);
            } else {
                ops::linear(input, weight, output, projection_stream);
            }
        };

        const ops::GqaExecutionEnvelope envelope{
            .min_visible_keys = 1,
            .max_visible_keys = plan.capacity(),
        };
        for (int layer = 0; layer < TextConfig::layers; ++layer) {
            const TextLayerWeights& w = model->text_layers[static_cast<std::size_t>(layer)];
            ops::rmsnorm(residual, w.input_norm, kRmsEps, kGemmaRms, h, stream);
            CUDA_CHECK(cudaEventRecord(ev_h.e, stream));
            CUDA_CHECK(cudaStreamWaitEvent(par_k.s, ev_h.e, 0));
            CUDA_CHECK(cudaStreamWaitEvent(par_v.s, ev_h.e, 0));
            CUDA_CHECK(cudaStreamWaitEvent(par_gate.s, ev_h.e, 0));
            project(h, w.attention.query, q_flat, 0, stream);
            project(h, w.attention.key, k_flat, 1, par_k.s);
            project(h, w.attention.value, v_flat, 2, par_v.s);
            project(h, w.attention.gate, gate_flat, 3, par_gate.s);
            CUDA_CHECK(cudaEventRecord(ev_k.e, par_k.s));
            CUDA_CHECK(cudaEventRecord(ev_v.e, par_v.s));
            CUDA_CHECK(cudaEventRecord(ev_gate.e, par_gate.s));

            Tensor q    = heads_q(q_flat, width, batch);
            Tensor k    = heads_kv(k_flat, width, batch);
            Tensor qn   = heads_q(q_normed, width, batch);
            Tensor kn   = heads_kv(k_normed, width, batch);
            Tensor v    = heads_kv(v_flat, width, batch);
            Tensor gate = heads_q(gate_flat, width, batch);
            ops::rmsnorm(q, q_norm_weight, kRmsEps, false, qn, stream);
            CUDA_CHECK(cudaStreamWaitEvent(stream, ev_k.e, 0));
            ops::rmsnorm(k, k_norm_weight, kRmsEps, false, kn, stream);
            CUDA_CHECK(cudaStreamWaitEvent(stream, ev_v.e, 0));
            if (!w.full_attention) {
                Tensor rope_positions = positions.view({columns});
                Tensor rope_q = qn.view(
                    {TextConfig::head_dim, TextConfig::query_heads, columns, 1});
                Tensor rope_k = kn.view(
                    {TextConfig::head_dim, TextConfig::kv_heads, columns, 1});
                ops::rope(rope_positions, TextConfig::rotary_dim, TextConfig::rope_theta, rope_q,
                          rope_k, occupancy, stream);
            }
            const std::int32_t window = w.full_attention ? 0 : TextConfig::sliding_window;
            auto gqa_scope            = workspace->scope();
            ops::gqa_attention(qn, kn, v, positions, valid_columns, table_rows, kAttnScale,
                               batch_layer(*kv_pool, layer, kv_dtype, quant_group), envelope,
                               *workspace, attn, occupancy, stream, window);
            CUDA_CHECK(cudaStreamWaitEvent(stream, ev_gate.e, 0));
            ops::sigmoid_mul(gate, attn, stream);
            Tensor attn_flat = attn.view({TextConfig::query_size, columns});
            project(attn_flat, w.attention.output, attn_proj, 0, stream);
            ops::rmsnorm(attn_proj, w.post_attention_norm, kPostNormEps, kGemmaRms, alt, stream);
            ops::residual_add(alt, residual, stream);

            ops::rmsnorm(residual, w.pre_feedforward_norm, kRmsEps, kGemmaRms, h, stream);
            CUDA_CHECK(cudaEventRecord(ev_h.e, stream));
            CUDA_CHECK(cudaStreamWaitEvent(par_k.s, ev_h.e, 0));
            project(h, w.mlp.gate, ffn_gate, 0, stream);
            project(h, w.mlp.up, ffn_up, 1, par_k.s);
            CUDA_CHECK(cudaEventRecord(ev_up.e, par_k.s));
            CUDA_CHECK(cudaStreamWaitEvent(stream, ev_up.e, 0));
            ops::silu_mul(ffn_gate, ffn_up, ffn_mid, stream);
            project(ffn_mid, w.mlp.down, ffn_down, 0, stream);
            ops::rmsnorm(ffn_down, w.post_feedforward_norm, kPostNormEps, kGemmaRms, alt, stream);
            ops::residual_add(alt, residual, stream);
            if (capture_packed != nullptr) {
                const int slot = capture_index(layer);
                if (slot >= 0) {
                    copy_feature_slice(residual, *capture_packed, slot, columns, stream);
                }
            }
        }
        ops::rmsnorm(residual, model->output_norm, kRmsEps, false, h, stream);
        copy_d2d(h, residual, stream);
    }

    ops::SamplingConfig to_cfg(const ResolvedSamplingParameters& s, std::uint32_t lane) {
        if (lane >= plan.max_concurrency()) {
            throw std::out_of_range("muse sampling lane is out of range");
        }
        ops::SamplingConfig c;
        c.temperature       = s.temperature;
        c.top_k             = s.top_k;
        c.top_p             = s.top_p;
        c.min_p             = s.min_p;
        c.presence_penalty  = s.presence_penalty;
        c.frequency_penalty = s.frequency_penalty;
        c.seed              = s.seed;
        const bool penalties = s.presence_penalty != 0.0F || s.frequency_penalty != 0.0F;
        auto* counts = static_cast<std::int32_t*>(token_counts_store.data) +
                       static_cast<std::size_t>(lane) * TextConfig::vocab;
        c.token_counts = penalties ? counts : nullptr;
        return c;
    }

    void reset_sampling_state(std::uint32_t lane, const ResolvedSamplingParameters& sampling,
                              Lane& request) {
        auto* counts = static_cast<std::int32_t*>(token_counts_store.data) +
                       static_cast<std::size_t>(lane) * TextConfig::vocab;
        CUDA_CHECK(cudaMemsetAsync(counts, 0, sizeof(std::int32_t) * TextConfig::vocab, stream));
        request.sampling  = to_cfg(sampling, lane);
        request.spec_stats = {};
        if (dflash_enabled) {
            request.spec_stats.backend      = SpeculativeBackend::DFlash;
            request.spec_stats.enabled      = true;
            request.spec_stats.draft_window = static_cast<std::uint32_t>(dflash_draft_tokens);
            request.spec_stats.verification_window =
                static_cast<std::uint32_t>(dflash_draft_tokens);
            request.spec_stats.accepted_per_position.assign(
                static_cast<std::size_t>(std::max(dflash_draft_tokens, 0)), 0);
        }
    }

    TokenId sample_from_hidden(const Tensor& hidden, std::int32_t tokens, std::uint32_t position,
                               const ops::SamplingConfig& cfg) {
        Tensor last = tokens == 1 ? hidden : hidden.slice(1, tokens - 1, 1);
        auto scope  = workspace->scope();
        Tensor logits = workspace->alloc(DType::BF16, {TextConfig::vocab, 1});
        Tensor capped = workspace->alloc(DType::BF16, {TextConfig::vocab, 1});
        ops::linear(last, model->output_head, logits, stream);
        logit_softcap(logits, capped, TextConfig::output_multiplier,
                      TextConfig::final_logit_softcapping, stream);

        CUDA_CHECK(cudaMemcpyAsync(sampling_configs.data, &cfg, sizeof(cfg), cudaMemcpyHostToDevice,
                                   stream));
        const std::int32_t pos = static_cast<std::int32_t>(position);
        CUDA_CHECK(cudaMemcpyAsync(sample_positions.data, &pos, sizeof(pos), cudaMemcpyHostToDevice,
                                   stream));
        Tensor out_ids(sample_ids.data, DType::I32, {1});
        Tensor pos_t(sample_positions.data, DType::I32, {1});
        ops::sample(capped, out_ids, TextConfig::vocab,
                    static_cast<const ops::SamplingConfig*>(sampling_configs.data), pos_t,
                    ops::kSamplePurposePrefill, *workspace, stream);
        std::int32_t host_id = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&host_id, sample_ids.data, sizeof(host_id), cudaMemcpyDeviceToHost));
        return host_id;
    }

    void embed_tokens(const std::vector<TokenId>& ids, Tensor& hidden) {
        auto scope = workspace->scope();
        Tensor id_t = workspace->alloc(DType::I32, {static_cast<std::int32_t>(ids.size())});
        Tensor raw  = workspace->alloc(DType::BF16, {TextConfig::hidden, hidden.ne[1]});
        CUDA_CHECK(cudaMemcpyAsync(id_t.data, ids.data(), ids.size() * sizeof(TokenId),
                                   cudaMemcpyHostToDevice, stream));
        ops::embedding(id_t, model->token_embedding, raw, stream);
        ops::rmsnorm(raw, embed_norm_weight, kRmsEps, false, hidden, stream);
    }

    void project_and_append_dflash_batch(
        const Tensor& packed_features, const Tensor& positions, std::int32_t width,
        std::int32_t batch, const Tensor& counts, const Tensor& lane_ids,
        ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
        auto scope = workspace->scope();
        const std::int32_t columns = width * batch;
        DFlashProjectionWorkspace storage =
            allocate_dflash_projection_workspace(*workspace, width, batch);
        Tensor& projected = storage.projected;
        Tensor& context   = storage.context;
        ops::linear(packed_features.view({DFlash2Config::captured_rows, columns}),
                    model->dflash2.fc, projected, stream);
        ops::rmsnorm(projected, model->dflash2.hidden_norm, kRmsEps, false, context, stream);

        for (int layer = 0; layer < DFlash2Config::layers; ++layer) {
            const DFlash2LayerWeights& w = model->dflash2.layers[static_cast<std::size_t>(layer)];
            ops::linear(context, w.k_proj, storage.k_flat, stream);
            ops::linear(context, w.v_proj, storage.v_flat, stream);
            Tensor kn_flat_heads =
                storage.k_normed.view({TextConfig::head_dim, DFlash2Config::kv_heads, columns});
            Tensor k  = dflash_heads_kv(storage.k_flat, width, batch);
            Tensor kn = dflash_heads_kv(storage.k_normed, width, batch);
            Tensor v  = dflash_heads_kv(storage.v_flat, width, batch);
            ops::rmsnorm(k, w.k_norm, kRmsEps, false, kn, stream);
            Tensor flat_positions = positions.view({columns});
            ops::rope(flat_positions, TextConfig::rotary_dim, TextConfig::rope_theta,
                      kn_flat_heads, occupancy, stream);
            ops::kv_cache_append_prefix(kn, v, positions, counts, lane_ids, envelope,
                                        dflash_kv->layer_view(static_cast<std::uint32_t>(layer)),
                                        stream);
        }
    }

    void project_and_append_dflash(const Tensor& packed_features, const Tensor& positions,
                                   std::int32_t tokens, std::int32_t lane) {
        auto scope = workspace->scope();
        Tensor counts = workspace->alloc(DType::I32, {1});
        Tensor lanes  = workspace->alloc(DType::I32, {1});
        CUDA_CHECK(cudaMemcpyAsync(counts.data, &tokens, sizeof(tokens), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(lanes.data, &lane, sizeof(lane), cudaMemcpyHostToDevice,
                                   stream));
        project_and_append_dflash_batch(
            packed_features.view({DFlash2Config::captured_rows, tokens, 1}),
            positions.view({tokens, 1}), tokens, 1, counts, lanes,
            {static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(tokens)});
    }

    void run_dflash_layers(Tensor residual, const Tensor& positions, std::int32_t width,
                           std::int32_t batch, const Tensor& valid, const Tensor& lanes) {
        auto scope = workspace->scope();
        const std::int32_t columns = width * batch;
        DFlashLayersWorkspace storage =
            allocate_dflash_layers_workspace(*workspace, width, batch);
        Tensor& h                       = storage.h;
        Tensor& conv_in                 = storage.conv_in;
        Tensor& conv_out                = storage.conv_out;
        Tensor& dynamic                 = storage.dynamic;
        Tensor& q_flat                  = storage.q_flat;
        Tensor& k_flat                  = storage.k_flat;
        Tensor& v_flat                  = storage.v_flat;
        Tensor& q_n                     = storage.q_normed;
        Tensor& k_n                     = storage.k_normed;
        Tensor& attn                    = storage.attn;
        Tensor& attn_proj               = storage.attn_proj;
        Tensor& ffn_gate                = storage.ffn_gate;
        Tensor& ffn_up                  = storage.ffn_up;
        Tensor& ffn_mid                 = storage.ffn_mid;
        Tensor& ffn_down                = storage.ffn_down;
        const ops::SwaContextExecutionEnvelope swa_env{0, plan.capacity()};
        Tensor residual_flat = residual.view({TextConfig::hidden, columns});
        Tensor flat_positions = positions.view({columns});
        Tensor h_batch = h.view({TextConfig::hidden, width, batch});
        Tensor conv_in_batch = conv_in.view({TextConfig::hidden, width, batch});
        Tensor conv_out_batch = conv_out.view({TextConfig::hidden, width, batch});
        Tensor dynamic_batch =
            dynamic.view({DFlash2Config::kernel_projection_rows, width, batch});

        for (int layer = 0; layer < DFlash2Config::layers; ++layer) {
            const DFlash2LayerWeights& w = model->dflash2.layers[static_cast<std::size_t>(layer)];
            ops::rmsnorm(residual_flat, w.input_layernorm, kRmsEps, false, h, stream);
            ops::grouped_dynamic_conv_prepare(
                h_batch, w.attention_conv_proj, w.attention_conv_base, dynamic_batch,
                conv_in_batch, DFlash2Config::conv_group, stream);
            ops::linear(conv_in, w.q_proj, q_flat, stream);
            ops::linear(conv_in, w.k_proj, k_flat, stream);
            ops::linear(conv_in, w.v_proj, v_flat, stream);
            Tensor q  = dflash_heads_q(q_flat, width, batch);
            Tensor k  = dflash_heads_kv(k_flat, width, batch);
            Tensor qn = dflash_heads_q(q_n, width, batch);
            Tensor kn = dflash_heads_kv(k_n, width, batch);
            Tensor v  = dflash_heads_kv(v_flat, width, batch);
            ops::rmsnorm(q, w.q_norm, kRmsEps, false, qn, stream);
            ops::rmsnorm(k, w.k_norm, kRmsEps, false, kn, stream);
            Tensor qn_flat = q_n.view({TextConfig::head_dim, TextConfig::query_heads, columns});
            Tensor kn_flat = k_n.view({TextConfig::head_dim, DFlash2Config::kv_heads, columns});
            ops::rope(flat_positions, TextConfig::rotary_dim, TextConfig::rope_theta, qn_flat,
                      kn_flat, occupancy, stream);
            ops::swa(qn, kn, v, positions, valid, lanes, DFlash2Config::attention_scale,
                     dflash_kv->layer_view(static_cast<std::uint32_t>(layer)), swa_env, *workspace,
                     attn, stream);
            Tensor attn_flat = attn.view({TextConfig::query_size, columns});
            ops::linear(attn_flat, w.o_proj, attn_proj, stream);
            Tensor attn_proj_batch = attn_proj.view({TextConfig::hidden, width, batch});
            ops::grouped_dynamic_conv_finish(attn_proj_batch, dynamic_batch,
                                             w.attention_conv_base, conv_out_batch,
                                             DFlash2Config::conv_group, stream);
            ops::residual_add(conv_out, residual_flat, stream);

            ops::rmsnorm(residual_flat, w.post_attention_layernorm, kRmsEps, false, h, stream);
            ops::grouped_dynamic_conv_prepare(h_batch, w.mlp_conv_proj, w.mlp_conv_base,
                                              dynamic_batch, conv_in_batch,
                                              DFlash2Config::conv_group, stream);
            ops::linear(conv_in, w.mlp_gate, ffn_gate, stream);
            ops::linear(conv_in, w.mlp_up, ffn_up, stream);
            ops::silu_mul(ffn_gate, ffn_up, ffn_mid, stream);
            ops::linear(ffn_mid, w.mlp_down, ffn_down, stream);
            Tensor ffn_down_batch = ffn_down.view({TextConfig::hidden, width, batch});
            ops::grouped_dynamic_conv_finish(ffn_down_batch, dynamic_batch, w.mlp_conv_base,
                                             conv_out_batch, DFlash2Config::conv_group, stream);
            ops::residual_add(conv_out, residual_flat, stream);
        }
        ops::rmsnorm(residual_flat, model->dflash2.norm, kRmsEps, false, h, stream);
        copy_d2d(h, residual_flat, stream);
    }

    void decode_dflash_batch(std::span<const std::uint32_t> lane_members,
                             std::span<const runtime::RoundBudget> budgets) {
        const auto started         = Clock::now();
        const std::int32_t batch   = static_cast<std::int32_t>(lane_members.size());
        const std::int32_t drafts_n = dflash_draft_tokens;
        const std::int32_t width   = drafts_n + 1;
        const std::int32_t columns = width * batch;
        const std::int32_t draft_columns = drafts_n * batch;
        auto scope                 = workspace->scope();
        DecodeIngress& in          = ingress();

        for (std::int32_t row = 0; row < batch; ++row) {
            const Lane& request = lanes[lane_members[static_cast<std::size_t>(row)]];
            const std::uint32_t remaining =
                budgets[static_cast<std::size_t>(row)].generated_tokens_remaining;
            const std::int32_t extent = static_cast<std::int32_t>(
                std::min<std::uint32_t>(static_cast<std::uint32_t>(drafts_n),
                                        remaining > 0 ? remaining - 1U : 0U));
            in.tokens[static_cast<std::size_t>(row)] = request.sampled;
            in.positions[static_cast<std::size_t>(row)] =
                static_cast<std::int32_t>(request.seq_len);
            in.table_rows[static_cast<std::size_t>(row)] =
                static_cast<std::int32_t>(lane_members[static_cast<std::size_t>(row)]);
            in.sample_positions[static_cast<std::size_t>(row)] = extent + 1;
            in.extents[static_cast<std::size_t>(row)]          = extent;
            in.sampling[static_cast<std::size_t>(row)]         = request.sampling;
        }

        Tensor anchors = workspace->alloc(DType::I32, {batch});
        Tensor lengths = workspace->alloc(DType::I32, {batch});
        Tensor valid   = workspace->alloc(DType::I32, {batch});
        Tensor extents = workspace->alloc(DType::I32, {batch});
        Tensor lane_ids = workspace->alloc(DType::I32, {batch});
        Tensor ids      = workspace->alloc(DType::I32, {width, batch});
        Tensor positions = workspace->alloc(DType::I32, {width, batch});
        const std::size_t i32_bytes = static_cast<std::size_t>(batch) * sizeof(std::int32_t);
        const std::size_t config_bytes =
            static_cast<std::size_t>(batch) * sizeof(ops::SamplingConfig);
        CUDA_CHECK(cudaMemcpyAsync(anchors.data, in.tokens.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(lengths.data, in.positions.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(valid.data, in.sample_positions.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(extents.data, in.extents.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(lane_ids.data, in.table_rows.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(table_rows_store.data, in.table_rows.data(), i32_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sampling_configs.data, in.sampling.data(), config_bytes,
                                   cudaMemcpyHostToDevice, stream));
        ops::prepare_masked_block(anchors, lengths, valid, DFlash2Config::mask_token, ids,
                                  positions, stream);

        Tensor residual = workspace->alloc(DType::BF16, {TextConfig::hidden, width, batch});
        Tensor residual_flat = residual.view({TextConfig::hidden, columns});
        ops::embedding(ids.view({columns}), model->token_embedding, residual_flat, stream);
        run_dflash_layers(residual, positions, width, batch, valid, lane_ids);

        Tensor draft_hidden =
            workspace->alloc(DType::BF16, {TextConfig::hidden, drafts_n, batch});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t draft_row_bytes =
            static_cast<std::size_t>(TextConfig::hidden) * drafts_n * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(TextConfig::hidden) * width * element_bytes;
        const auto* draft_source = static_cast<const std::byte*>(residual.data) +
                                   static_cast<std::size_t>(TextConfig::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(draft_hidden.data, draft_row_bytes, draft_source,
                                     source_pitch, draft_row_bytes,
                                     static_cast<std::size_t>(batch), cudaMemcpyDeviceToDevice,
                                     stream));
        Tensor proposal_logits =
            workspace->alloc(DType::BF16, {TextConfig::vocab, drafts_n, batch});
        Tensor proposal_capped =
            workspace->alloc(DType::BF16, {TextConfig::vocab, drafts_n, batch});
        Tensor proposal_logits_flat =
            proposal_logits.view({TextConfig::vocab, draft_columns});
        ops::linear(draft_hidden.view({TextConfig::hidden, draft_columns}), model->output_head,
                    proposal_logits_flat, stream);
        logit_softcap(proposal_logits, proposal_capped, TextConfig::output_multiplier,
                      TextConfig::final_logit_softcapping, stream);
        Tensor drafts = workspace->alloc(DType::I32, {drafts_n, batch});
        Tensor candidates = workspace->alloc(
            DType::I32, {DFlash2Config::selector_top_k, drafts_n, batch});
        Tensor proposal_probabilities = workspace->alloc(
            DType::FP32, {DFlash2Config::selector_top_k, drafts_n, batch});
        ops::dflash2_select_path(
            proposal_capped, draft_hidden, model->dflash2.selector_hidden,
            model->dflash2.predecessor_codebook, model->dflash2.successor_codebook, anchors,
            static_cast<const ops::SamplingConfig*>(sampling_configs.data), lengths, extents,
            TextConfig::vocab, DFlash2Config::selector_top_k, *workspace, drafts, candidates,
            proposal_probabilities, stream);

        Tensor verify_ids = workspace->alloc(DType::I32, {width, batch});
        Tensor verify_positions = workspace->alloc(DType::I32, {width, batch});
        ops::speculative_prepare_verify_inputs(anchors, drafts, lengths, extents, verify_ids,
                                               verify_positions, stream);
        Tensor target_hidden = workspace->alloc(DType::BF16, {TextConfig::hidden, columns});
        Tensor capture =
            workspace->alloc(DType::BF16, {DFlash2Config::captured_rows, width, batch});
        Tensor raw = workspace->alloc(DType::BF16, {TextConfig::hidden, columns});
        ops::embedding(verify_ids.view({columns}), model->token_embedding, raw, stream);
        ops::rmsnorm(raw, embed_norm_weight, kRmsEps, false, target_hidden, stream);
        run_layers(target_hidden, verify_positions, width, batch, &capture, valid);

        Tensor target_logits =
            workspace->alloc(DType::BF16, {TextConfig::vocab, width, batch});
        Tensor target_capped =
            workspace->alloc(DType::BF16, {TextConfig::vocab, width, batch});
        Tensor target_logits_flat = target_logits.view({TextConfig::vocab, columns});
        ops::linear(target_hidden, model->output_head, target_logits_flat, stream);
        logit_softcap(target_logits, target_capped, TextConfig::output_multiplier,
                      TextConfig::final_logit_softcapping, stream);

        Tensor licensed = workspace->alloc(DType::I32, {width, batch});
        Tensor licensed_counts = workspace->alloc(DType::I32, {batch});
        Tensor accepted = workspace->alloc(DType::I32, {batch});
        ops::dflash2_accept(
            target_capped, drafts, candidates, proposal_probabilities, extents,
            static_cast<const ops::SamplingConfig*>(sampling_configs.data), TextConfig::vocab,
            lengths, anchors, licensed, licensed_counts, accepted, *workspace, stream);
        project_and_append_dflash_batch(
            capture, verify_positions, width, batch, licensed_counts, lane_ids,
            {1U, static_cast<std::uint32_t>(width)});

        CUDA_CHECK(cudaMemcpyAsync(in.licensed.data(), licensed.data,
                                   static_cast<std::size_t>(columns) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(in.licensed_counts.data(), licensed_counts.data, i32_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(in.accepted.data(), accepted.data, i32_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(in.sampled.data(), anchors.data, i32_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::int32_t row = 0; row < batch; ++row) {
            Lane& request = lanes[lane_members[static_cast<std::size_t>(row)]];
            const std::int32_t extent = in.extents[static_cast<std::size_t>(row)];
            const std::int32_t produced = in.licensed_counts[static_cast<std::size_t>(row)];
            const std::int32_t accepted_count = in.accepted[static_cast<std::size_t>(row)];
            if (produced <= 0 || produced > extent + 1 || accepted_count < 0 ||
                accepted_count > extent) {
                throw std::logic_error("muse DFlash2 batch returned invalid row counts");
            }
            const std::uint32_t base_seq = request.seq_len;
            request.seq_len += static_cast<std::uint32_t>(produced);
            request.sampled = in.sampled[static_cast<std::size_t>(row)];
            decode_row_counts[static_cast<std::size_t>(row)] = produced;
            request.pending.kind =
                extent == 0 ? PendingKind::Ordinary : PendingKind::Speculative;
            request.pending.base_seq = base_seq;
            request.pending.produced = static_cast<std::uint32_t>(produced);
            for (std::int32_t i = 0; i < produced; ++i) {
                const TokenId token = in.licensed[static_cast<std::size_t>(i + width * row)];
                decode_tokens[static_cast<std::size_t>(i + DFlash2Config::block_size * row)] =
                    token;
                request.pending.licensed[static_cast<std::size_t>(i)] = token;
            }
            if (extent > 0) {
                request.spec_stats.rounds += 1;
                request.spec_stats.drafted_tokens += static_cast<std::uint64_t>(extent);
                request.spec_stats.accepted_tokens +=
                    static_cast<std::uint64_t>(accepted_count);
                if (accepted_count == 0) { request.spec_stats.fallback_steps += 1; }
                for (std::int32_t i = 0; i < accepted_count; ++i) {
                    request.spec_stats.accepted_per_position[static_cast<std::size_t>(i)] += 1;
                }
            }
            request.timings.decode_seconds += elapsed;
        }
    }

    const RuntimeModel* model = nullptr;
    SequencePlan plan;
    DeviceContext* device = nullptr;
    OccupancyPolicy occupancy{};
    cudaStream_t stream   = nullptr;
    DType kv_dtype        = DType::I8;
    std::int32_t quant_group = 0;

    DeviceBuffer sequence_storage;
    PagedKVPoolLayout kv_layout{};
    std::unique_ptr<PagedKVPool> kv_pool;
    CyclicKVCacheLayout dflash_kv_layout{};
    std::unique_ptr<CyclicKVCache> dflash_kv;
    std::unique_ptr<DeviceArena> workspace;
    bool dflash_enabled            = false;
    std::int32_t dflash_draft_tokens = 0;
    Tensor sample_ids;
    Tensor sample_positions;
    DeviceSpan sampling_configs;
    Tensor token_counts_store;
    Tensor table_rows_store;
    Tensor q_norm_weight;
    Tensor k_norm_weight;
    Tensor embed_norm_weight;

    CudaStream par_k;
    CudaStream par_v;
    CudaStream par_gate;
    CudaEvent ev_h;
    CudaEvent ev_k;
    CudaEvent ev_v;
    CudaEvent ev_gate;
    CudaEvent ev_up;

    PinnedHostBuffer decode_ingress_host;
    Tensor decode_token;
    Tensor decode_positions;
    Tensor decode_embed_raw;
    Tensor decode_hidden;
    std::array<DecodeGraphExecutable, kMaximumConcurrency> decode_execs;
    bool graphs_ready = false;

    runtime::AdmissionResources admission{};
    MemorySummary memory{};

    struct Lane {
        bool busy = false;
        bool retained = false;
        bool prefix_reusable = false;
        std::vector<TokenId> prompt;
        std::vector<TokenId> ledger;
        std::uint32_t reuse_base = 0;
        std::uint32_t processed = 0;
        std::uint32_t seq_len   = 0;
        std::uint32_t requested = 0;
        TokenId sampled         = 0;
        PagedKVAllocation kv;
        ops::SamplingConfig sampling{};
        GenerationTimings timings{};
        SpeculativeStats spec_stats{};
        PendingRound pending{};
        Clock::time_point prefill_started{};
    };
    std::array<Lane, kMaximumConcurrency> lanes{};
    std::array<TokenId, kMaximumConcurrency * DFlash2Config::block_size> decode_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> decode_row_counts{};
};

Program::Program(const RuntimeModel& model, SequencePlan plan, DeviceContext& device)
    : impl_(std::make_unique<Impl>(model, std::move(plan), device)) {}

Program::~Program() = default;

RequestBasePlan Program::plan_request_base(const PreparedPrompt& prompt,
                                           const runtime::ResolvedExecutionOptions& options) {
    RequestBasePlan base;
    const std::uint32_t n = static_cast<std::uint32_t>(prompt.token_ids().size());
    if (n == 0) { throw std::invalid_argument("muse prompt is empty"); }
    if (n > impl_->plan.capacity()) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prompt exceeds configured context capacity");
    }
    const std::uint32_t room = impl_->plan.capacity() - n + 1U;
    const std::uint32_t want = options.requested_output_tokens;
    std::uint32_t effective = std::min(want, room);
    FinishReason limit =
        want > room ? FinishReason::ContextCapacity : FinishReason::OutputLimit;
    if (effective == 0) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded, "no remaining KV capacity");
    }
    base.summary_value.prompt_tokens           = n;
    base.summary_value.reusable_prompt_tokens  = 0;
    base.summary_value.requested_output_tokens = want;
    base.summary_value.effective_output_tokens = effective;
    base.summary_value.effective_limit_reason  = limit;
    base.summary_value.transient_bytes         = impl_->plan.request_transient_capacity_bytes();
    base.summary_value.transient_alignment =
        base.summary_value.transient_bytes == 0 ? 1 : 256;
    base.summary_value.admission.active_lanes  = 1;
    base.summary_value.admission.main_kv_pages = pages_for(n + effective - 1U);
    base.summary_value.admission.backend_kv_pages = 0;
    base.summary_value.service_work_quanta = projected_service_work(
        base.summary_value, 0, impl_->plan.layout().prefill_chunk);
    base.sampling                              = options.sampling;
    base.allow_prefix_reuse =
        options.allow_prefix_reuse && prompt.data_->prefix_identity_reusable;
    return base;
}

RequestPlan Program::plan_request_for_lane(std::uint32_t lane, const PreparedPrompt& prompt,
                                           const RequestBasePlan& base) {
    if (lane >= impl_->plan.max_concurrency()) {
        throw std::out_of_range("muse request-plan lane is out of range");
    }
    RequestPlan plan;
    plan.summary_value = base.summary();
    plan.sampling      = base.sampling;
    const Impl::Lane& resident = impl_->lanes[lane];
    if (base.allow_prefix_reuse && resident.retained && resident.prefix_reusable &&
        resident.kv.valid() &&
        !resident.ledger.empty()) {
        const auto tokens = prompt.token_ids();
        const std::size_t limit = std::min<std::size_t>(
            {tokens.size(), resident.ledger.size(), resident.seq_len});
        std::size_t common = 0;
        while (common < limit && tokens[common] == resident.ledger[common]) { ++common; }
        // The hidden state for a completely reused prompt is not retained. Re-execute its final
        // token so the first generated token is sampled from an exact target activation.
        if (common == tokens.size() && common != 0) { --common; }
        // DFlash2 retains only the cyclic interval [seq_len-W, seq_len). Once that interval has
        // wrapped, an earlier matching frontier is not a valid snapshot: at least one K/V slot
        // needed by that frontier has already been overwritten. The exact retained frontier is
        // still reusable when the new prompt has a suffix to execute. An identical prompt backs
        // up one token above, so it must reset after the first wrap.
        if (impl_->dflash_enabled &&
            resident.seq_len > static_cast<std::uint32_t>(DFlash2Config::cyclic_capacity) &&
            !(common == resident.seq_len && tokens.size() > common)) {
            common = 0;
        }
        plan.reuse_base = static_cast<std::uint32_t>(common);
        plan.summary_value.reusable_prompt_tokens = plan.reuse_base;
        plan.summary_value.service_work_quanta = projected_service_work(
            plan.summary_value, plan.reuse_base, impl_->plan.layout().prefill_chunk);
    }
    return plan;
}

bool Program::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= impl_->plan.max_concurrency()) { return false; }
    const Impl::Lane& resident = impl_->lanes[lane];
    if (resident.busy) { return false; }
    const std::uint32_t old_pages = resident.kv.valid() ? resident.kv.page_entitlement() : 0;
    const std::uint32_t new_pages = plan.summary().admission.main_kv_pages;
    const PagedKVPool& pool       = *impl_->kv_pool;
    return old_pages <= pool.entitled_pages() && new_pages <= pool.logical_page_capacity() &&
           new_pages <= pool.page_group_count() - (pool.entitled_pages() - old_pages);
}

bool Program::can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                     const RequestPlan& plan) const noexcept {
    if (lane >= impl_->plan.max_concurrency() || impl_->lanes[lane].busy) { return false; }
    const PagedKVPool& pool = *impl_->kv_pool;
    const std::uint32_t old_pages =
        impl_->lanes[lane].kv.valid() ? impl_->lanes[lane].kv.page_entitlement() : 0;
    std::uint32_t reclaimable = 0;
    for (std::uint32_t other = 0; other < impl_->plan.max_concurrency(); ++other) {
        if (other == lane || !impl_->lanes[other].retained || !impl_->lanes[other].kv.valid()) {
            continue;
        }
        reclaimable += impl_->lanes[other].kv.page_entitlement();
    }
    if (old_pages > pool.entitled_pages() ||
        reclaimable > pool.entitled_pages() - old_pages) {
        return false;
    }
    const std::uint32_t committed = pool.entitled_pages() - old_pages - reclaimable;
    const std::uint32_t new_pages = plan.summary().admission.main_kv_pages;
    return new_pages <= pool.logical_page_capacity() &&
           new_pages <= pool.page_group_count() - committed;
}

runtime::AdmissionResources Program::admission_capacity() const noexcept {
    return impl_->admission;
}

runtime::PrefillStepResult Program::start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt,
                                                       RequestPlan&& plan,
                                                       runtime::TransientRegion) {
    if (lane >= impl_->plan.max_concurrency()) {
        throw std::invalid_argument("muse lane is out of range");
    }
    Impl::Lane& L = impl_->lanes[lane];
    if (L.busy) { throw std::logic_error("muse lane is already active"); }
    try {
        const std::uint32_t reuse = plan.reuse_base;
        const auto prompt_tokens  = prompt.token_ids();
        if (reuse != 0) {
            if (!L.retained || !L.prefix_reusable || !L.kv.valid() || reuse > L.seq_len ||
                reuse > L.ledger.size() || reuse > prompt_tokens.size() ||
                !std::equal(prompt_tokens.begin(), prompt_tokens.begin() + reuse,
                            L.ledger.begin())) {
                throw std::logic_error("planned Muse prefix is no longer reusable");
            }
            if (impl_->dflash_enabled &&
                L.seq_len > static_cast<std::uint32_t>(DFlash2Config::cyclic_capacity) &&
                !(reuse == L.seq_len && prompt_tokens.size() > reuse)) {
                throw std::logic_error("planned Muse prefix is outside retained DFlash2 state");
            }
            L.kv.trim_tokens(reuse);
            L.kv.set_page_entitlement(plan.summary().admission.main_kv_pages);
            L.ledger.resize(reuse);
        } else {
            L = Impl::Lane{};
            L.kv = impl_->kv_pool->reserve(plan.summary().admission.main_kv_pages);
        }

        L.busy            = true;
        L.retained        = false;
        L.prefix_reusable = false;
        L.reuse_base      = reuse;
        L.processed       = reuse;
        L.seq_len         = reuse;
        L.prompt.assign(prompt_tokens.begin(), prompt_tokens.end());
        L.ledger.reserve(static_cast<std::size_t>(L.prompt.size()) +
                         plan.summary().effective_output_tokens);
        L.requested       = plan.summary().effective_output_tokens;
        L.timings         = {};
        L.pending         = {};
        L.prefill_started = Clock::now();
        impl_->reset_sampling_state(lane, plan.sampling, L);
        L.kv.bind_row(static_cast<std::int32_t>(lane), impl_->stream);
        L.kv.materialize_tokens(
            static_cast<std::uint32_t>(L.prompt.size()) + L.requested - 1U, impl_->stream);
        return advance_prefill_lane(lane);
    } catch (...) {
        L = Impl::Lane{};
        throw;
    }
}

runtime::PrefillStepResult Program::advance_prefill_lane(std::uint32_t lane) {
    Impl::Lane& L = impl_->lanes[lane];
    const std::uint32_t chunk = std::max(1U, impl_->plan.layout().prefill_chunk);
    const std::uint32_t begin = L.processed;
    const std::uint32_t end   = std::min(static_cast<std::uint32_t>(L.prompt.size()), begin + chunk);
    const std::int32_t T      = static_cast<std::int32_t>(end - begin);
    if (T <= 0) { throw std::logic_error("muse prefill advanced with no tokens"); }

    auto scope          = impl_->workspace->scope();
    Tensor hidden       = impl_->workspace->alloc(DType::BF16, {TextConfig::hidden, T});
    Tensor positions    = impl_->workspace->alloc(DType::I32, {T, 1});
    std::vector<TokenId> ids(L.prompt.begin() + begin, L.prompt.begin() + end);
    impl_->embed_tokens(ids, hidden);
    std::vector<std::int32_t> pos(static_cast<std::size_t>(T));
    for (std::int32_t i = 0; i < T; ++i) { pos[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(begin + i); }
    CUDA_CHECK(cudaMemcpyAsync(positions.data, pos.data(), pos.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->stream));
    impl_->set_table_row(static_cast<std::int32_t>(lane));
    Tensor* capture = nullptr;
    Tensor capture_storage;
    if (impl_->dflash_enabled) {
        capture_storage = impl_->workspace->alloc(DType::BF16, {DFlash2Config::captured_rows, T});
        capture         = &capture_storage;
    }
    impl_->run_layers(hidden, positions, T, 1, capture);
    if (impl_->dflash_enabled) {
        impl_->project_and_append_dflash(capture_storage, positions, T,
                                         static_cast<std::int32_t>(lane));
    }
    L.processed = end;

    runtime::PrefillStepResult result;
    result.processed_prompt_tokens = static_cast<std::uint32_t>(T);
    result.summary.prompt_tokens   = static_cast<std::uint32_t>(L.prompt.size());
    result.summary.reused_prompt_tokens = L.reuse_base;
    result.summary.prefix_reuse_path =
        L.reuse_base == 0 ? PrefixReusePath::FullReset : PrefixReusePath::AppendAtFrontier;
    result.complete                = L.processed == L.prompt.size();
    if (result.complete) {
        impl_->device->synchronize();
        L.seq_len = L.processed;
        L.sampled = impl_->sample_from_hidden(hidden, T, L.seq_len, L.sampling);
        L.ledger.assign(L.prompt.begin(), L.prompt.end());
        L.pending.kind        = Impl::PendingKind::Begin;
        L.pending.base_seq    = L.seq_len;
        L.pending.produced    = 1;
        L.pending.licensed[0] = L.sampled;
        L.timings.prefill_seconds =
            std::chrono::duration<double>(Clock::now() - L.prefill_started).count();
        result.round.tokens = std::span<const TokenId>(&L.sampled, 1);
    }
    return result;
}

runtime::BatchedGeneratedRound
Program::decode_batch(std::span<const std::uint32_t> lanes,
                      std::span<const runtime::RoundBudget> budgets) {
    if (lanes.empty() || lanes.size() > impl_->plan.max_concurrency() ||
        budgets.size() != lanes.size()) {
        throw std::invalid_argument("Muse decode batch has invalid membership");
    }
    const std::uint32_t stride =
        impl_->dflash_enabled ? static_cast<std::uint32_t>(DFlash2Config::block_size) : 1U;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (lanes[row] >= impl_->plan.max_concurrency()) {
            throw std::out_of_range("Muse decode lane is out of range");
        }
        Impl::Lane& L = impl_->lanes[lanes[row]];
        if (!L.busy || L.retained || L.pending.kind != Impl::PendingKind::None) {
            throw std::logic_error("Muse decode lane is not active");
        }
    }

    if (!impl_->dflash_enabled) {
        const auto started = Clock::now();
        Impl::DecodeIngress& in = impl_->ingress();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const Impl::Lane& L      = impl_->lanes[lanes[row]];
            in.tokens[row]           = L.sampled;
            in.positions[row]        = static_cast<std::int32_t>(L.seq_len);
            in.table_rows[row]       = static_cast<std::int32_t>(lanes[row]);
            in.sample_positions[row] = static_cast<std::int32_t>(L.seq_len + 1U);
            in.sampling[row]         = L.sampling;
            in.sampled[row]          = 0;
        }
        if (impl_->graphs_ready) {
            DecodeGraphExecutable& executable = impl_->decode_execs[lanes.size() - 1U];
            if (!executable.ready()) {
                throw std::logic_error("Muse exact-batch CUDA Graph executable is unavailable");
            }
            executable.launch(impl_->stream);
        } else {
            impl_->decode_ordinary_body(static_cast<std::int32_t>(lanes.size()));
        }
        CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
        const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            Impl::Lane& L = impl_->lanes[lanes[row]];
            const std::uint32_t base_seq = L.seq_len;
            L.seq_len += 1;
            L.sampled = in.sampled[row];
            impl_->decode_tokens[row] = L.sampled;
            impl_->decode_row_counts[row] = 1;
            L.pending.kind        = Impl::PendingKind::Ordinary;
            L.pending.base_seq    = base_seq;
            L.pending.produced    = 1;
            L.pending.licensed[0] = L.sampled;
            L.timings.decode_seconds += elapsed;
        }
        runtime::BatchedGeneratedRound round;
        round.tokens     = std::span<const TokenId>(impl_->decode_tokens.data(), lanes.size());
        round.row_stride = 1;
        return round;
    }

    impl_->decode_dflash_batch(lanes, budgets);
    runtime::BatchedGeneratedRound round;
    round.tokens     = std::span<const TokenId>(impl_->decode_tokens.data(),
                                                lanes.size() * stride);
    round.row_stride = stride;
    if (stride > 1) {
        round.row_counts =
            std::span<const std::int32_t>(impl_->decode_row_counts.data(), lanes.size());
    }
    return round;
}

void Program::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= impl_->plan.max_concurrency()) {
        throw std::out_of_range("Muse prefill resolution lane is out of range");
    }
    Impl::Lane& L = impl_->lanes[lane];
    if (!L.busy || L.pending.kind != Impl::PendingKind::Begin || L.pending.produced != 1) {
        throw std::logic_error("Muse prefill resolution has no pending first token");
    }
    L.ledger.push_back(L.pending.licensed[0]);
    L.sampled = L.pending.licensed[0];
    L.pending = {};
    if (terminal) {
        L.kv.trim_tokens(L.seq_len);
        L.kv.cancel_unmapped_entitlement();
        L.kv.unbind_row();
        L.prompt.clear();
        L.processed       = 0;
        L.requested       = 0;
        L.reuse_base      = 0;
        L.busy            = false;
        L.retained        = true;
        L.prefix_reusable = true;
    }
}

void Program::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                    std::span<const std::uint32_t> accepted_tokens,
                                    std::span<const std::uint8_t> terminal,
                                    std::span<const std::uint8_t> cancelled) {
    if (lanes.empty() || lanes.size() > impl_->plan.max_concurrency() ||
        accepted_tokens.size() != lanes.size() || terminal.size() != lanes.size() ||
        cancelled.size() != lanes.size()) {
        throw std::invalid_argument("Muse pending resolution has invalid membership");
    }
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= impl_->plan.max_concurrency()) {
            throw std::out_of_range("Muse pending resolution lane is out of range");
        }
        Impl::Lane& L = impl_->lanes[lane];
        if (!L.busy || (L.pending.kind != Impl::PendingKind::Ordinary &&
                        L.pending.kind != Impl::PendingKind::Speculative)) {
            throw std::logic_error("Muse lane has no pending decode round");
        }
        if (cancelled[row]) {
            if (accepted_tokens[row] != 0) {
                throw std::logic_error("cancelled Muse round committed output");
            }
            L = Impl::Lane{};
            continue;
        }
        const std::uint32_t accepted = accepted_tokens[row];
        if (accepted == 0 || accepted > L.pending.produced ||
            (!terminal[row] && accepted != L.pending.produced)) {
            throw std::logic_error("Muse round committed an invalid licensed prefix");
        }
        L.seq_len = L.pending.base_seq + accepted;
        L.ledger.insert(L.ledger.end(), L.pending.licensed.begin(),
                        L.pending.licensed.begin() + accepted);
        L.sampled = L.pending.licensed[accepted - 1U];
        const bool retained_state_is_exact =
            !impl_->dflash_enabled || accepted == L.pending.produced;
        L.pending = {};
        if (terminal[row]) {
            L.kv.trim_tokens(L.seq_len);
            L.kv.cancel_unmapped_entitlement();
            L.kv.unbind_row();
            L.prompt.clear();
            L.processed  = 0;
            L.requested  = 0;
            L.reuse_base = 0;
            L.busy       = false;
            // DFlash appends the licensed round before output policy resolves it. If a stop or
            // tool boundary commits only a prefix, later speculative writes can already have
            // overwritten cyclic slots needed at the shorter frontier. Do not advertise that
            // state as an exact reusable prefix.
            L.retained        = true;
            L.prefix_reusable = retained_state_is_exact;
        }
    }
}

void Program::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= impl_->lanes.size()) { return; }
    impl_->lanes[lane] = {};
}

bool Program::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->plan.max_concurrency() && impl_->lanes[lane].retained;
}

void Program::evict_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return; }
    impl_->lanes[lane] = {};
}

GenerationTimings Program::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->plan.max_concurrency() ? impl_->lanes[lane].timings
                                                : GenerationTimings{};
}

SpeculativeStats Program::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < impl_->plan.max_concurrency() ? impl_->lanes[lane].spec_stats
                                                : SpeculativeStats{};
}

MemorySummary Program::memory_summary() const noexcept {
    MemorySummary out               = impl_->memory;
    out.weights = ArenaMemorySummary{impl_->model->weights_arena->capacity(),
                                     impl_->model->weights_arena->used(),
                                     impl_->model->weights_arena->peak_used()};
    out.workspace.used_bytes        = impl_->workspace->used();
    out.workspace.peak_used_bytes   = impl_->workspace->peak_used();
    out.workspace_logical_peak_bytes = impl_->workspace->peak_used();
    return out;
}

void Program::reset_memory_peaks() noexcept {
    impl_->model->weights_arena->reset_peak();
    impl_->workspace->reset_peak();
}

} // namespace ginfer::targets::muse_glimmer_30b::detail
