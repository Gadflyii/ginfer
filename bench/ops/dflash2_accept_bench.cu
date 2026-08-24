// Qualification benchmark for the registered Muse DFlash2 acceptance geometry.
//
//   ./ginfer_dflash2_accept_bench --batch 1 --drafts 15
//   ./ginfer_dflash2_accept_bench --batch 8 --drafts 15
#include "core/device.h"
#include "core/tensor.h"
#include "ginfer/ops/dflash2_select.h"
#include "ginfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ginfer;
using namespace ginfer::bench;

namespace {

constexpr int kTokenDomain = 202048;
constexpr int kSelectorTopK = 16;

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int result   = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing characters"); }
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

struct Options {
    int batch  = 1;
    int drafts = 15;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto value = [&](const char* name) {
            if (++i >= argc) { throw std::invalid_argument(std::string(name) + " needs a value"); }
            return std::string_view(argv[i]);
        };
        if (arg == "--batch") {
            options.batch = parse_int(value("--batch"), "--batch");
        } else if (arg == "--drafts") {
            options.drafts = parse_int(value("--drafts"), "--drafts");
        } else if (arg == "-h" || arg == "--help") {
            std::printf("usage: %s [--batch 1..8] [--drafts 1..15]\n", argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.batch < 1 || options.batch > 8) {
        throw std::invalid_argument("--batch must be in [1,8]");
    }
    if (options.drafts < 1 || options.drafts > 15) {
        throw std::invalid_argument("--drafts must be in [1,15]");
    }
    return options;
}

template <class T>
DeviceBuffer to_device(const std::vector<T>& values) {
    DeviceBuffer out(values.size() * sizeof(T));
    out.copy_from_host(values.data(), out.bytes);
    return out;
}

void run(const Options& options) {
    const int columns = options.drafts + 1;
    const int lanes   = columns * options.batch;
    std::vector<std::uint16_t> host_logits(static_cast<std::size_t>(kTokenDomain) * lanes);
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(options.drafts) * options.batch);
    std::vector<std::int32_t> candidates(
        static_cast<std::size_t>(kSelectorTopK) * options.drafts * options.batch);
    std::vector<float> q_probs(candidates.size(), 0.0f);
    for (int batch = 0; batch < options.batch; ++batch) {
        for (int column = 0; column < columns; ++column) {
            const int hot = (17 + batch * 7919 + column * 4001) % kTokenDomain;
            const std::size_t base =
                static_cast<std::size_t>(batch * columns + column) * kTokenDomain;
            for (int token = 0; token < kTokenDomain; ++token) {
                float value = -8.0f + static_cast<float>((token * 17 + column * 31) % 4096) /
                                           4096.0f;
                if (token == hot) { value = 8.0f; }
                host_logits[base + static_cast<std::size_t>(token)] = f32_to_bf16(value);
            }
            if (column < options.drafts) {
                drafts[static_cast<std::size_t>(batch) * options.drafts + column] = hot;
                const std::size_t candidate_base =
                    (static_cast<std::size_t>(batch) * options.drafts + column) * kSelectorTopK;
                candidates[candidate_base] = hot;
                for (int rank = 1; rank < kSelectorTopK; ++rank) {
                    candidates[candidate_base + rank] = (hot + rank) % kTokenDomain;
                }
            }
        }
    }

    std::vector<ops::SamplingConfig> host_configs(static_cast<std::size_t>(options.batch));
    for (int batch = 0; batch < options.batch; ++batch) {
        auto& config       = host_configs[static_cast<std::size_t>(batch)];
        config.temperature = 0.7f;
        config.top_k       = 64;
        config.top_p       = 0.95f;
        config.seed        = 20260820ull + static_cast<unsigned long long>(batch);
    }
    std::vector<std::int32_t> extents(static_cast<std::size_t>(options.batch), options.drafts);
    std::vector<std::int32_t> lengths(static_cast<std::size_t>(options.batch), 4096);
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(options.batch), -1);

    DeviceBuffer d_logits = to_device(host_logits);
    DeviceBuffer d_drafts = to_device(drafts);
    DeviceBuffer d_candidates = to_device(candidates);
    DeviceBuffer d_q_probs = to_device(q_probs);
    DeviceBuffer d_extents = to_device(extents);
    DeviceBuffer d_configs = to_device(host_configs);
    DeviceBuffer d_lengths = to_device(lengths);
    DeviceBuffer d_anchors = to_device(anchors);
    DeviceBuffer d_licensed(
        static_cast<std::size_t>(columns) * options.batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(options.batch) * sizeof(std::int32_t));
    DeviceBuffer d_accepted(static_cast<std::size_t>(options.batch) * sizeof(std::int32_t));

    Tensor t_logits(d_logits.p, DType::BF16, {kTokenDomain, columns, options.batch});
    Tensor t_drafts(d_drafts.p, DType::I32, {options.drafts, options.batch});
    Tensor t_candidates(d_candidates.p, DType::I32,
                        {kSelectorTopK, options.drafts, options.batch});
    Tensor t_q_probs(d_q_probs.p, DType::FP32,
                     {kSelectorTopK, options.drafts, options.batch});
    Tensor t_extents(d_extents.p, DType::I32, {options.batch});
    Tensor t_lengths(d_lengths.p, DType::I32, {options.batch});
    Tensor t_anchors(d_anchors.p, DType::I32, {options.batch});
    Tensor t_licensed(d_licensed.p, DType::I32, {columns, options.batch});
    Tensor t_counts(d_counts.p, DType::I32, {options.batch});
    Tensor t_accepted(d_accepted.p, DType::I32, {options.batch});
    const auto* config_ptr = static_cast<const ops::SamplingConfig*>(d_configs.p);
    const std::size_t workspace_bytes = ops::dflash2_accept_workspace_capacity_bytes(
        kTokenDomain, options.drafts, options.drafts, options.batch, options.batch);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    const double input_bytes = static_cast<double>(kTokenDomain) * lanes * sizeof(std::uint16_t);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::dflash2_accept(t_logits, t_drafts, t_candidates, t_q_probs, t_extents,
                                config_ptr, kTokenDomain, t_lengths, t_anchors, t_licensed,
                                t_counts, t_accepted, workspace, stream);
        },
        input_bytes, 5, 30, 300);
    const std::string label = "DFlash2 accept K=" + std::to_string(options.drafts) +
                              " B=" + std::to_string(options.batch);
    std::printf("payload: %.2f MiB, all drafts accepted, workspace: %.2f MiB\n",
                input_bytes / 1048576.0, static_cast<double>(workspace_bytes) / 1048576.0);
    print_result(label.c_str(), result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        run(parse_args(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ginfer_dflash2_accept_bench: %s\n", error.what());
        return 2;
    }
}
