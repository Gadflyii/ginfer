#include "ginfer/ops/grouped_dynamic_conv.h"

#include "core/device.h"
#include "ginfer_bench_common.h"

#include <cuda_runtime.h>

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

constexpr std::int32_t kHidden      = 6656;
constexpr std::int32_t kDynamicRows = 1664;
constexpr std::int32_t kGroupSize   = 16;

struct Options {
    std::int32_t batch  = 1;
    std::int32_t width  = 15;
    std::int32_t layers = 10;
    bool q4             = false;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ginfer_grouped_dynamic_conv_bench "
                 "[--batch 1..8] [--width 1..16] [--layers 1..10] "
                 "[--format bf16|q4]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    try {
        std::size_t parsed = 0;
        const int value    = std::stoi(std::string(text), &parsed);
        if (parsed != text.size() || value < minimum || value > maximum) { usage(flag); }
        return value;
    } catch (const std::exception&) {
        usage(flag);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> std::string_view {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--batch") {
            options.batch = parse_i32(next("--batch"), 1, 8, "--batch");
        } else if (argument == "--width") {
            options.width = parse_i32(next("--width"), 1, 16, "--width");
        } else if (argument == "--layers") {
            options.layers = parse_i32(next("--layers"), 1, 10, "--layers");
        } else if (argument == "--format") {
            const std::string_view format = next("--format");
            if (format == "bf16") {
                options.q4 = false;
            } else if (format == "q4") {
                options.q4 = true;
            } else {
                usage("--format");
            }
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    return options;
}

constexpr std::size_t q4_projection_bytes() {
    constexpr std::size_t elements = static_cast<std::size_t>(kHidden) * kDynamicRows;
    return elements / 2 + elements / 64 * sizeof(std::uint16_t);
}

Weight q4_projection(void* payload) {
    constexpr std::size_t code_bytes =
        static_cast<std::size_t>(kHidden) * kDynamicRows / 2;
    Weight weight{};
    weight.qtype             = QType::Q4G64_F16S;
    weight.layout            = QuantLayout::RowSplit;
    weight.scale_dtype       = DType::FP16;
    weight.payload           = payload;
    weight.payload_bytes     = q4_projection_bytes();
    weight.high_plane_bytes  = 0;
    weight.qdata             = payload;
    weight.qhigh             = nullptr;
    weight.scales            = static_cast<std::uint8_t*>(payload) + code_bytes;
    weight.group_size        = 64;
    weight.group             = 64;
    weight.ndim              = 2;
    weight.shape[0]          = kDynamicRows;
    weight.shape[1]          = kHidden;
    weight.shape[2]          = 1;
    weight.shape[3]          = 1;
    weight.padded_shape[0]   = kDynamicRows;
    weight.padded_shape[1]   = kHidden;
    weight.padded_shape[2]   = 1;
    weight.padded_shape[3]   = 1;
    weight.n                 = kDynamicRows;
    weight.k                 = kHidden;
    return weight;
}

double useful_bytes(std::int32_t columns, std::int32_t layers, bool q4) {
    const double projection =
        q4 ? static_cast<double>(q4_projection_bytes())
           : static_cast<double>(kHidden) * kDynamicRows * sizeof(std::uint16_t);
    const double hidden = static_cast<double>(kHidden) * columns * sizeof(std::uint16_t);
    const double dynamic = static_cast<double>(kDynamicRows) * columns * sizeof(std::uint16_t);
    const double output = static_cast<double>(kHidden) * columns * sizeof(std::uint16_t);
    const double base = static_cast<double>(2) * kHidden * sizeof(std::uint16_t);
    return static_cast<double>(layers) *
           (projection + 3.0 * hidden + 3.0 * dynamic + base + output);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t columns = options.batch * options.width;

        DeviceBuffer hidden = make_bf16(static_cast<std::size_t>(kHidden) * columns);
        DeviceBuffer base = make_bf16(static_cast<std::size_t>(2) * 2 * kHidden);
        DeviceBuffer dynamic = make_zeros(static_cast<std::size_t>(kDynamicRows) * columns * 2);
        DeviceBuffer output = make_zeros(static_cast<std::size_t>(kHidden) * columns * 2);
        std::vector<DeviceBuffer> projections;
        projections.reserve(static_cast<std::size_t>(options.layers));
        for (int layer = 0; layer < options.layers; ++layer) {
            projections.push_back(options.q4
                                      ? make_zeros(q4_projection_bytes())
                                      : make_bf16(static_cast<std::size_t>(kHidden) *
                                                  kDynamicRows));
        }

        Tensor hidden_tensor(hidden.p, DType::BF16,
                             {kHidden, options.width, options.batch});
        Tensor base_tensor(base.p, DType::BF16, {kHidden, 2, 2});
        Tensor dynamic_tensor(dynamic.p, DType::BF16,
                              {kDynamicRows, options.width, options.batch});
        Tensor output_tensor(output.p, DType::BF16,
                             {kHidden, options.width, options.batch});

        const double bytes = useful_bytes(columns, options.layers, options.q4);
        const Result result = bench_loop(
            [&](cudaStream_t stream) {
                for (int layer = 0; layer < options.layers; ++layer) {
                    void* projection = projections[static_cast<std::size_t>(layer)].p;
                    if (options.q4) {
                        const Weight weight = q4_projection(projection);
                        ops::grouped_dynamic_conv_prepare(
                            hidden_tensor, weight, base_tensor, dynamic_tensor, output_tensor,
                            kGroupSize, stream);
                    } else {
                        Tensor tensor(projection, DType::BF16, {kHidden, kDynamicRows});
                        ops::grouped_dynamic_conv_prepare(
                            hidden_tensor, tensor, base_tensor, dynamic_tensor, output_tensor,
                            kGroupSize, stream);
                    }
                }
            },
            bytes, 10, 100, 1000);

        char label[96];
        std::snprintf(label, sizeof(label), "Muse grouped conv %s warm B=%d W=%d L=%d",
                      options.q4 ? "Q4" : "BF16", options.batch, options.width, options.layers);
        print_result(label, result);
        std::printf("per prepare=%8.2f us  columns=%d\n",
                    result.median_us / static_cast<double>(options.layers), columns);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
