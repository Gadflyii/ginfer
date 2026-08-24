#include "ginfer/ops/grouped_dynamic_conv.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ginfer;
using namespace ginfer::test;

namespace {

constexpr int kHidden      = 12;
constexpr int kWidth       = 3;
constexpr int kBatch       = 3;
constexpr int kGroupSize   = 3;
constexpr int kGroups      = kHidden / kGroupSize;
constexpr int kKernel      = 2;
constexpr int kStages      = 2;
constexpr int kPack        = kKernel * kGroups;
constexpr int kDynamicRows = kStages * kPack;
constexpr int kColumns     = kWidth * kBatch;

static_assert(kBatch >= 2);
static_assert(kHidden % kGroupSize == 0);

std::size_t hidden_index(int h, int column, int batch) {
    return static_cast<std::size_t>(h) +
           static_cast<std::size_t>(kHidden) * (column + kWidth * batch);
}

std::size_t projection_index(int row, int h) {
    return static_cast<std::size_t>(h) + static_cast<std::size_t>(kHidden) * row;
}

std::size_t base_index(int stage, int tap, int h) {
    return static_cast<std::size_t>(h) +
           static_cast<std::size_t>(kHidden) * (tap + kKernel * stage);
}

std::size_t dynamic_index(int row, int column, int batch) {
    return static_cast<std::size_t>(row) +
           static_cast<std::size_t>(kDynamicRows) * (column + kWidth * batch);
}

// Direct round-to-nearest-even conversion from the FP64 oracle value to BF16. The cases remain
// in the normal finite range, so this deliberately does not share the production FP32 path.
std::uint16_t fp64_to_bf16_rne(double value) {
    if (!std::isfinite(value)) { throw std::invalid_argument("oracle requires a finite value"); }
    const bool negative = std::signbit(value);
    double magnitude    = std::abs(value);
    if (magnitude == 0.0) { return negative ? UINT16_C(0x8000) : UINT16_C(0); }

    int exponent             = 0;
    const double significand = std::frexp(magnitude, &exponent) * 2.0;
    int unbiased_exponent    = exponent - 1;
    if (unbiased_exponent <= -127 || unbiased_exponent >= 128) {
        throw std::out_of_range("oracle BF16 value is outside the normal finite range");
    }

    const double scaled = (significand - 1.0) * 128.0;
    const auto lower     = static_cast<std::uint32_t>(std::floor(scaled));
    const double tail    = scaled - static_cast<double>(lower);
    std::uint32_t fraction = lower;
    if (tail > 0.5 || (tail == 0.5 && (fraction & 1u) != 0u)) { ++fraction; }
    if (fraction == 128u) {
        fraction = 0;
        ++unbiased_exponent;
    }
    if (unbiased_exponent >= 128) {
        throw std::out_of_range("oracle BF16 rounding overflowed");
    }

    return static_cast<std::uint16_t>((negative ? UINT16_C(0x8000) : UINT16_C(0)) |
                                      ((unbiased_exponent + 127) << 7) | fraction);
}

double represented(std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); }

std::vector<std::uint16_t> make_prepare_hidden() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kHidden) * kColumns);
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int column = 0; column < kWidth; ++column) {
            for (int h = 0; h < kHidden; ++h) {
                double value = 0.0;
                if (column == 0) {
                    value = -0.5 + static_cast<double>(h % 5) / 16.0 + batch / 8.0;
                } else if (column == kWidth - 1) {
                    value = 1.5 + static_cast<double>(h % 4) / 8.0 + batch / 4.0;
                } else {
                    value = static_cast<double>((h * 3 + batch * 5 + column) % 9 - 4) / 8.0;
                }
                result[hidden_index(h, column, batch)] = fp64_to_bf16_rne(value);
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_finish_hidden() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kHidden) * kColumns);
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int column = 0; column < kWidth; ++column) {
            for (int h = 0; h < kHidden; ++h) {
                double value = 0.0;
                if (column == 0) {
                    value = 0.375 + static_cast<double>(h % 7) / 32.0 + batch / 16.0;
                } else if (column == kWidth - 1) {
                    value = -1.25 - static_cast<double>(h % 3) / 8.0 - batch / 4.0;
                } else {
                    value = static_cast<double>((h * 5 + batch * 2 + column) % 11 - 5) / 16.0;
                }
                result[hidden_index(h, column, batch)] = fp64_to_bf16_rne(value);
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_projection() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kDynamicRows) * kHidden);
    for (int row = 0; row < kDynamicRows; ++row) {
        for (int h = 0; h < kHidden; ++h) {
            const double value = static_cast<double>((row * 7 + h * 3) % 15 - 7) / 128.0;
            result[projection_index(row, h)] = fp64_to_bf16_rne(value);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_base() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kStages) * kKernel * kHidden);
    for (int stage = 0; stage < kStages; ++stage) {
        for (int tap = 0; tap < kKernel; ++tap) {
            for (int h = 0; h < kHidden; ++h) {
                const double value =
                    tap == 1 ? (stage == 0 ? 1.125 : -1.25) + (h % 4) / 32.0
                             : static_cast<double>((stage * 5 + h * 2) % 9 - 4) / 16.0;
                result[base_index(stage, tap, h)] = fp64_to_bf16_rne(value);
            }
        }
    }
    return result;
}

std::vector<std::uint16_t>
project_dynamic(const std::vector<std::uint16_t>& hidden,
                const std::vector<std::uint16_t>& projection) {
    std::vector<std::uint16_t> dynamic(static_cast<std::size_t>(kDynamicRows) * kColumns);
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int column = 0; column < kWidth; ++column) {
            for (int row = 0; row < kDynamicRows; ++row) {
                double dot = 0.0;
                for (int h = 0; h < kHidden; ++h) {
                    dot += represented(projection[projection_index(row, h)]) *
                           represented(hidden[hidden_index(h, column, batch)]);
                }
                // This is an observable semantic boundary: finish consumes the represented BF16
                // dynamic tensor, not the unquantized projection sum.
                dynamic[dynamic_index(row, column, batch)] = fp64_to_bf16_rne(dot);
            }
        }
    }
    return dynamic;
}

std::uint16_t conv_value(const std::vector<std::uint16_t>& hidden,
                         const std::vector<std::uint16_t>& dynamic,
                         const std::vector<std::uint16_t>& base, int stage, int h, int column,
                         int batch, bool leak_previous_batch_tail) {
    const int group = h / kGroupSize;
    double sum      = 0.0;
    for (int tap = 0; tap < kKernel; ++tap) {
        double x = 0.0;
        if (column >= tap) {
            x = represented(hidden[hidden_index(h, column - tap, batch)]);
        } else if (leak_previous_batch_tail && batch > 0 && tap == 1) {
            x = represented(hidden[hidden_index(h, kWidth - 1, batch - 1)]);
        }
        const int dynamic_row = stage * kPack + tap * kGroups + group;
        const double weight = represented(base[base_index(stage, tap, h)]) +
                              represented(dynamic[dynamic_index(dynamic_row, column, batch)]);
        sum += weight * x;
    }
    return fp64_to_bf16_rne(sum);
}

std::vector<std::uint16_t> conv_oracle(const std::vector<std::uint16_t>& hidden,
                                       const std::vector<std::uint16_t>& dynamic,
                                       const std::vector<std::uint16_t>& base, int stage) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kHidden) * kColumns);
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int column = 0; column < kWidth; ++column) {
            for (int h = 0; h < kHidden; ++h) {
                out[hidden_index(h, column, batch)] =
                    conv_value(hidden, dynamic, base, stage, h, column, batch, false);
            }
        }
    }
    return out;
}

int verify_boundary_sensitivity(const std::vector<std::uint16_t>& hidden,
                                const std::vector<std::uint16_t>& dynamic,
                                const std::vector<std::uint16_t>& base, int stage) {
    for (int batch = 1; batch < kBatch; ++batch) {
        bool detects_leak = false;
        for (int h = 0; h < kHidden; ++h) {
            const auto causal = conv_value(hidden, dynamic, base, stage, h, 0, batch, false);
            const auto leaked = conv_value(hidden, dynamic, base, stage, h, 0, batch, true);
            detects_leak |= causal != leaked;
        }
        if (!detects_leak) {
            std::cerr << "stage " << stage << " batch " << batch
                      << " does not detect a cross-row tap-one leak\n";
            return 1;
        }
    }
    return 0;
}

void copy_bits(GuardedDeviceBuffer& device, const std::vector<std::uint16_t>& bits) {
    device.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
}

std::vector<std::uint16_t> read_bits(const GuardedDeviceBuffer& device, std::size_t count) {
    std::vector<std::uint16_t> result(count);
    device.copy_to_host(result.data(), result.size() * sizeof(std::uint16_t));
    return result;
}

int run_case() {
    const auto prepare_hidden = make_prepare_hidden();
    const auto finish_hidden  = make_finish_hidden();
    const auto projection     = make_projection();
    const auto base           = make_base();
    const auto dynamic        = project_dynamic(prepare_hidden, projection);
    const auto prepare_out    = conv_oracle(prepare_hidden, dynamic, base, 0);
    const auto finish_out     = conv_oracle(finish_hidden, dynamic, base, 1);

    int failures = 0;
    failures += verify_boundary_sensitivity(prepare_hidden, dynamic, base, 0);
    failures += verify_boundary_sensitivity(finish_hidden, dynamic, base, 1);

    GuardedDeviceBuffer d_prepare_hidden(prepare_hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_finish_hidden(finish_hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_projection(projection.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_base(base.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_dynamic(dynamic.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_prepare_out(prepare_out.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_finish_out(finish_out.size() * sizeof(std::uint16_t));
    copy_bits(d_prepare_hidden, prepare_hidden);
    copy_bits(d_finish_hidden, finish_hidden);
    copy_bits(d_projection, projection);
    copy_bits(d_base, base);
    d_dynamic.fill(0x6d);
    d_prepare_out.fill(0x7e);
    d_finish_out.fill(0x8f);

    Tensor prepare_hidden_tensor(d_prepare_hidden.data(), DType::BF16,
                                 {kHidden, kWidth, kBatch});
    Tensor finish_hidden_tensor(d_finish_hidden.data(), DType::BF16,
                                {kHidden, kWidth, kBatch});
    Tensor projection_tensor(d_projection.data(), DType::BF16, {kHidden, kDynamicRows});
    Tensor base_tensor(d_base.data(), DType::BF16, {kHidden, kKernel, kStages});
    Tensor dynamic_tensor(d_dynamic.data(), DType::BF16, {kDynamicRows, kWidth, kBatch});
    Tensor prepare_out_tensor(d_prepare_out.data(), DType::BF16, {kHidden, kWidth, kBatch});
    Tensor finish_out_tensor(d_finish_out.data(), DType::BF16, {kHidden, kWidth, kBatch});

    ops::grouped_dynamic_conv_prepare(prepare_hidden_tensor, projection_tensor, base_tensor,
                                      dynamic_tensor, prepare_out_tensor, kGroupSize, nullptr);
    cuda_synchronize();

    failures += verify_exact("prepare dynamic BF16 boundary",
                             read_bits(d_dynamic, dynamic.size()), dynamic);
    failures += verify_exact("prepare output", read_bits(d_prepare_out, prepare_out.size()),
                             prepare_out);
    failures += verify_exact("prepare hidden read-only",
                             read_bits(d_prepare_hidden, prepare_hidden.size()), prepare_hidden);
    failures += verify_exact("prepare projection read-only",
                             read_bits(d_projection, projection.size()), projection);
    failures += verify_exact("prepare base read-only", read_bits(d_base, base.size()), base);

    ops::grouped_dynamic_conv_finish(finish_hidden_tensor, dynamic_tensor, base_tensor,
                                     finish_out_tensor, kGroupSize, nullptr);
    cuda_synchronize();

    failures += verify_exact("finish output", read_bits(d_finish_out, finish_out.size()),
                             finish_out);
    failures += verify_exact("finish hidden read-only",
                             read_bits(d_finish_hidden, finish_hidden.size()), finish_hidden);
    failures += verify_exact("finish dynamic read-only", read_bits(d_dynamic, dynamic.size()),
                             dynamic);
    failures += verify_exact("finish base read-only", read_bits(d_base, base.size()), base);

    failures += d_prepare_hidden.verify_guards("prepare hidden");
    failures += d_finish_hidden.verify_guards("finish hidden");
    failures += d_projection.verify_guards("projection");
    failures += d_base.verify_guards("base");
    failures += d_dynamic.verify_guards("dynamic");
    failures += d_prepare_out.verify_guards("prepare output");
    failures += d_finish_out.verify_guards("finish output");
    return failures;
}

constexpr int kMuseHidden      = 6656;
constexpr int kMuseWidth       = 15;
constexpr int kMuseBatch       = 1;
constexpr int kMuseColumns     = kMuseWidth * kMuseBatch;
constexpr int kMuseGroupSize   = 16;
constexpr int kMuseGroups      = kMuseHidden / kMuseGroupSize;
constexpr int kMuseDynamicRows = 4 * kMuseGroups;

std::size_t muse_hidden_index(int h, int column) {
    return static_cast<std::size_t>(column) * kMuseHidden + h;
}

std::size_t muse_projection_index(int row, int h) {
    return static_cast<std::size_t>(row) * kMuseHidden + h;
}

std::size_t muse_dynamic_index(int row, int column) {
    return static_cast<std::size_t>(column) * kMuseDynamicRows + row;
}

std::vector<std::uint16_t> make_muse_hidden() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kMuseHidden) * kMuseColumns);
    for (int column = 0; column < kMuseColumns; ++column) {
        for (int h = 0; h < kMuseHidden; ++h) {
            const int centered = ((h * 29 + column * 71 + 17) & 0xff) - 128;
            result[muse_hidden_index(h, column)] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<std::uint16_t> make_muse_projection() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kMuseDynamicRows) * kMuseHidden);
    for (int row = 0; row < kMuseDynamicRows; ++row) {
        for (int h = 0; h < kMuseHidden; ++h) {
            const int centered = ((row * 43 + h * 19 + 11) & 0xff) - 128;
            result[muse_projection_index(row, h)] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 4096.0F));
        }
    }
    return result;
}

std::vector<std::uint16_t> make_muse_base() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(4) * kMuseHidden);
    for (int stage = 0; stage < 2; ++stage) {
        for (int tap = 0; tap < 2; ++tap) {
            for (int h = 0; h < kMuseHidden; ++h) {
                const int centered = ((stage * 31 + tap * 17 + h * 7) & 0x3f) - 32;
                result[static_cast<std::size_t>(h) +
                       static_cast<std::size_t>(kMuseHidden) * (tap + 2 * stage)] =
                    f32_to_bf16(static_cast<float>(centered) * (1.0F / 256.0F));
            }
        }
    }
    return result;
}

double muse_projection_oracle(const std::vector<std::uint16_t>& hidden,
                              const std::vector<std::uint16_t>& projection, int row, int column) {
    double result = 0.0;
    for (int h = 0; h < kMuseHidden; ++h) {
        result += represented(projection[muse_projection_index(row, h)]) *
                  represented(hidden[muse_hidden_index(h, column)]);
    }
    return result;
}

std::vector<float> muse_hidden_fp32(const std::vector<std::uint16_t>& hidden) {
    std::vector<float> result(hidden.size());
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        result[index] = bf16_to_f32(hidden[index]);
    }
    return result;
}

double muse_q4_projection_oracle(const std::vector<float>& hidden,
                                 const quantized_weight::PackedWeight& projection, int row,
                                 int column) {
    return quantized_weight::dot_fp64(
        projection, row,
        hidden.data() + static_cast<std::size_t>(column) * kMuseHidden, kMuseHidden);
}

int run_muse_geometry_case() {
    const auto hidden     = make_muse_hidden();
    const auto projection = make_muse_projection();
    const auto base       = make_muse_base();

    GuardedDeviceBuffer d_hidden(hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_projection(projection.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_base(base.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_dynamic(static_cast<std::size_t>(kMuseDynamicRows) * kMuseColumns *
                                  sizeof(std::uint16_t));
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(kMuseHidden) * kMuseColumns *
                              sizeof(std::uint16_t));
    copy_bits(d_hidden, hidden);
    copy_bits(d_projection, projection);
    copy_bits(d_base, base);
    d_dynamic.fill(0xff);
    d_out.fill(0xff);

    Tensor hidden_tensor(d_hidden.data(), DType::BF16, {kMuseHidden, kMuseWidth, kMuseBatch});
    Tensor projection_tensor(d_projection.data(), DType::BF16,
                             {kMuseHidden, kMuseDynamicRows});
    Tensor base_tensor(d_base.data(), DType::BF16, {kMuseHidden, 2, 2});
    Tensor dynamic_tensor(d_dynamic.data(), DType::BF16,
                          {kMuseDynamicRows, kMuseWidth, kMuseBatch});
    Tensor out_tensor(d_out.data(), DType::BF16, {kMuseHidden, kMuseWidth, kMuseBatch});
    ops::grouped_dynamic_conv_prepare(hidden_tensor, projection_tensor, base_tensor, dynamic_tensor,
                                      out_tensor, kMuseGroupSize, nullptr);
    cuda_synchronize();

    int failures = 0;
    failures += d_hidden.verify_guards("Muse hidden");
    failures += d_projection.verify_guards("Muse projection");
    failures += d_base.verify_guards("Muse base");
    failures += d_dynamic.verify_guards("Muse dynamic");
    failures += d_out.verify_guards("Muse output");

    const std::vector<std::uint16_t> actual_dynamic =
        read_bits(d_dynamic, static_cast<std::size_t>(kMuseDynamicRows) * kMuseColumns);
    const std::vector<std::uint16_t> actual_out =
        read_bits(d_out, static_cast<std::size_t>(kMuseHidden) * kMuseColumns);
    for (const auto bits : actual_dynamic) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "Muse dynamic projection did not fully overwrite its output\n";
            ++failures;
            break;
        }
    }
    for (const auto bits : actual_out) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "Muse grouped conv did not fully overwrite its output\n";
            ++failures;
            break;
        }
    }

    const std::vector<int> sampled_rows{0, 1, 415, 416, 831, 832, 833, 1247, 1248, 1662, 1663};
    std::vector<double> projection_actual;
    std::vector<double> projection_expected;
    projection_actual.reserve(sampled_rows.size() * kMuseColumns);
    projection_expected.reserve(projection_actual.capacity());
    for (const int row : sampled_rows) {
        for (int column = 0; column < kMuseColumns; ++column) {
            projection_actual.push_back(
                represented(actual_dynamic[muse_dynamic_index(row, column)]));
            projection_expected.push_back(muse_projection_oracle(hidden, projection, row, column));
        }
    }
    constexpr ReductionCriterion kMuseProjectionTolerance{1.0 / 256.0, 1.0 / 256.0,
                                                           2.0 / 256.0};
    failures += verify_reduction("Muse grouped dynamic projection FP64 oracle", projection_actual,
                                 projection_expected, kMuseProjectionTolerance);

    const std::vector<int> sampled_hidden{0, 1, 15, 16, 2048, 4095, 6640, 6654, 6655};
    std::vector<double> conv_actual;
    std::vector<double> conv_expected;
    conv_actual.reserve(sampled_hidden.size() * kMuseColumns);
    conv_expected.reserve(conv_actual.capacity());
    for (const int h : sampled_hidden) {
        const int group = h / kMuseGroupSize;
        for (int column = 0; column < kMuseColumns; ++column) {
            double expected = 0.0;
            for (int tap = 0; tap < 2; ++tap) {
                if (column < tap) { continue; }
                const double x = represented(hidden[muse_hidden_index(h, column - tap)]);
                const double base_weight = represented(
                    base[static_cast<std::size_t>(h) +
                         static_cast<std::size_t>(kMuseHidden) * tap]);
                const int dynamic_row = tap * kMuseGroups + group;
                const double projected = muse_projection_oracle(hidden, projection, dynamic_row,
                                                                 column);
                const double dynamic_weight =
                    represented(fp64_to_bf16_rne(projected));
                expected += (base_weight + dynamic_weight) * x;
            }
            conv_actual.push_back(represented(actual_out[muse_hidden_index(h, column)]));
            conv_expected.push_back(expected);
        }
    }
    constexpr ReductionCriterion kMuseConvTolerance{1.0 / 128.0, 1.0 / 256.0, 2.0 / 256.0};
    failures += verify_reduction("Muse grouped conv FP64 oracle", conv_actual, conv_expected,
                                 kMuseConvTolerance);
    return failures;
}

int run_muse_q4_geometry_case() {
    const auto hidden = make_muse_hidden();
    const auto hidden_fp32 = muse_hidden_fp32(hidden);
    const auto base = make_muse_base();
    const auto projection = quantized_weight::make_patterned_weight(
        QType::Q4G64_F16S, kMuseDynamicRows, kMuseHidden, 1801U,
        {quantized_weight::RowSplitScalePattern::Small,
         quantized_weight::RowSplitCodePattern::Hashed});

    GuardedDeviceBuffer d_hidden(hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_projection(projection.payload.size());
    GuardedDeviceBuffer d_base(base.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_dynamic(static_cast<std::size_t>(kMuseDynamicRows) * kMuseColumns *
                                  sizeof(std::uint16_t));
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(kMuseHidden) * kMuseColumns *
                              sizeof(std::uint16_t));
    copy_bits(d_hidden, hidden);
    d_projection.copy_from_host(projection.payload.data(), projection.payload.size());
    copy_bits(d_base, base);
    d_dynamic.fill(0xff);
    d_out.fill(0xff);

    Tensor hidden_tensor(d_hidden.data(), DType::BF16, {kMuseHidden, kMuseWidth, kMuseBatch});
    const Weight projection_weight = projection.device_weight(d_projection.data());
    Tensor base_tensor(d_base.data(), DType::BF16, {kMuseHidden, 2, 2});
    Tensor dynamic_tensor(d_dynamic.data(), DType::BF16,
                          {kMuseDynamicRows, kMuseWidth, kMuseBatch});
    Tensor out_tensor(d_out.data(), DType::BF16, {kMuseHidden, kMuseWidth, kMuseBatch});
    ops::grouped_dynamic_conv_prepare(hidden_tensor, projection_weight, base_tensor, dynamic_tensor,
                                      out_tensor, kMuseGroupSize, nullptr);
    cuda_synchronize();

    int failures = 0;
    failures += d_hidden.verify_guards("Muse Q4 hidden");
    failures += d_projection.verify_guards("Muse Q4 projection");
    failures += d_base.verify_guards("Muse Q4 base");
    failures += d_dynamic.verify_guards("Muse Q4 dynamic");
    failures += d_out.verify_guards("Muse Q4 output");

    const std::vector<std::uint16_t> actual_dynamic =
        read_bits(d_dynamic, static_cast<std::size_t>(kMuseDynamicRows) * kMuseColumns);
    const std::vector<std::uint16_t> actual_out =
        read_bits(d_out, static_cast<std::size_t>(kMuseHidden) * kMuseColumns);
    for (const auto bits : actual_dynamic) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "Muse Q4 dynamic projection did not fully overwrite its output\n";
            ++failures;
            break;
        }
    }
    for (const auto bits : actual_out) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "Muse Q4 grouped conv did not fully overwrite its output\n";
            ++failures;
            break;
        }
    }

    const std::vector<int> sampled_rows{0, 1, 415, 416, 831, 832, 833, 1247, 1248, 1662, 1663};
    std::vector<double> projection_actual;
    std::vector<double> projection_expected;
    projection_actual.reserve(sampled_rows.size() * kMuseColumns);
    projection_expected.reserve(projection_actual.capacity());
    for (const int row : sampled_rows) {
        for (int column = 0; column < kMuseColumns; ++column) {
            projection_actual.push_back(
                represented(actual_dynamic[muse_dynamic_index(row, column)]));
            projection_expected.push_back(
                muse_q4_projection_oracle(hidden_fp32, projection, row, column));
        }
    }
    constexpr ReductionCriterion kMuseQ4ProjectionTolerance{1.0 / 256.0, 1.0 / 256.0,
                                                             2.0 / 256.0};
    failures += verify_reduction("Muse Q4 grouped dynamic projection FP64 oracle",
                                 projection_actual, projection_expected,
                                 kMuseQ4ProjectionTolerance);

    const std::vector<int> sampled_hidden{0, 1, 15, 16, 2048, 4095, 6640, 6654, 6655};
    std::vector<double> conv_actual;
    std::vector<double> conv_expected;
    conv_actual.reserve(sampled_hidden.size() * kMuseColumns);
    conv_expected.reserve(conv_actual.capacity());
    for (const int h : sampled_hidden) {
        const int group = h / kMuseGroupSize;
        for (int column = 0; column < kMuseColumns; ++column) {
            double expected = 0.0;
            for (int tap = 0; tap < 2; ++tap) {
                if (column < tap) { continue; }
                const double x = represented(hidden[muse_hidden_index(h, column - tap)]);
                const double base_weight = represented(
                    base[static_cast<std::size_t>(h) +
                         static_cast<std::size_t>(kMuseHidden) * tap]);
                const int dynamic_row = tap * kMuseGroups + group;
                const double projected =
                    muse_q4_projection_oracle(hidden_fp32, projection, dynamic_row, column);
                const double dynamic_weight = represented(fp64_to_bf16_rne(projected));
                expected += (base_weight + dynamic_weight) * x;
            }
            conv_actual.push_back(represented(actual_out[muse_hidden_index(h, column)]));
            conv_expected.push_back(expected);
        }
    }
    constexpr ReductionCriterion kMuseQ4ConvTolerance{1.0 / 128.0, 1.0 / 256.0,
                                                       2.0 / 256.0};
    failures += verify_reduction("Muse Q4 grouped conv FP64 oracle", conv_actual, conv_expected,
                                 kMuseQ4ConvTolerance);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const int failures = run_case() + run_muse_geometry_case() + run_muse_q4_geometry_case();
    if (failures == 0) {
        std::cout << "OK grouped_dynamic_conv B=" << kBatch << " W=" << kWidth << '\n';
    }
    return failures == 0 ? 0 : 1;
}
