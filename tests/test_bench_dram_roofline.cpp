#include "ginfer_bench_common.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string_view>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int check_spec(std::string_view name, bool known, double spec_gbs, const char* label) {
    const auto roof = ginfer::bench::dram_roofline_for_gpu_name(name);
    if (roof.known != known) {
        std::cerr << label << " known=" << roof.known << " expected " << known << '\n';
        return 1;
    }
    if (!known) { return 0; }
    if (roof.spec_gbs != spec_gbs) {
        std::cerr << label << " spec_gbs=" << roof.spec_gbs << " expected " << spec_gbs << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    int failures = 0;

    failures += check_spec("NVIDIA GeForce RTX 5090", true, 1792.0, "5090");
    failures += check_spec("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", true, 1792.0,
                           "PRO 6000 Workstation");
    failures +=
        check_spec("NVIDIA RTX PRO 6000 Blackwell Server Edition", true, 1597.0, "PRO 6000 Server");
    failures += check_spec("NVIDIA GeForce RTX 4090", true, 1008.0, "4090");
    failures += check_spec("NVIDIA GeForce RTX 3090", true, 936.0, "3090");
    failures += check_spec("NVIDIA DGX Spark", false, 0.0, "unlisted Spark");
    failures += check_spec("Tesla V100-SXM2-32GB", false, 0.0, "unlisted V100");
    failures += check_spec("Some Server GPU", false, 0.0, "Server without PRO 6000");
    failures += check_spec("", false, 0.0, "empty name");

    char spec[32];
    ginfer::bench::format_dram_spec_gbs(
        spec, sizeof(spec), ginfer::bench::dram_roofline_for_gpu_name("NVIDIA GeForce RTX 5090"));
    failures += check(std::strcmp(spec, "1792.0") == 0, "5090 dram_spec_gbs print");

    char roof[32];
    ginfer::bench::format_dram_roof(
        roof, sizeof(roof), 896.0,
        ginfer::bench::dram_roofline_for_gpu_name("NVIDIA GeForce RTX 5090"));
    failures += check(std::strcmp(roof, "50.0% of 1792") == 0, "5090 dram roof percent");

    ginfer::bench::format_dram_spec_gbs(
        spec, sizeof(spec), ginfer::bench::dram_roofline_for_gpu_name("NVIDIA DGX Spark"));
    failures += check(std::strcmp(spec, "unknown") == 0, "unlisted dram_spec_gbs print");
    failures += check(std::strstr(spec, "1792") == nullptr, "unlisted spec must not print 1792");

    ginfer::bench::format_dram_roof(
        roof, sizeof(roof), 896.0, ginfer::bench::dram_roofline_for_gpu_name("NVIDIA DGX Spark"));
    failures += check(std::strcmp(roof, "unknown") == 0, "unlisted dram roof print");
    failures += check(std::strstr(roof, "1792") == nullptr, "unlisted roof must not print 1792");
    failures += check(std::strstr(roof, "%") == nullptr, "unlisted roof must not print a fake %");

    const auto unknown = ginfer::bench::dram_roofline_for_gpu_name("NVIDIA DGX Spark");
    failures += check(std::isnan(ginfer::bench::dram_spec_pct(896.0, unknown)),
                      "unlisted dram_spec_pct is NaN");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
