#include "core/gpu_capability.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // 5090: 170 SMs
    auto occ = ginfer::make_occupancy_policy(ginfer::SmArch::Blackwell120, 170);
    failures += check(occ.sm_arch == ginfer::SmArch::Blackwell120, "5090 occupancy sm_arch");
    failures += check(occ.sm_count == 170, "5090 occupancy sm_count");
    failures += check(occ.rope_large_block_wave == 1020, "5090 rope_large_block_wave");
    failures += check(occ.gdn_output_target_ctas == 680, "5090 gdn_output_target_ctas");
    failures += check(occ.moe_prefill_persistent == 510, "5090 moe_prefill_persistent");
    failures += check(occ.gqa_decode_splits_base == 85, "5090 gqa_decode_splits_base");

    // PRO 6000: 188 SMs
    occ = ginfer::make_occupancy_policy(ginfer::SmArch::Blackwell120, 188);
    failures += check(occ.gqa_decode_splits_base == 94, "PRO 6000 gqa_decode_splits_base");

    // 4090 / 3090
    failures += check(ginfer::make_occupancy_policy(ginfer::SmArch::Ada89, 128)
                              .gqa_decode_splits_base == 64,
                      "4090 gqa_decode_splits_base");
    failures += check(ginfer::make_occupancy_policy(ginfer::SmArch::Ampere86, 82).sm_arch ==
                          ginfer::SmArch::Ampere86,
                      "3090 occupancy sm_arch");

    bool sm_count_rejected = false;
    try {
        (void)ginfer::make_occupancy_policy(ginfer::SmArch::Blackwell120, 0);
    } catch (const std::invalid_argument&) { sm_count_rejected = true; }
    failures += check(sm_count_rejected, "make_occupancy_policy accepted sm_count <= 0");

    sm_count_rejected = false;
    try {
        (void)ginfer::make_occupancy_policy(ginfer::SmArch::Blackwell120, -1);
    } catch (const std::invalid_argument&) { sm_count_rejected = true; }
    failures += check(sm_count_rejected, "make_occupancy_policy accepted negative sm_count");

    auto c120 = ginfer::capability_for_cc(120, 170, 32ull << 30);
    failures += check(c120.sm_arch == ginfer::SmArch::Blackwell120, "cc120 sm_arch");
    failures += check(c120.compute_capability == 120, "cc120 compute_capability");
    failures += check(c120.sm_count == 170, "cc120 sm_count");
    failures += check(c120.total_vram_bytes == (32ull << 30), "cc120 total_vram_bytes");
    failures += check(c120.nvfp4 && c120.pdl && c120.tma && c120.fp8_kind && !c120.fp8_ada,
                      "cc120 feature flags");

    const ginfer::GpuRequirements all_groupwise{
        .sm_86 = true, .sm_89 = true, .sm_120 = true, .nvfp4 = false};
    const ginfer::GpuRequirements blackwell_nvfp4{
        .sm_86 = false, .sm_89 = false, .sm_120 = true, .nvfp4 = true};
    const ginfer::GpuRequirements blackwell_only{
        .sm_86 = false, .sm_89 = false, .sm_120 = true, .nvfp4 = false};
    failures += check(ginfer::gpu_requirements_satisfied(c120, all_groupwise),
                      "cc120 should satisfy all-image groupwise requirements");
    failures += check(ginfer::gpu_requirements_satisfied(c120, blackwell_nvfp4),
                      "cc120 should satisfy Blackwell NVFP4 requirements");
    failures += check(ginfer::gpu_requirements_satisfied(c120, blackwell_only),
                      "cc120 should satisfy Blackwell-only requirements");

    auto c89 = ginfer::capability_for_cc(89, 128, 24ull << 30);
    failures += check(c89.sm_arch == ginfer::SmArch::Ada89, "cc89 sm_arch");
    failures += check(!c89.nvfp4 && !c89.pdl && !c89.tma && !c89.fp8_kind && !c89.fp8_ada,
                      "cc89 feature flags");
    failures += check(ginfer::gpu_requirements_satisfied(c89, all_groupwise),
                      "cc89 should satisfy all-image groupwise requirements");
    failures += check(!ginfer::gpu_requirements_satisfied(c89, blackwell_nvfp4),
                      "cc89 must reject Blackwell NVFP4 requirements");
    failures += check(!ginfer::gpu_requirements_satisfied(c89, blackwell_only),
                      "cc89 must reject Blackwell-only requirements");

    auto c86 = ginfer::capability_for_cc(86, 82, 24ull << 30);
    failures += check(c86.sm_arch == ginfer::SmArch::Ampere86, "cc86 sm_arch");
    failures += check(!c86.nvfp4 && !c86.pdl && !c86.tma && !c86.fp8_kind && !c86.fp8_ada,
                      "cc86 feature flags");
    failures += check(ginfer::gpu_requirements_satisfied(c86, all_groupwise),
                      "cc86 should satisfy all-image groupwise requirements");
    failures += check(!ginfer::gpu_requirements_satisfied(c86, blackwell_nvfp4),
                      "cc86 must reject Blackwell NVFP4 requirements");
    failures += check(!ginfer::gpu_requirements_satisfied(c86, blackwell_only),
                      "cc86 must reject Blackwell-only requirements");

    bool unknown_cc_rejected = false;
    try {
        (void)ginfer::capability_for_cc(75, 80, 8ull << 30);
    } catch (const std::invalid_argument&) { unknown_cc_rejected = true; }
    failures += check(unknown_cc_rejected, "capability_for_cc accepted unknown CC");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
