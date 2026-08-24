#include "core/gpu_capability.h"

#include <stdexcept>

namespace ginfer {

OccupancyPolicy make_occupancy_policy(SmArch sm_arch, int sm_count) {
    if (sm_count <= 0) {
        throw std::invalid_argument("make_occupancy_policy requires sm_count > 0");
    }
    switch (sm_arch) {
    case SmArch::Ampere86:
    case SmArch::Ada89:
    case SmArch::Blackwell120:
        break;
    default:
        throw std::invalid_argument("make_occupancy_policy requires a supported SM architecture");
    }
    return OccupancyPolicy{
        .sm_arch                = sm_arch,
        .sm_count               = sm_count,
        .rope_large_block_wave  = sm_count * 6,
        .gdn_output_target_ctas = sm_count * 4,
        .moe_prefill_persistent = sm_count * 3,
        .gqa_decode_splits_base = sm_count / 2,
    };
}

GpuCapability capability_for_cc(int compute_capability, int sm_count,
                                std::size_t total_vram_bytes) {
    GpuCapability cap{};
    cap.compute_capability = compute_capability;
    cap.sm_count           = sm_count;
    cap.total_vram_bytes   = total_vram_bytes;
    cap.fp8_ada            = false;

    switch (compute_capability) {
    case 86:
        cap.sm_arch  = SmArch::Ampere86;
        cap.pdl      = false;
        cap.tma      = false;
        cap.nvfp4    = false;
        cap.fp8_kind = false;
        break;
    case 89:
        cap.sm_arch  = SmArch::Ada89;
        cap.pdl      = false;
        cap.tma      = false;
        cap.nvfp4    = false;
        cap.fp8_kind = false;
        break;
    case 120:
        cap.sm_arch  = SmArch::Blackwell120;
        cap.pdl      = true;
        cap.tma      = true;
        cap.nvfp4    = true;
        cap.fp8_kind = true;
        break;
    default:
        throw std::invalid_argument("capability_for_cc: unsupported compute capability");
    }
    return cap;
}

bool gpu_requirements_satisfied(const GpuCapability& cap,
                                const GpuRequirements& requirements) noexcept {
    bool architecture_supported = false;
    switch (cap.sm_arch) {
    case SmArch::Ampere86:
        architecture_supported = requirements.sm_86;
        break;
    case SmArch::Ada89:
        architecture_supported = requirements.sm_89;
        break;
    case SmArch::Blackwell120:
        architecture_supported = requirements.sm_120;
        break;
    }
    return architecture_supported && (!requirements.nvfp4 || cap.nvfp4);
}

} // namespace ginfer
