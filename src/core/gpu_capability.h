#pragma once

#include <cstddef>
namespace ginfer {

enum class SmArch : int { Ampere86 = 86, Ada89 = 89, Blackwell120 = 120 };

struct GpuCapability {
    SmArch sm_arch;
    int compute_capability; // major*10+minor, e.g. 86, 89, 120
    int sm_count;
    bool pdl;
    bool tma;
    bool nvfp4;    // kind::mxf4nvf4 + e2m1
    bool fp8_kind; // kind::f8f6f4
    bool fp8_ada;  // sm_89 e4m3 MMA (false until Phase D)
    std::size_t total_vram_bytes;
};

struct OccupancyPolicy {
    SmArch sm_arch;
    int sm_count;
    int rope_large_block_wave;  // sm_count * 6
    int gdn_output_target_ctas; // sm_count * 4
    int moe_prefill_persistent; // sm_count * 3
    int gqa_decode_splits_base; // sm_count / 2
};

// A target package owns this declaration for each of its closed weight/feature profiles. Core
// owns only the mechanical comparison with the selected device image.
struct GpuRequirements {
    bool sm_86;
    bool sm_89;
    bool sm_120;
    bool nvfp4;
};

[[nodiscard]] OccupancyPolicy make_occupancy_policy(SmArch sm_arch, int sm_count);
[[nodiscard]] GpuCapability capability_for_cc(int compute_capability, int sm_count,
                                              std::size_t total_vram_bytes);
[[nodiscard]] bool gpu_requirements_satisfied(const GpuCapability& cap,
                                              const GpuRequirements& requirements) noexcept;

} // namespace ginfer
