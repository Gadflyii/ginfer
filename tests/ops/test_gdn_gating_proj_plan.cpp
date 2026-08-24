#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using ginfer::OccupancyPolicy;
using ginfer::SmArch;
using ginfer::make_occupancy_policy;
using ginfer::ops::detail::Bf16GdnGatingProblem;
using ginfer::ops::detail::Bf16GdnGatingScheduleId;
using ginfer::ops::detail::bf16_gdn_gating_capacity_workspace_bytes;
using ginfer::ops::detail::bf16_gdn_gating_resolve_plan;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int check_route(const char* label, const OccupancyPolicy& occupancy, std::int32_t heads,
                std::int32_t input_rows, std::int32_t cols,
                Bf16GdnGatingScheduleId expected) {
    const auto plan = bf16_gdn_gating_resolve_plan({heads, input_rows, cols}, occupancy);
    return check(plan.schedule == expected,
                 std::string(label) + " T=" + std::to_string(cols) + " resolved " +
                     ginfer::ops::detail::bf16_gdn_gating_schedule_name(plan.schedule));
}

int check_interval(const char* label, const OccupancyPolicy& occupancy, std::int32_t heads,
                   std::int32_t input_rows, std::int32_t first, std::int32_t last,
                   std::size_t expected) {
    std::size_t exhaustive = 0;
    for (std::int32_t cols = first; cols <= last; ++cols) {
        exhaustive = std::max(
            exhaustive,
            bf16_gdn_gating_resolve_plan({heads, input_rows, cols}, occupancy).workspace_bytes);
    }
    const std::size_t capacity = bf16_gdn_gating_capacity_workspace_bytes(
        heads, input_rows, first, last, occupancy);
    int failures = 0;
    failures += check(capacity == exhaustive,
                      std::string(label) + " interval capacity missed an exact-T plan");
    failures += check(capacity == expected,
                      std::string(label) + " interval capacity changed unexpectedly");
    return failures;
}

} // namespace

int main() {
    using Schedule = Bf16GdnGatingScheduleId;
    const OccupancyPolicy sm86 = make_occupancy_policy(SmArch::Ampere86, 82);
    const OccupancyPolicy sm89 = make_occupancy_policy(SmArch::Ada89, 128);
    const OccupancyPolicy sm120 = make_occupancy_policy(SmArch::Blackwell120, 170);

    int failures = 0;

    // SM86 27B: Split8 is limited by 40 KiB shared memory to two CTAs/SM; the 512-thread
    // Split4/2 kernels are register-limited to one CTA/SM.
    failures += check_route("sm86 27B last Split8", sm86, 48, 5120, 768,
                            Schedule::MmaCooperativeSplit8);
    failures += check_route("sm86 27B first lower-split fallback", sm86, 48, 5120, 769,
                            Schedule::MmaCooperativeSplit2);
    failures += check_route("sm86 27B last Split2", sm86, 48, 5120, 1664,
                            Schedule::MmaCooperativeSplit2);
    failures += check_route("sm86 27B first unsplit", sm86, 48, 5120, 1665,
                            Schedule::MmaUnsplit);

    // SM86 35B Split8/4/2 use 74 registers with 256 threads, admitting three CTAs/SM.
    failures += check_route("sm86 35B last Split8", sm86, 32, 2048, 960,
                            Schedule::MmaCooperativeSplit8);
    failures += check_route("sm86 35B first Split4 fallback", sm86, 32, 2048, 961,
                            Schedule::MmaCooperativeSplit4);
    failures += check_route("sm86 35B last Split4", sm86, 32, 2048, 1920,
                            Schedule::MmaCooperativeSplit4);
    failures += check_route("sm86 35B first Split2 fallback", sm86, 32, 2048, 1921,
                            Schedule::MmaCooperativeSplit2);
    failures += check_route("sm86 35B last Split2", sm86, 32, 2048, 3904,
                            Schedule::MmaCooperativeSplit2);
    failures += check_route("sm86 35B first unsplit", sm86, 32, 2048, 3905,
                            Schedule::MmaUnsplit);

    // SM89 has the same per-CTA resource profile but enough SMs for every 35B catalog grid.
    failures += check_route("sm89 27B catalog Split8", sm89, 48, 5120, 1024,
                            Schedule::MmaCooperativeSplit8);
    failures += check_route("sm89 27B last Split4", sm89, 48, 5120, 1280,
                            Schedule::MmaCooperativeSplit4);
    failures += check_route("sm89 27B first Split2 fallback", sm89, 48, 5120, 1281,
                            Schedule::MmaCooperativeSplit2);
    failures += check_route("sm89 27B last Split2", sm89, 48, 5120, 2688,
                            Schedule::MmaCooperativeSplit2);
    failures += check_route("sm89 27B first unsplit", sm89, 48, 5120, 2689,
                            Schedule::MmaUnsplit);
    failures += check_route("sm89 35B Split8 endpoint", sm89, 32, 2048, 1024,
                            Schedule::MmaCooperativeSplit8);
    failures += check_route("sm89 35B Split4 endpoint", sm89, 32, 2048, 2048,
                            Schedule::MmaCooperativeSplit4);
    failures += check_route("sm89 35B Split2 endpoint", sm89, 32, 2048, 4096,
                            Schedule::MmaCooperativeSplit2);

    // Preserve the qualified Blackwell catalog while architecture-specific policies diverge.
    failures += check_route("sm120 27B Split8 endpoint", sm120, 48, 5120, 1024,
                            Schedule::MmaCooperativeSplit8);
    failures += check_route("sm120 35B Split8 endpoint", sm120, 32, 2048, 1024,
                            Schedule::MmaCooperativeSplit8);

    // Exhaustive fixed-domain witnesses protect interval capacity across every residency step.
    failures += check_interval("sm86 27B Split8 fallback interval", sm86, 48, 5120, 9, 1024,
                               2'359'296);
    failures += check_interval("sm86 35B Split8 fallback interval", sm86, 32, 2048, 128, 1024,
                               1'966'080);
    failures += check_interval("sm89 27B Split4 fallback interval", sm89, 48, 5120, 1025, 2048,
                               1'966'080);
    failures += check_interval("sm86 27B", sm86, 48, 5120, 1, 4096, 2'359'296);
    failures += check_interval("sm86 35B", sm86, 32, 2048, 1, 4096, 1'998'848);
    failures += check_interval("sm89 27B", sm89, 48, 5120, 1, 4096, 3'145'728);
    failures += check_interval("sm89 35B", sm89, 32, 2048, 1, 4096, 2'097'152);

    if (failures == 0) { std::cout << "OK BF16 GDN architecture plans\n"; }
    return failures == 0 ? 0 : 1;
}
