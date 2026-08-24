#include "artifact/reader.h"
#include "core/gpu_capability.h"
#include <ginfer/targets/muse_glimmer_30b/package.h>
#include <ginfer/targets/qwen3_6_27b/package.h>
#include <ginfer/targets/qwen3_6_35b_a3b/package.h>

#include <iostream>
#include <stdexcept>

namespace {

using Qwen27 = ginfer::targets::qwen3_6_27b::Package;
using Qwen35 = ginfer::targets::qwen3_6_35b_a3b::Package;
using Muse   = ginfer::targets::muse_glimmer_30b::Package;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const auto c120 = ginfer::capability_for_cc(120, 170, 32ull << 30);
    const auto c120_pro = ginfer::capability_for_cc(120, 188, 96ull << 30);
    const auto c89  = ginfer::capability_for_cc(89, 128, 24ull << 30);
    const auto c86  = ginfer::capability_for_cc(86, 82, 24ull << 30);
    ginfer::EngineOptions ar;
    ginfer::EngineOptions dflash;
    dflash.speculative.backend = ginfer::SpeculativeBackend::DFlash;

    const auto q36_groupwise =
        Qwen27::resolve_weights({"qwen3.6-27b", "groupwise-int"});
    const auto q38_dflash2_q4 =
        Qwen27::resolve_weights({"qwen3.8-27b", "groupwise-int-dflash2-q4"});
    const auto q38_dflash2_w8 =
        Qwen27::resolve_weights({"qwen3.8-27b", "groupwise-int-dflash2-w8"});
    const auto q36_nvfp4 = Qwen27::resolve_weights({"qwen3.6-27b", "nvfp4"});
    const auto q38_nvfp4 = Qwen27::resolve_weights({"qwen3.8-27b", "nvfp4"});
    const auto q38_nvfp4_dflash2_q4 =
        Qwen27::resolve_weights({"qwen3.8-27b", "nvfp4-dflash2-q4"});
    const auto q38_nvfp4_dflash2_w8 =
        Qwen27::resolve_weights({"qwen3.8-27b", "nvfp4-dflash2-w8"});
    ginfer::EngineOptions q38_automatic;
    Qwen27::resolve_startup_options(q38_dflash2_q4, q38_automatic);
    failures += check(q38_automatic.speculative.backend == ginfer::SpeculativeBackend::DFlash &&
                          q38_automatic.speculative.draft_tokens == 4 &&
                          q38_automatic.speculative.proposal_head == ginfer::ProposalHead::Full,
                      "Qwen3.8 DFlash2 automatic policy did not resolve to DFlash k4");
    ginfer::EngineOptions q38_nvfp4_automatic;
    Qwen27::resolve_startup_options(q38_nvfp4_dflash2_q4, q38_nvfp4_automatic);
    failures += check(
        q38_nvfp4_automatic.speculative.backend == ginfer::SpeculativeBackend::DFlash &&
            q38_nvfp4_automatic.speculative.draft_tokens == 4,
        "Qwen3.8 NVFP4 DFlash2 automatic policy did not resolve to DFlash k4");
    ginfer::EngineOptions q38_explicit_none;
    q38_explicit_none.speculative.backend = ginfer::SpeculativeBackend::None;
    q38_explicit_none.enable_vision        = true;
    Qwen27::resolve_startup_options(q38_dflash2_w8, q38_explicit_none);
    failures += check(q38_explicit_none.speculative.backend == ginfer::SpeculativeBackend::None &&
                          q38_explicit_none.speculative.draft_tokens == 0 &&
                          q38_explicit_none.enable_vision,
                      "Qwen3.8 DFlash2 identity did not preserve explicit AR/Vision startup");
    ginfer::EngineOptions q38_explicit_mtp;
    q38_explicit_mtp.speculative.backend      = ginfer::SpeculativeBackend::Mtp;
    q38_explicit_mtp.speculative.draft_tokens = 3;
    Qwen27::resolve_startup_options(q38_dflash2_q4, q38_explicit_mtp);
    failures += check(q38_explicit_mtp.speculative.backend == ginfer::SpeculativeBackend::Mtp &&
                          q38_explicit_mtp.speculative.draft_tokens == 3,
                      "Qwen3.8 DFlash2 identity did not preserve explicit MTP startup");
    for (const auto profile : {q36_groupwise, q38_dflash2_q4, q38_dflash2_w8}) {
        const auto support = Qwen27::gpu_requirements(profile, ar);
        failures += check(ginfer::gpu_requirements_satisfied(c120, support),
                          "Qwen 27B groupwise must support sm_120a");
        failures += check(ginfer::gpu_requirements_satisfied(c120_pro, support),
                          "Qwen 27B groupwise must admit RTX PRO 6000 Blackwell sm_120a");
        failures += check(ginfer::gpu_requirements_satisfied(c89, support),
                          "Qwen 27B groupwise must support sm_89");
        failures += check(ginfer::gpu_requirements_satisfied(c86, support),
                          "Qwen 27B groupwise must support sm_86");
    }
    for (const auto profile : {q36_nvfp4, q38_nvfp4, q38_nvfp4_dflash2_q4,
                               q38_nvfp4_dflash2_w8}) {
        const auto support = Qwen27::gpu_requirements(profile, ar);
        failures += check(ginfer::gpu_requirements_satisfied(c120, support),
                          "Qwen 27B NVFP4 must support sm_120a");
        failures += check(ginfer::gpu_requirements_satisfied(c120_pro, support),
                          "Qwen 27B NVFP4 must admit RTX PRO 6000 Blackwell sm_120a");
        failures += check(!ginfer::gpu_requirements_satisfied(c89, support),
                          "Qwen 27B NVFP4 must reject sm_89");
        failures += check(!ginfer::gpu_requirements_satisfied(c86, support),
                          "Qwen 27B NVFP4 must reject sm_86");
    }

    const auto q35 = Qwen35::resolve_weights({"qwen3.6-35b-a3b", "groupwise-int"});
    const auto q35_ar = Qwen35::gpu_requirements(q35, ar);
    failures += check(ginfer::gpu_requirements_satisfied(c120, q35_ar),
                      "Qwen 35B AR must support sm_120a");
    failures += check(ginfer::gpu_requirements_satisfied(c89, q35_ar),
                      "Qwen 35B AR must support sm_89");
    failures += check(ginfer::gpu_requirements_satisfied(c86, q35_ar),
                      "Qwen 35B AR must support sm_86");
    const auto q35_dflash = Qwen35::gpu_requirements(q35, dflash);
    failures += check(ginfer::gpu_requirements_satisfied(c120, q35_dflash),
                      "Qwen 35B DFlash must support sm_120a");
    failures += check(!ginfer::gpu_requirements_satisfied(c89, q35_dflash),
                      "Qwen 35B DFlash must reject unqualified sm_89");
    failures += check(!ginfer::gpu_requirements_satisfied(c86, q35_dflash),
                      "Qwen 35B DFlash must reject unqualified sm_86");

    const auto muse = Muse::resolve_weights({"muse-glimmer-30b", "groupwise-int"});
    const auto muse_q4 =
        Muse::resolve_weights({"muse-glimmer-30b", "groupwise-int-dflash-q4"});
    const auto muse_nvfp4 = Muse::resolve_weights({"muse-glimmer-30b", "nvfp4"});
    const auto muse_support = Muse::gpu_requirements(muse, ar);
    failures += check(ginfer::gpu_requirements_satisfied(c120, muse_support),
                      "Muse must support qualified sm_120a");
    failures += check(ginfer::gpu_requirements_satisfied(c120_pro, muse_support),
                      "Muse must admit RTX PRO 6000 Blackwell sm_120a");
    failures += check(ginfer::gpu_requirements_satisfied(c89, muse_support),
                      "Muse must admit sm_89 for physical qualification");
    failures += check(ginfer::gpu_requirements_satisfied(c86, muse_support),
                      "Muse must admit sm_86 for physical qualification");
    const auto muse_q4_support = Muse::gpu_requirements(muse_q4, dflash);
    failures += check(ginfer::gpu_requirements_satisfied(c120, muse_q4_support),
                      "Muse DFlash Q4 must support qualified sm_120a");
    failures += check(ginfer::gpu_requirements_satisfied(c120_pro, muse_q4_support),
                      "Muse DFlash Q4 must admit RTX PRO 6000 Blackwell sm_120a");
    failures += check(ginfer::gpu_requirements_satisfied(c89, muse_q4_support),
                      "Muse DFlash Q4 must admit sm_89 for physical qualification");
    failures += check(ginfer::gpu_requirements_satisfied(c86, muse_q4_support),
                      "Muse DFlash Q4 must admit sm_86 for physical qualification");

    const auto muse_nvfp4_support = Muse::gpu_requirements(muse_nvfp4, dflash);
    failures += check(ginfer::gpu_requirements_satisfied(c120, muse_nvfp4_support),
                      "Muse NVFP4 must support qualified sm_120a");
    failures += check(ginfer::gpu_requirements_satisfied(c120_pro, muse_nvfp4_support),
                      "Muse NVFP4 must admit RTX PRO 6000 Blackwell sm_120a");
    failures += check(!ginfer::gpu_requirements_satisfied(c89, muse_nvfp4_support),
                      "Muse NVFP4 must reject sm_89");
    failures += check(!ginfer::gpu_requirements_satisfied(c86, muse_nvfp4_support),
                      "Muse NVFP4 must reject sm_86");

    bool retired_q38_groupwise_rejected = false;
    try {
        (void)Qwen27::resolve_weights({"qwen3.8-27b", "groupwise-int"});
    } catch (const std::runtime_error&) { retired_q38_groupwise_rejected = true; }
    failures += check(retired_q38_groupwise_rejected,
                      "Qwen3.8 accepted the retired plain groupwise identity");

    bool invalid_muse_rejected = false;
    try {
        (void)Muse::resolve_weights({"muse-glimmer-30b", "nvfp4-radix"});
    } catch (const std::runtime_error&) { invalid_muse_rejected = true; }
    failures += check(invalid_muse_rejected, "Muse accepted an unregistered NVFP4 identity");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
