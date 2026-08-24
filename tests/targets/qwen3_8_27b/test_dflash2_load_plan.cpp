#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <ginfer/targets/qwen3_6_27b/package.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace {

using ginfer::targets::qwen3_6_27b::Package;
using namespace ginfer::targets::qwen3_6_27b::detail;

constexpr std::size_t kObjectCount        = 1205;
constexpr std::size_t kResourceCount      = 6;
constexpr std::size_t kGroupwiseBaseTensorCount = 771;
constexpr std::size_t kNvfp4BaseTensorCount     = 659;
constexpr std::size_t kDFlash2TensorCount = 81;
constexpr std::uint64_t kQ4CompanionBytes = 1'022'732'800ULL;
constexpr std::uint64_t kQ8CompanionBytes = 2'044'930'560ULL;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(GINFER_SOURCE_DIR) / "out" / filename;
}

bool is_dflash2(std::string_view name) { return name.starts_with("dflash2/"); }

struct PlanSummary {
    std::uint64_t device_bytes{};
};

PlanSummary verify_plan(const std::filesystem::path& path, WeightsProfile expected_profile,
                        std::size_t base_tensor_count, bool enable_dflash2) {
    const ginfer::artifact::Reader reader(path);
    const WeightsProfile profile = Package::resolve_weights(reader.identity());
    if (profile != expected_profile) {
        std::cerr << "Qwen3.8 DFlash2 identity resolved to the wrong profile\n";
        throw std::runtime_error("Qwen3.8 DFlash2 identity resolved to the wrong profile");
    }

    ginfer::EngineOptions options;
    options.speculative.backend = enable_dflash2 ? ginfer::SpeculativeBackend::DFlash
                                                 : ginfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = enable_dflash2 ? 4 : 0;
    options.speculative.proposal_head = ginfer::ProposalHead::Full;

    ginfer::artifact::Binder direct_binder(reader);
    const auto direct = bind_artifact(
        direct_binder, profile,
        ginfer::targets::qwen3_6::startup_features(options));
    ginfer::artifact::Binder package_binder(reader);
    const auto package = Package::plan_load(package_binder, options, profile);
    const auto& materialization = direct.materialization;
    const auto& public_materialization = package.materialization();
    if (materialization.object_count != public_materialization.object_count ||
        materialization.device_objects.size() != public_materialization.device_objects.size() ||
        materialization.host_objects.size() != public_materialization.host_objects.size() ||
        materialization.device_capacity_bytes != public_materialization.device_capacity_bytes) {
        std::cerr << "Qwen3.8 startup options did not reach the target load plan\n";
        throw std::runtime_error("Qwen3.8 startup options did not reach the target load plan");
    }

    const std::size_t expected_device =
        base_tensor_count + (enable_dflash2 ? kDFlash2TensorCount : 0);
    const std::size_t expected_validate =
        kObjectCount - kResourceCount - expected_device;
    const std::size_t validate_only = materialization.object_count -
                                      materialization.device_objects.size() -
                                      materialization.host_objects.size();
    if (direct.bindings.dflash2.has_value() != enable_dflash2 ||
        materialization.object_count != kObjectCount ||
        materialization.device_objects.size() != expected_device ||
        materialization.host_objects.size() != kResourceCount ||
        validate_only != expected_validate) {
        std::cerr << "Qwen3.8 " << (enable_dflash2 ? "DFlash" : "base")
                  << " load plan is wrong: objects=" << materialization.object_count
                  << " device=" << materialization.device_objects.size()
                  << " host=" << materialization.host_objects.size()
                  << " validate=" << validate_only
                  << " bytes=" << materialization.device_capacity_bytes << '\n';
        throw std::runtime_error("Qwen3.8 load plan inventory is wrong");
    }

    std::size_t resident_companion = 0;
    for (const auto& placement : materialization.device_objects) {
        const std::string_view name =
            ginfer::artifact::object_name(reader.objects().at(placement.object.index));
        resident_companion += is_dflash2(name) ? 1 : 0;
    }
    if (resident_companion != (enable_dflash2 ? kDFlash2TensorCount : 0)) {
        std::cerr << "Qwen3.8 load plan has the wrong resident DFlash2 inventory\n";
        throw std::runtime_error("Qwen3.8 load plan has the wrong resident DFlash2 inventory");
    }
    return PlanSummary{materialization.device_capacity_bytes};
}

} // namespace

int main() {
    try {
        struct ProfileCase {
            const char* environment;
            const char* filename;
            WeightsProfile profile;
            std::size_t base_tensor_count;
            std::uint64_t companion_bytes;
        };
        constexpr std::array profiles{
            ProfileCase{"GINFER_QWEN3_8_27B_DFLASH2_Q4_WEIGHTS",
                        "qwen3_8_27b_dflash2_q4.ginfer",
                        WeightsProfile::Qwen38GroupwiseIntDFlash2Q4,
                        kGroupwiseBaseTensorCount, kQ4CompanionBytes},
            ProfileCase{"GINFER_QWEN3_8_27B_DFLASH2_W8_WEIGHTS",
                        "qwen3_8_27b_dflash2_w8.ginfer",
                        WeightsProfile::Qwen38GroupwiseIntDFlash2W8,
                        kGroupwiseBaseTensorCount, kQ8CompanionBytes},
            ProfileCase{"GINFER_QWEN3_8_27B_NVFP4_DFLASH2_Q4_WEIGHTS",
                        "qwen3_8_27b_nvfp4_dflash2_q4.ginfer",
                        WeightsProfile::Qwen38Nvfp4DFlash2Q4,
                        kNvfp4BaseTensorCount, kQ4CompanionBytes},
            ProfileCase{"GINFER_QWEN3_8_27B_NVFP4_DFLASH2_Q8_WEIGHTS",
                        "qwen3_8_27b_nvfp4_dflash2_q8.ginfer",
                        WeightsProfile::Qwen38Nvfp4DFlash2W8,
                        kNvfp4BaseTensorCount, kQ8CompanionBytes},
        };

        bool exercised = false;
        for (const ProfileCase& profile : profiles) {
            const auto path = artifact_path(profile.environment, profile.filename);
            if (!std::filesystem::is_regular_file(path)) {
                continue;
            }
            exercised = true;
            const PlanSummary ar =
                verify_plan(path, profile.profile, profile.base_tensor_count, false);
            const PlanSummary dflash =
                verify_plan(path, profile.profile, profile.base_tensor_count, true);
            if (dflash.device_bytes - ar.device_bytes != profile.companion_bytes) {
                std::cerr << "Qwen3.8 DFlash2 companion has the wrong resident byte delta: "
                          << (dflash.device_bytes - ar.device_bytes) << '\n';
                return 1;
            }
        }
        if (!exercised) {
            std::cerr << "skip: no real Qwen3.8 DFlash2 artifact is available\n";
            return 77;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
