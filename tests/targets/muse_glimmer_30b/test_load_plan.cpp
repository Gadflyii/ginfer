#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/muse_glimmer_30b/impl/load/bindings.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace {

constexpr std::size_t kObjectCount            = 713;
constexpr std::size_t kResourceCount          = 5;
constexpr std::size_t kTextTensorCount        = 627;
constexpr std::size_t kDFlash2TensorCount     = 81;
constexpr std::uint64_t kTextDeviceBytes      = 16'228'779'008ULL;
constexpr std::size_t kNvfp4ObjectCount       = 921;
constexpr std::size_t kNvfp4DivisorCount      = 208;
constexpr std::uint64_t kNvfp4TextDeviceBytes = 17'015'251'968ULL;
constexpr std::uint64_t kDFlash2DeviceBytes   = 1'790'335'488ULL;
constexpr std::uint64_t kDFlash2Q4DeviceBytes = 1'473'220'096ULL;

std::filesystem::path artifact_path() {
    if (const char* value = std::getenv("GINFER_MUSE_GLIMMER_30B_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(GINFER_SOURCE_DIR) / "out/muse_glimmer_30b.ginfer";
}

bool is_dflash2(std::string_view name) { return name.starts_with("dflash2/"); }

int verify_plan(const ginfer::artifact::Reader& reader, bool enable_dflash2) {
    using Profile = ginfer::targets::muse_glimmer_30b::detail::WeightsProfile;
    const Profile profile =
        ginfer::targets::muse_glimmer_30b::Package::resolve_weights(reader.identity());
    const bool nvfp4 = profile == Profile::Nvfp4;
    const std::uint64_t dflash2_device_bytes =
        profile == Profile::GroupwiseIntDFlashQ4 || nvfp4 ? kDFlash2Q4DeviceBytes
                                                           : kDFlash2DeviceBytes;
    ginfer::artifact::Binder binder(reader);
    const auto plan = ginfer::targets::muse_glimmer_30b::detail::bind_artifact(
        binder, profile, enable_dflash2);

    ginfer::EngineOptions options;
    options.speculative.backend = enable_dflash2 ? ginfer::SpeculativeBackend::DFlash
                                                 : ginfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = enable_dflash2 ? 7 : 0;
    ginfer::artifact::Binder startup_binder(reader);
    const auto startup_plan = ginfer::targets::muse_glimmer_30b::Package::plan_load(
        startup_binder, options, profile);
    const auto& startup_materialization = startup_plan.materialization();
    if (startup_materialization.object_count != plan.materialization.object_count ||
        startup_materialization.device_objects.size() !=
            plan.materialization.device_objects.size() ||
        startup_materialization.host_objects.size() != plan.materialization.host_objects.size() ||
        startup_materialization.device_capacity_bytes !=
            plan.materialization.device_capacity_bytes) {
        std::cerr << "Muse startup options did not reach the target load plan\n";
        return 1;
    }

    const std::size_t expected_tensors =
        kTextTensorCount + (enable_dflash2 ? kDFlash2TensorCount : 0);
    const std::uint64_t expected_bytes =
        (nvfp4 ? kNvfp4TextDeviceBytes : kTextDeviceBytes) +
        (enable_dflash2 ? dflash2_device_bytes : 0);
    const std::size_t expected_objects = nvfp4 ? kNvfp4ObjectCount : kObjectCount;
    if (plan.bindings.dflash2.has_value() != enable_dflash2 ||
        plan.materialization.object_count != expected_objects ||
        plan.materialization.host_objects.size() != kResourceCount ||
        plan.materialization.device_objects.size() != expected_tensors ||
        plan.materialization.device_capacity_bytes != expected_bytes) {
        std::cerr << "Muse " << (enable_dflash2 ? "DFlash" : "AR")
                  << " load plan is wrong: objects=" << plan.materialization.object_count
                  << " device=" << plan.materialization.device_objects.size()
                  << " host=" << plan.materialization.host_objects.size()
                  << " bytes=" << plan.materialization.device_capacity_bytes << '\n';
        return 1;
    }

    std::size_t resident_dflash2          = 0;
    std::uint64_t resident_dflash2_bytes = 0;
    for (const auto& placement : plan.materialization.device_objects) {
        const std::string_view name =
            ginfer::artifact::object_name(reader.objects().at(placement.object.index));
        if (is_dflash2(name)) {
            ++resident_dflash2;
            resident_dflash2_bytes += placement.bytes;
        }
    }
    if (resident_dflash2 != (enable_dflash2 ? kDFlash2TensorCount : 0) ||
        resident_dflash2_bytes != (enable_dflash2 ? dflash2_device_bytes : 0)) {
        std::cerr << "Muse " << (enable_dflash2 ? "DFlash" : "AR")
                  << " plan has the wrong resident DFlash inventory: tensors="
                  << resident_dflash2 << " bytes=" << resident_dflash2_bytes << '\n';
        return 1;
    }

    const std::size_t validate_only =
        plan.materialization.object_count - plan.materialization.device_objects.size() -
        plan.materialization.host_objects.size();
    const std::size_t expected_validate_only =
        (enable_dflash2 ? 0 : kDFlash2TensorCount) + (nvfp4 ? kNvfp4DivisorCount : 0);
    if (validate_only != expected_validate_only) {
        std::cerr << "Muse load plan has the wrong validate-only count: " << validate_only << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    try {
        const std::filesystem::path path = artifact_path();
        if (!std::filesystem::is_regular_file(path)) {
            std::cerr << "skip: real Muse artifact is unavailable at " << path << '\n';
            return 77;
        }
        const ginfer::artifact::Reader reader(path);
        return verify_plan(reader, false) + verify_plan(reader, true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
