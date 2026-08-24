#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {

std::filesystem::path artifact_path() {
    if (const char* env = std::getenv("GINFER_QWEN3_6_35B_A3B_WEIGHTS");
        env != nullptr && *env != '\0') {
        return env;
    }
    return std::filesystem::path(GINFER_SOURCE_DIR) / "out/qwen3_6_35b_a3b.ginfer";
}

ginfer::targets::qwen3_6::StartupFeatures load_features(bool dflash) {
    return {
        .vision = !dflash,
        .speculative =
            dflash ? ginfer::SpeculativeBackend::DFlash : ginfer::SpeculativeBackend::Mtp,
        .proposal_head = ginfer::ProposalHead::Optimized,
    };
}

} // namespace

int main() {
    const std::filesystem::path path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real 35B artifact is unavailable at " << path << '\n';
        return 77;
    }

    ginfer::artifact::Reader reader(path);
    {
        ginfer::artifact::Binder binder(reader);
        const auto plan =
            ginfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(binder, load_features(false));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 883 ||
            plan.materialization.host_objects.size() != 6 ||
            plan.materialization.device_capacity_bytes != 22'360'207'360ULL ||
            plan.bindings.dflash.feature_projection.index != 889 ||
            plan.bindings.dflash.final_norm.index != 939) {
            std::cerr << "DFlash-disabled materialization plan changed resident weights\n";
            return 1;
        }
    }
    {
        ginfer::artifact::Binder binder(reader);
        const auto plan =
            ginfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(binder, load_features(true));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 586 ||
            plan.materialization.host_objects.size() != 6 ||
            plan.materialization.device_capacity_bytes != 21'591'653'888ULL) {
            std::cerr << "DFlash-enabled materialization plan is incomplete: device_objects="
                      << plan.materialization.device_objects.size()
                      << " device_bytes=" << plan.materialization.device_capacity_bytes << '\n';
            return 1;
        }
    }
    return 0;
}
