#include "ginfer/engine.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace {

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(GINFER_SOURCE_DIR) / "out" / filename;
}

ginfer::EngineOptions engine_options(const std::filesystem::path& artifact) {
    ginfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 2304;
    options.kv_capacity     = ginfer::KvCapacityPolicy::explicit_capacity(2304);
    options.prefill_chunk   = 2048;
    // Automatic is the public production default. The exact DFlash2 artifact identity owns the
    // resolution to DFlash with four proposal tokens; CUDA Graphs remain enabled by default.
    options.speculative.backend = ginfer::SpeculativeBackend::Automatic;
    return options;
}

ginfer::PromptInput prompt() {
    ginfer::PromptInput input;
    input.options.enable_thinking = false;
    ginfer::ChatMessage user;
    user.role = ginfer::ChatRole::User;
    user.parts.push_back(ginfer::MessagePart{
        .kind = ginfer::MessagePartKind::Text,
        .text = "Say hello in one short sentence.",
        .media = {},
    });
    input.messages.push_back(std::move(user));
    return input;
}

ginfer::RequestOptions request_options() {
    ginfer::RequestOptions options;
    options.execution.requested_output_tokens   = 4;
    options.execution.sampling.temperature      = 0.7F;
    options.execution.sampling.top_k            = 20;
    options.execution.sampling.top_p            = 0.8F;
    options.execution.sampling.seed             = 42;
    options.execution.allow_prefix_reuse        = false;
    options.stop.include_model_defaults         = false;
    return options;
}

int exercise(const std::filesystem::path& artifact, std::string_view weights_id) {
    ginfer::Engine engine(engine_options(artifact));
    if (engine.load_summary().model_id != "qwen3.8-27b" ||
        engine.load_summary().weights_id != weights_id) {
        std::cerr << "Qwen3.8 DFlash2 artifact resolved to the wrong identity\n";
        return 1;
    }
    if (engine.options().speculative.backend != ginfer::SpeculativeBackend::DFlash ||
        engine.options().speculative.draft_tokens != 4 ||
        engine.options().speculative.proposal_head != ginfer::ProposalHead::Full) {
        std::cerr << "Qwen3.8 DFlash2 automatic startup policy did not resolve to DFlash k4\n";
        return 1;
    }

    const ginfer::GenerationResult result =
        engine.generate(engine.prepare(prompt()), request_options());
    if (result.generated_token_ids.size() != 4 ||
        result.finish_reason != ginfer::FinishReason::OutputLimit ||
        !result.speculative.enabled ||
        result.speculative.backend != ginfer::SpeculativeBackend::DFlash ||
        result.speculative.draft_window != 4 || result.speculative.verification_window != 15 ||
        result.speculative.rounds == 0 || result.speculative.expanded_rounds != 0 ||
        result.speculative.drafted_tokens == 0 ||
        result.speculative.accepted_per_position.size() != 15) {
        std::cerr << "Qwen3.8 request did not execute through the sampled DFlash2 k4 route: "
                  << "generated=" << result.generated_token_ids.size()
                  << " finish=" << static_cast<int>(result.finish_reason)
                  << " rounds=" << result.speculative.rounds
                  << " drafted=" << result.speculative.drafted_tokens << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    struct ProfileCase {
        const char* environment;
        const char* filename;
        const char* weights_id;
    };
    constexpr std::array profiles{
        ProfileCase{"GINFER_QWEN3_8_27B_DFLASH2_Q4_WEIGHTS",
                    "qwen3_8_27b_dflash2_q4.ginfer", "groupwise-int-dflash2-q4"},
        ProfileCase{"GINFER_QWEN3_8_27B_DFLASH2_W8_WEIGHTS",
                    "qwen3_8_27b_dflash2_w8.ginfer", "groupwise-int-dflash2-w8"},
        ProfileCase{"GINFER_QWEN3_8_27B_NVFP4_DFLASH2_Q4_WEIGHTS",
                    "qwen3_8_27b_nvfp4_dflash2_q4.ginfer", "nvfp4-dflash2-q4"},
        ProfileCase{"GINFER_QWEN3_8_27B_NVFP4_DFLASH2_Q8_WEIGHTS",
                    "qwen3_8_27b_nvfp4_dflash2_q8.ginfer", "nvfp4-dflash2-w8"},
    };

    bool exercised = false;
    for (const ProfileCase& profile : profiles) {
        const std::filesystem::path artifact = artifact_path(profile.environment, profile.filename);
        if (!std::filesystem::is_regular_file(artifact)) {
            continue;
        }
        exercised = true;
        if (const int result = exercise(artifact, profile.weights_id); result != 0) {
            return result;
        }
    }
    if (!exercised) {
        std::cerr << "skip: no real Qwen3.8 DFlash2 artifact is available\n";
        return 77;
    }
    return 0;
}
