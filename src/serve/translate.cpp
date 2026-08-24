#include "serve/translate.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ginfer::serve {
namespace {

std::uint64_t random_seed() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng();
}

[[noreturn]] void invalid_sampling(std::string message, std::string param) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

[[noreturn]] void invalid_prompt_option(std::string message, std::string param, std::string code) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

ginfer::SamplingOverrides resolve_sampling_overrides(const SamplingParams& request,
                                                     const ServeOptions& server) {
    ginfer::SamplingOverrides sampling = server.sampling_overrides;
    if (request.temperature) { sampling.temperature = static_cast<float>(*request.temperature); }
    if (request.top_p) { sampling.top_p = static_cast<float>(*request.top_p); }
    if (request.top_k) { sampling.top_k = static_cast<std::int32_t>(*request.top_k); }
    if (request.presence_penalty) {
        sampling.presence_penalty = static_cast<float>(*request.presence_penalty);
    }
    if (request.frequency_penalty) {
        sampling.frequency_penalty = static_cast<float>(*request.frequency_penalty);
    }
    if (request.seed) {
        sampling.seed = *request.seed;
    } else if (server.sampling_overrides.seed) {
        sampling.seed = *server.sampling_overrides.seed;
    } else {
        sampling.seed = random_seed();
    }

    const auto finite = [](const std::optional<float>& value) {
        return !value || std::isfinite(*value);
    };
    if (!finite(sampling.temperature) || !finite(sampling.top_p) || !finite(sampling.min_p) ||
        !finite(sampling.presence_penalty) || !finite(sampling.frequency_penalty)) {
        invalid_sampling("sampling parameters must be finite", "sampling");
    }
    if (sampling.temperature && (*sampling.temperature < 0.0F || *sampling.temperature > 2.0F)) {
        invalid_sampling("temperature must be in [0,2]", "temperature");
    }
    if (sampling.top_p && (*sampling.top_p < 0.0F || *sampling.top_p > 1.0F)) {
        invalid_sampling("top_p must be in [0,1]", "top_p");
    }
    if (sampling.top_k && *sampling.top_k < 0) {
        invalid_sampling("top_k must be nonnegative", "top_k");
    }
    if (sampling.min_p && (*sampling.min_p < 0.0F || *sampling.min_p > 1.0F)) {
        invalid_sampling("min_p must be in [0,1]", "min_p");
    }
    if (sampling.presence_penalty &&
        (*sampling.presence_penalty < -2.0F || *sampling.presence_penalty > 2.0F)) {
        invalid_sampling("presence_penalty must be in [-2,2]", "presence_penalty");
    }
    if (sampling.frequency_penalty &&
        (*sampling.frequency_penalty < -2.0F || *sampling.frequency_penalty > 2.0F)) {
        invalid_sampling("frequency_penalty must be in [-2,2]", "frequency_penalty");
    }
    if (server.greedy) { sampling.temperature = 0.0F; }
    return sampling;
}

std::vector<std::string> effective_tool_jsons(const GenerationRequest& request) {
    std::vector<std::string> tools;
    if (!request.uses_tools()) { return tools; }
    if (request.tool_choice.mode == ToolChoiceMode::Named) {
        for (const ToolDefinition& tool : request.tools) {
            if (tool.name == request.tool_choice.name) {
                tools.push_back(tool.definition_json);
                break;
            }
        }
        return tools;
    }
    tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) { tools.push_back(tool.definition_json); }
    return tools;
}

int requested_effort_rank(RequestedReasoningEffort effort) {
    switch (effort) {
    case RequestedReasoningEffort::Minimal:
    case RequestedReasoningEffort::Low:
        return 0;
    case RequestedReasoningEffort::Medium:
        return 1;
    case RequestedReasoningEffort::High:
        return 2;
    case RequestedReasoningEffort::XHigh:
        return 3;
    case RequestedReasoningEffort::Max:
        return 4;
    case RequestedReasoningEffort::None:
        break;
    }
    throw std::logic_error("none has no positive reasoning-effort rank");
}

std::optional<ginfer::ReasoningEffort>
nearest_supported_effort(RequestedReasoningEffort requested,
                         const ginfer::ReasoningEffortCapabilities& capabilities) {
    struct Candidate {
        ginfer::ReasoningEffort effort;
        int rank;
        bool supported;
    };
    const Candidate candidates[] = {
        {ginfer::ReasoningEffort::Low, 0, capabilities.low},
        {ginfer::ReasoningEffort::Medium, 1, capabilities.medium},
        {ginfer::ReasoningEffort::High, 2, capabilities.high},
        {ginfer::ReasoningEffort::XHigh, 3, capabilities.xhigh},
    };

    const int requested_rank = requested_effort_rank(requested);
    const Candidate* best    = nullptr;
    int best_distance        = 100;
    for (const Candidate& candidate : candidates) {
        if (!candidate.supported) { continue; }
        const int distance = candidate.rank > requested_rank ? candidate.rank - requested_rank
                                                              : requested_rank - candidate.rank;
        // Prefer the stronger native level when a protocol value lies exactly between two
        // supported levels (for example Qwen high between medium and xhigh).
        if (best == nullptr || distance < best_distance ||
            (distance == best_distance && candidate.rank > best->rank)) {
            best          = &candidate;
            best_distance = distance;
        }
    }
    return best != nullptr ? std::optional<ginfer::ReasoningEffort>(best->effort) : std::nullopt;
}

} // namespace

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& request,
                                                 const ServeOptions& server,
                                                 const ginfer::PromptCapabilities& capabilities) {
    ResolvedPromptSemantics result{
        .enable_thinking   = request.enable_thinking.value_or(server.enable_thinking),
        .reasoning_effort  = std::nullopt,
        .preserve_thinking = request.preserve_thinking.value_or(server.preserve_thinking),
    };
    if (!request.reasoning_effort) {
        if (!result.enable_thinking && !capabilities.enable_thinking) {
            invalid_prompt_option("the loaded chat template cannot disable thinking",
                                  request.enable_thinking_param,
                                  "reasoning_effort_not_supported");
        }
        return result;
    }

    const RequestedReasoningEffort requested = *request.reasoning_effort;
    const bool enables_thinking              = requested != RequestedReasoningEffort::None;
    if (request.enable_thinking && *request.enable_thinking != enables_thinking) {
        invalid_prompt_option("reasoning effort conflicts with enable_thinking",
                              request.reasoning_effort_param, "conflicting_template_option");
    }
    result.enable_thinking = enables_thinking;

    if (requested == RequestedReasoningEffort::None) {
        if (!capabilities.enable_thinking) {
            invalid_prompt_option("the loaded chat template cannot disable thinking",
                                  request.reasoning_effort_param, "reasoning_effort_not_supported");
        }
        return result;
    }

    // Protocol vocabularies are wider than any one registered chat template. Resolve every
    // positive label to the closest native level. A binary thinking template has no native
    // effort levels, so the explicit effort still selects its thinking-on mode and leaves the
    // template's own default depth unchanged.
    result.reasoning_effort = nearest_supported_effort(requested, capabilities.reasoning_effort);
    return result;
}

ginfer::PromptInput to_prompt_input(const GenerationRequest& request,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media) {
    ginfer::PromptInput input;
    input.messages.reserve(request.messages.size());
    for (const ChatTurn& turn : request.messages) {
        ginfer::ChatMessage message;
        message.role              = turn.role;
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(ginfer::ToolCall{call.id, call.name, call.arguments_json});
        }

        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Text) {
                if (!message.parts.empty() && !part.text.empty() &&
                    message.parts.back().kind == ginfer::MessagePartKind::Text) {
                    ginfer::MessagePart newline;
                    newline.text = "\n";
                    message.parts.push_back(std::move(newline));
                }
                ginfer::MessagePart text;
                text.text = part.text;
                message.parts.push_back(std::move(text));
                continue;
            }
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                if (!acquire_media) {
                    throw std::logic_error("media acquisition callback is not configured");
                }
                ginfer::MessagePart media;
                media.kind  = ginfer::MessagePartKind::Media;
                media.media = acquire_media(part);
                message.parts.push_back(std::move(media));
                continue;
            }

            ApiError error;
            error.message = "content type '" + part.type_raw + "' is not supported";
            error.param   = "messages";
            error.code    = "modality_not_supported";
            throw ApiException(std::move(error));
        }
        input.messages.push_back(std::move(message));
    }

    input.options.add_generation_prompt = true;
    input.options.enable_thinking       = semantics.enable_thinking;
    input.options.reasoning_effort      = semantics.reasoning_effort;
    input.options.preserve_thinking     = semantics.preserve_thinking;
    input.options.add_vision_id         = false;
    input.options.tool_jsons            = effective_tool_jsons(request);
    return input;
}

ginfer::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server) {
    ginfer::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = server.allow_prefix_reuse;
    options.execution.sampling             = resolve_sampling_overrides(request.sampling, server);
    options.output.raw                     = false;
    options.output.preserve_special_tokens = request.uses_tools() || request.has_tool_history();
    options.stop.strings.reserve(request.stop_strings.size());
    for (const std::string& stop : request.stop_strings) {
        if (!stop.empty()) {
            options.stop.strings.push_back(
                ginfer::StopString{.text              = stop,
                                   .channel           = ginfer::OutputChannel::Content,
                                   .include_in_output = false});
        }
    }
    return options;
}

const char* finish_reason_wire(ginfer::FinishReason reason) {
    switch (reason) {
    case ginfer::FinishReason::OutputLimit:
    case ginfer::FinishReason::ContextCapacity:
        return "length";
    case ginfer::FinishReason::None:
    case ginfer::FinishReason::StopToken:
    case ginfer::FinishReason::StopString:
    case ginfer::FinishReason::Cancelled:
        return "stop";
    }
    return "stop";
}

} // namespace ginfer::serve
