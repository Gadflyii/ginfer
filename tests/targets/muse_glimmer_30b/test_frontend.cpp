#include <ginfer/targets/muse_glimmer_30b/package.h>

#include "targets/muse_glimmer_30b/impl/frontend/chat_template.h"
#include "targets/muse_glimmer_30b/impl/load/bindings.h"

#include <nlohmann/json.hpp>

#include <array>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace muse = ginfer::targets::muse_glimmer_30b;
namespace fi   = muse::frontend_internal;
namespace md   = muse::detail;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

ginfer::ChatMessage message(ginfer::ChatRole role, std::string text) {
    ginfer::ChatMessage result;
    result.role = role;
    result.parts.push_back(ginfer::MessagePart{.kind = ginfer::MessagePartKind::Text,
                                               .text = std::move(text),
                                               .media = {}});
    return result;
}

ginfer::PromptInput protocol_input() {
    ginfer::PromptInput input;
    input.messages.push_back(message(ginfer::ChatRole::System, "test"));
    input.messages.push_back(message(ginfer::ChatRole::User, "x"));
    input.options.reasoning_effort = ginfer::ReasoningEffort::High;
    return input;
}

const std::string& encoded_protocol_prompt() {
    static const std::string value =
        "<|begin_of_text|><|start|>system<|message|>test\n\nReasoning strength: high.\n\n"
        "# Valid recipients: \"self\", \"user\".<|eot|><|start|>user<|message|>x<|eot|>"
        "<|start|>assistant";
    return value;
}

md::FrontendResources resources() {
    md::FrontendResources result;
    const nlohmann::json tokens = nlohmann::json::array({
        added(1, encoded_protocol_prompt()),
        added(2, "helloST"),
        added(3, "OPtail"),
        added(4, " to"),
        added(5, "=self"),
        added(6, "thought"),
        added(7, "assistant"),
        added(8, "=user"),
        added(9, "answer"),
        added(13, "ordinary"),
        added(200000, "<|begin_of_text|>", true),
        added(200001, "<|end_of_text|>", true),
        added(200007, "<|eom|>", true),
        added(200008, "<|eot|>", true),
        added(200022, "<|start|>", true),
        added(200023, "<|message|>", true),
    });
    result.tokenizer_json =
        nlohmann::json{{"model",
                        {{"type", "BPE"},
                         {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}, {"À", 14}}},
                         {"merges", nlohmann::json::array()}}},
                       {"added_tokens", tokens}}
            .dump();
    result.tokenizer_config_json  = "{}";
    result.chat_template_jinja    = "registered Muse ATEM template";
    result.generation_config_json = R"({"eos_token_id":[200001,200008]})";
    result.processor_config_json  = "{}";
    return result;
}

md::Frontend frontend() {
    ginfer::EngineOptions options;
    options.max_context = 1024;
    return md::make_frontend(resources(), options);
}

std::string channel_text(const md::PublishedOutput& output, ginfer::OutputChannel channel) {
    std::string result;
    for (const ginfer::OutputDelta& delta : output) {
        if (delta.channel == channel) { result += delta.text; }
    }
    return result;
}

int test_default_prompt_and_effort_options() {
    ginfer::PromptOptions options;
    const std::string rendered = fi::render_chat(
        {message(ginfer::ChatRole::User, "hello")}, options, "2030-01-02");
    const std::string expected =
        "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n"
        "Knowledge cutoff: 2026-01-04.\nCurrent date: 2030-01-02.\n\n"
        "Reasoning strength: high.\n\n# Valid recipients: \"self\", \"user\".<|eot|>"
        "<|start|>user<|message|>hello<|eot|><|start|>assistant";

    int failures = check(rendered == expected, "default Muse prompt did not match ATEM template");

    for (const auto& [effort, strength] :
         std::array<std::pair<ginfer::ReasoningEffort, std::string_view>, 4>{{
             {ginfer::ReasoningEffort::Low, "low"},
             {ginfer::ReasoningEffort::Medium, "medium"},
             {ginfer::ReasoningEffort::High, "high"},
             {ginfer::ReasoningEffort::XHigh, "xhigh"},
         }}) {
        ginfer::PromptOptions effort_options;
        effort_options.reasoning_effort = effort;
        const std::string effort_prompt = fi::render_chat(
            {message(ginfer::ChatRole::User, "hello")}, effort_options, "2030-01-02");
        failures += check(effort_prompt.find("Reasoning strength: " + std::string(strength) +
                                             ".") != std::string::npos,
                          "Muse reasoning-strength option was not rendered");
    }

    options.add_generation_prompt = false;
    options.reasoning_effort      = ginfer::ReasoningEffort::XHigh;
    const std::string system = fi::render_chat(
        {message(ginfer::ChatRole::System, "Policy\nReasoning effort: medium."),
         message(ginfer::ChatRole::User, "hello")},
        options, "ignored");
    failures += check(system.find("Reasoning strength: medium.") != std::string::npos &&
                          system.find("Reasoning strength: xhigh.") == std::string::npos,
                      "system reasoning directive was not normalized or was duplicated");
    failures += check(!system.ends_with("<|start|>assistant"),
                      "add_generation_prompt=false was ignored");

    const ginfer::PromptCapabilities capabilities = fi::prompt_capabilities();
    failures += check(capabilities.enable_thinking && capabilities.reasoning_effort.low &&
                          capabilities.reasoning_effort.medium &&
                          capabilities.reasoning_effort.high &&
                          capabilities.reasoning_effort.xhigh &&
                          capabilities.reasoning_effort.default_effort ==
                              ginfer::ReasoningEffort::High,
                      "Muse reasoning-strength capabilities are incomplete");

    options.add_generation_prompt = true;
    options.enable_thinking       = false;
    options.reasoning_effort.reset();
    const std::string instruct = fi::render_chat(
        {message(ginfer::ChatRole::User, "hello")}, options, "2030-01-02");
    failures += check(instruct.find("Reasoning strength: instruct.") != std::string::npos &&
                          instruct.find("Reasoning strength: high.") == std::string::npos,
                      "Muse non-thinking mode did not render its native instruct strength");

    options.reasoning_effort = ginfer::ReasoningEffort::Low;
    bool conflict_rejected   = false;
    try {
        (void)fi::render_chat({message(ginfer::ChatRole::User, "hello")}, options, "ignored");
    } catch (const std::invalid_argument&) { conflict_rejected = true; }
    failures += check(conflict_rejected,
                      "Muse accepted instruct mode combined with a reasoning strength");
    return failures;
}

int test_tools_and_history_prompt() {
    ginfer::PromptInput input;
    input.messages.push_back(message(ginfer::ChatRole::User, "weather?"));

    ginfer::ChatMessage assistant;
    assistant.role              = ginfer::ChatRole::Assistant;
    assistant.reasoning_content = "I should look it up.";
    assistant.tool_calls.push_back(ginfer::ToolCall{
        .id = "call_1", .name = "weather.lookup", .arguments_json = R"({"city":" New York ","days":2,"flags":[true,false]})"});
    input.messages.push_back(std::move(assistant));

    ginfer::ChatMessage tool = message(ginfer::ChatRole::Tool, R"({"temp":20})");
    tool.tool_call_id         = "call_1";
    input.messages.push_back(std::move(tool));
    input.messages.push_back(message(ginfer::ChatRole::User, "summarize"));
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"weather.lookup","description":"Fetch weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}})");

    const std::string rendered = fi::render_chat(input.messages, input.options, "2030-01-02");
    int failures = 0;
    failures += check(
        rendered.find("# Valid recipients: \"self\", \"weather.*\", \"user\".") !=
            std::string::npos,
        "tool namespace was not added to valid recipients");
    failures += check(rendered.find(
                          R"({"name": "weather.lookup", "description": "Fetch weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}})") !=
                          std::string::npos,
                      "Muse tool schema was not rendered in JSONSchema form");
    failures += check(
        rendered.find("<|start|>assistant to=self<|message|>I should look it up.<|eom|>") !=
            std::string::npos,
        "assistant reasoning history was not rendered to self");
    failures += check(
        rendered.find("<|start|>assistant to=weather.lookup<|message|>"
                      "<atem:function_calls>\n<atem:invoke name=\"weather.lookup\">\n"
                      "<atem:parameter name=\"city\"> New York </atem:parameter>\n"
                      "<atem:parameter name=\"days\">2</atem:parameter>\n"
                      "<atem:parameter name=\"flags\">[true, false]</atem:parameter>\n"
                      "</atem:invoke>\n</atem:function_calls><|eot|>") != std::string::npos,
        "assistant ATEM tool call history was formatted incorrectly");
    failures += check(
        rendered.find("<|start|>tool weather.lookup<|message|><tool_output "
                      "name=\"weather.lookup\">\n{\"temp\":20}\n</tool_output><|eot|>") !=
            std::string::npos,
        "tool result did not resolve its function name from tool_call_id");
    return failures;
}

int test_protocol_channels_and_utf8() {
    const md::Frontend component = frontend();
    auto prompt                  = component.prepare(protocol_input());
    auto session                 = component.make_output_session(prompt, {});

    constexpr std::array<ginfer::TokenId, 12> first_tokens{
        4, 5, 200023, 6, 200007, 200022, 7, 4, 8, 200023, 10, 11,
    };
    const auto first_decision =
        session.preview(first_tokens, 14, ginfer::FinishReason::OutputLimit);
    const md::PublishedOutput first = session.commit_preview();

    int failures = check(first_decision.accepted_tokens == first_tokens.size() &&
                             !first_decision.finished(),
                         "Muse protocol round ended before its stop token");
    failures += check(channel_text(first, ginfer::OutputChannel::Reasoning) == "thought" &&
                          channel_text(first, ginfer::OutputChannel::Content).empty(),
                      "Muse protocol framing did not separate the reasoning channel");

    constexpr std::array<ginfer::TokenId, 2> last_tokens{12, 200008};
    const auto last_decision =
        session.preview(last_tokens, 2, ginfer::FinishReason::OutputLimit);
    const md::PublishedOutput last = session.commit_preview();
    failures += check(last_decision.accepted_tokens == 2 &&
                          last_decision.finish_reason == ginfer::FinishReason::StopToken,
                      "Muse EOT did not terminate the output session");
    failures += check(channel_text(last, ginfer::OutputChannel::Content) == "中",
                      "Muse content channel did not preserve a cross-round UTF-8 codepoint");
    failures += check(session.reasoning_tokens() == 2,
                      "Muse reasoning token accounting did not include body and channel close");
    return failures;
}

int test_cross_round_channel_stop() {
    const md::Frontend component = frontend();
    auto prompt                  = component.prepare(protocol_input());
    ginfer::StopPolicy stop;
    stop.strings.push_back(ginfer::StopString{
        .text = "STOP", .channel = ginfer::OutputChannel::Reasoning, .include_in_output = false});
    auto session = component.make_output_session(prompt, stop);

    constexpr std::array<ginfer::TokenId, 4> first_tokens{4, 5, 200023, 2};
    const auto first_decision =
        session.preview(first_tokens, 5, ginfer::FinishReason::OutputLimit);
    const md::PublishedOutput first = session.commit_preview();
    int failures = check(!first_decision.finished() &&
                             channel_text(first, ginfer::OutputChannel::Reasoning) == "hello",
                         "Muse decoder did not retain an ambiguous stop suffix across rounds");

    const auto second_decision = session.preview(std::array<ginfer::TokenId, 1>{3}, 1,
                                                 ginfer::FinishReason::OutputLimit);
    const md::PublishedOutput second = session.commit_preview();
    failures += check(second_decision.finish_reason == ginfer::FinishReason::StopString &&
                          second_decision.accepted_tokens == 1,
                      "Muse decoder did not match a reasoning stop across token rounds");
    failures += check(second.empty(), "Muse stop marker or same-token suffix leaked to output");
    return failures;
}

int test_plain_and_raw_output() {
    const md::Frontend component = frontend();
    auto plain_prompt            = component.prepare_tokens({0});
    auto plain = component.make_output_session(plain_prompt, {});
    const auto plain_decision = plain.preview(std::array<ginfer::TokenId, 1>{13}, 1,
                                              ginfer::FinishReason::OutputLimit);
    const md::PublishedOutput plain_output = plain.commit_preview();

    auto raw_prompt = component.prepare(protocol_input());
    auto raw = component.make_output_session(
        raw_prompt, {}, ginfer::OutputOptions{.raw = true, .preserve_special_tokens = false});
    constexpr std::array<ginfer::TokenId, 4> raw_tokens{4, 5, 200023, 200008};
    const auto raw_decision =
        raw.preview(raw_tokens, 4, ginfer::FinishReason::OutputLimit);
    const md::PublishedOutput raw_output = raw.commit_preview();

    int failures = check(plain_decision.finish_reason == ginfer::FinishReason::OutputLimit &&
                             channel_text(plain_output, ginfer::OutputChannel::Content) ==
                                 "ordinary",
                         "prepare_tokens output was incorrectly forced through ATEM framing");
    failures += check(raw_decision.finish_reason == ginfer::FinishReason::StopToken &&
                          channel_text(raw_output, ginfer::OutputChannel::Content) ==
                              " to=self<|message|><|eot|>",
                      "raw Muse output did not preserve protocol special tokens");
    return failures;
}

int test_unpublished_stop_bypasses_discarded_bytes() {
    const md::Frontend component = frontend();
    auto prompt                  = component.prepare_tokens({0});
    ginfer::StopPolicy stop;
    stop.include_model_defaults = false;
    stop.token_ids              = {14}; // Byte-level token for the invalid standalone byte 0xc0.
    auto session                = component.make_output_session(prompt, stop);

    const auto decision = session.preview(std::array<ginfer::TokenId, 1>{14}, 1,
                                          ginfer::FinishReason::OutputLimit);
    const md::PublishedOutput output = session.commit_preview();
    return check(decision.accepted_tokens == 1 &&
                     decision.finish_reason == ginfer::FinishReason::StopToken && output.empty(),
                 "unpublished stop token decoded bytes that must be discarded");
}

} // namespace

int main() {
    int failures = 0;
    failures += test_default_prompt_and_effort_options();
    failures += test_tools_and_history_prompt();
    failures += test_protocol_channels_and_utf8();
    failures += test_cross_round_channel_stop();
    failures += test_plain_and_raw_output();
    failures += test_unpublished_stop_bypasses_discarded_bytes();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
