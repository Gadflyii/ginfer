#include "targets/muse_glimmer_30b/impl/frontend/chat_template.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ginfer::targets::muse_glimmer_30b::frontend_internal {
namespace {

using OrderedJson = nlohmann::ordered_json;

struct ToolDefinition {
    std::string name;
    std::string description;
    OrderedJson parameters;
};

std::string message_text(const ChatMessage& message) {
    std::string text;
    for (const MessagePart& part : message.parts) {
        if (part.kind == MessagePartKind::Media) {
            throw std::invalid_argument(
                "muse-glimmer-30b v1 is text-only; image/video prompt parts are not supported");
        }
        text += part.text;
    }
    return text;
}

void validate_instruction(const ChatMessage& message) {
    if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
        !message.tool_call_id.empty()) {
        throw std::invalid_argument("system and developer messages may contain only text content");
    }
}

std::string tojson_text(const OrderedJson& value) {
    if (value.is_array()) {
        std::string rendered = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += tojson_text(value[index]);
        }
        rendered += "]";
        return rendered;
    }
    if (value.is_object()) {
        std::string rendered = "{";
        std::size_t index    = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += OrderedJson(it.key()).dump();
            rendered += ": ";
            rendered += tojson_text(it.value());
        }
        rendered += "}";
        return rendered;
    }
    return value.dump();
}

std::vector<ToolDefinition> parse_tools(const std::vector<std::string>& tool_jsons) {
    std::vector<ToolDefinition> tools;
    tools.reserve(tool_jsons.size());
    for (const std::string& text : tool_jsons) {
        OrderedJson tool;
        try {
            tool = OrderedJson::parse(text);
        } catch (const nlohmann::json::exception& error) {
            throw std::invalid_argument("tool definition is not valid JSON: " +
                                        std::string(error.what()));
        }
        if (!tool.is_object()) { throw std::invalid_argument("tool definition must be an object"); }
        const OrderedJson& function = tool.contains("function") ? tool.at("function") : tool;
        if (!function.is_object() || !function.contains("name") ||
            !function.at("name").is_string() || function.at("name").empty()) {
            throw std::invalid_argument("tool definition must contain a non-empty string name");
        }

        ToolDefinition parsed;
        parsed.name = function.at("name").get<std::string>();
        if (function.contains("description") && !function.at("description").is_null()) {
            if (!function.at("description").is_string()) {
                throw std::invalid_argument("tool description must be a string");
            }
            parsed.description = function.at("description").get<std::string>();
        }
        if (function.contains("parameters") && !function.at("parameters").is_null()) {
            if (!function.at("parameters").is_object()) {
                throw std::invalid_argument("tool parameters must be an object");
            }
            parsed.parameters = function.at("parameters");
        } else {
            parsed.parameters = OrderedJson{{"type", "object"},
                                            {"properties", OrderedJson::object()}};
        }
        tools.push_back(std::move(parsed));
    }
    return tools;
}

std::string reasoning_strength(const PromptOptions& options) {
    if (!options.enable_thinking) {
        if (options.reasoning_effort) {
            throw std::invalid_argument(
                "Muse instruct mode cannot be combined with a reasoning strength");
        }
        return "instruct";
    }
    switch (options.reasoning_effort.value_or(ReasoningEffort::High)) {
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    }
    throw std::invalid_argument("invalid Muse reasoning strength");
}

void replace_all(std::string& text, std::string_view before, std::string_view after) {
    std::size_t position = 0;
    while ((position = text.find(before, position)) != std::string::npos) {
        text.replace(position, before.size(), after);
        position += after.size();
    }
}

bool contains_reasoning_strength(std::string_view text) {
    std::string lower(text);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    return lower.find("reasoning strength") != std::string::npos;
}

std::string normalized_system_text(const ChatMessage& message) {
    validate_instruction(message);
    std::string text = message_text(message);
    replace_all(text, "Reasoning effort", "Reasoning strength");
    replace_all(text, "Reasoning Effort", "Reasoning Strength");
    replace_all(text, "reasoning effort", "reasoning strength");
    replace_all(text, "REASONING EFFORT", "REASONING STRENGTH");
    return text;
}

std::string tool_namespace(std::string_view name) {
    const std::size_t dot = name.find('.');
    return std::string(name.substr(0, dot));
}

std::string render_tool_definitions(const std::vector<ToolDefinition>& tools) {
    std::string rendered;
    rendered += "In this environment you have access to a set of tools you can use to answer the "
                "user's question.\n\n";
    rendered +=
        "You can invoke a function by writing a \"<atem:function_calls>\" block like the "
        "following:\n"
        "<atem:function_calls>\n"
        "<atem:invoke name=\"$FUNCTION_NAME\">\n"
        "<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n"
        "...\n"
        "</atem:invoke>\n"
        "</atem:function_calls>\n\n"
        "String and scalar parameters should be specified as is, while lists and objects should "
        "use JSON format. Note that spaces for string values are not stripped. The output is not "
        "expected to be valid XML and is parsed with regular expressions.\n"
        "Here are the functions available in JSONSchema format:\n"
        "// Tool metadata\n";

    std::unordered_set<std::string> seen_namespaces;
    for (const ToolDefinition& tool : tools) {
        const std::string name_space = tool_namespace(tool.name);
        if (seen_namespaces.insert(name_space).second) {
            rendered += tojson_text(
                OrderedJson{{"name", name_space}, {"description", std::string{}}});
            rendered += '\n';
        }
    }
    rendered += "// Function schemas";
    for (const ToolDefinition& tool : tools) {
        rendered += '\n';
        rendered += tojson_text(OrderedJson{{"name", tool.name},
                                            {"description", tool.description},
                                            {"parameters", tool.parameters}});
    }
    rendered +=
        "\n\nHere's an example of how to call a function in the tool set:\n"
        "(If the tool namespace is not specified, invoke the function directly as "
        "`example_function_name` rather than `example_tool_name.example_function_name`)\n\n"
        "to=example_tool_name.example_function_name\n\n"
        "<atem:function_calls>\n"
        "<atem:invoke name=\"example_tool_name.example_function_name\">\n"
        "<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n"
        "<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\n"
        "that can span\n"
        "\"multiple\" lines\n"
        "</atem:parameter>\n"
        "</atem:invoke>\n"
        "</atem:function_calls>";
    return rendered;
}

std::string render_system_meta(const std::vector<ToolDefinition>& tools) {
    std::string rendered = "# Valid recipients: \"self\"";
    std::unordered_set<std::string> seen_namespaces;
    for (const ToolDefinition& tool : tools) {
        const std::string name_space = tool_namespace(tool.name);
        if (seen_namespaces.insert(name_space).second) {
            rendered += ", \"";
            rendered += name_space;
            rendered += ".*\"";
        }
    }
    rendered += ", \"user\".";
    return rendered;
}

void append_system_suffix(std::string& rendered, std::string_view system_text,
                          std::string_view strength, const std::vector<ToolDefinition>& tools) {
    if (!contains_reasoning_strength(system_text)) {
        rendered += "\n\nReasoning strength: ";
        rendered += strength;
        rendered += '.';
    }
    if (!tools.empty()) {
        rendered += "\n\n";
        rendered += render_tool_definitions(tools);
    }
    rendered += "\n\n";
    rendered += render_system_meta(tools);
    rendered += "<|eot|>";
}

std::string render_atem(const ToolCall& call) {
    OrderedJson arguments;
    try {
        arguments = OrderedJson::parse(call.arguments_json);
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("tool call arguments are not valid JSON: " +
                                    std::string(error.what()));
    }
    if (!arguments.is_object()) {
        throw std::invalid_argument("tool call arguments must encode a JSON object");
    }

    std::string rendered = "<atem:function_calls>\n<atem:invoke name=\"" + call.name + "\">\n";
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
        rendered += "<atem:parameter name=\"";
        rendered += it.key();
        rendered += "\">";
        if (it.value().is_string()) {
            rendered += it.value().get<std::string>();
        } else {
            rendered += tojson_text(it.value());
        }
        rendered += "</atem:parameter>\n";
    }
    rendered += "</atem:invoke>\n</atem:function_calls>";
    return rendered;
}

std::string resolve_tool_name(const std::vector<ChatMessage>& messages,
                              const ChatMessage& tool_message) {
    for (const ChatMessage& message : messages) {
        for (const ToolCall& call : message.tool_calls) {
            if (!tool_message.tool_call_id.empty() && call.id == tool_message.tool_call_id) {
                return call.name;
            }
        }
    }
    return tool_message.tool_call_id;
}

} // namespace

std::string render_chat(const std::vector<ChatMessage>& messages, const PromptOptions& options,
                        std::string_view current_date) {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }
    const std::vector<ToolDefinition> tools = parse_tools(options.tool_jsons);
    const std::string strength              = reasoning_strength(options);

    bool has_system = false;
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::System) { has_system = true; }
    }

    std::string rendered = "<|begin_of_text|>";
    if (!has_system) {
        rendered += "<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: "
                    "2026-01-04.\nCurrent date: ";
        rendered += current_date;
        rendered += ".\n\nReasoning strength: ";
        rendered += strength;
        rendered += '.';
        if (!tools.empty()) {
            rendered += "\n\n";
            rendered += render_tool_definitions(tools);
        }
        rendered += "\n\n";
        rendered += render_system_meta(tools);
        rendered += "<|eot|>";
    }

    for (std::size_t index = 0; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        const bool same_next_role =
            index + 1 < messages.size() && messages[index + 1].role == message.role;
        const std::string_view end_token = same_next_role ? "<|eom|>" : "<|eot|>";

        switch (message.role) {
        case ChatRole::System: {
            std::string text = normalized_system_text(message);
            rendered += "<|start|>system<|message|>";
            rendered += text;
            append_system_suffix(rendered, text, strength, tools);
            break;
        }
        case ChatRole::Developer:
            validate_instruction(message);
            rendered += "<|start|>system<|message|>";
            rendered += message_text(message);
            rendered += "<|eot|>";
            break;
        case ChatRole::User:
            if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
                !message.tool_call_id.empty()) {
                throw std::invalid_argument("user messages may contain only prompt content");
            }
            rendered += "<|start|>user<|message|>";
            rendered += message_text(message);
            rendered += "<|eot|>";
            break;
        case ChatRole::Tool: {
            if (!message.reasoning_content.empty() || !message.tool_calls.empty()) {
                throw std::invalid_argument("tool messages cannot contain reasoning or tool calls");
            }
            const std::string name = resolve_tool_name(messages, message);
            rendered += "<|start|>tool ";
            rendered += name;
            rendered += "<|message|><tool_output name=\"";
            rendered += name;
            rendered += "\">\n";
            rendered += message_text(message);
            rendered += "\n</tool_output><|eot|>";
            break;
        }
        case ChatRole::Assistant:
            if (!message.tool_call_id.empty()) {
                throw std::invalid_argument("assistant messages cannot carry a tool_call_id");
            }
            if (!message.reasoning_content.empty()) {
                rendered += "<|start|>assistant to=self<|message|>";
                rendered += message.reasoning_content;
                rendered += "<|eom|>";
            }
            if (!message.tool_calls.empty()) {
                for (std::size_t call_index = 0; call_index < message.tool_calls.size();
                     ++call_index) {
                    const ToolCall& call = message.tool_calls[call_index];
                    if (call.name.empty()) {
                        throw std::invalid_argument("assistant tool call name must not be empty");
                    }
                    rendered += "<|start|>assistant to=";
                    rendered += call.name;
                    rendered += "<|message|>";
                    rendered += render_atem(call);
                    rendered += call_index + 1 == message.tool_calls.size() ? end_token : "<|eom|>";
                }
            } else {
                rendered += "<|start|>assistant to=user<|message|>";
                rendered += message_text(message);
                rendered += "<|eot|>";
            }
            break;
        }
    }
    if (options.add_generation_prompt) { rendered += "<|start|>assistant"; }
    return rendered;
}

PromptCapabilities prompt_capabilities() noexcept {
    PromptCapabilities capabilities;
    capabilities.enable_thinking                 = true;
    capabilities.reasoning_effort.low            = true;
    capabilities.reasoning_effort.medium         = true;
    capabilities.reasoning_effort.high           = true;
    capabilities.reasoning_effort.xhigh          = true;
    capabilities.reasoning_effort.default_effort = ReasoningEffort::High;
    return capabilities;
}

} // namespace ginfer::targets::muse_glimmer_30b::frontend_internal
