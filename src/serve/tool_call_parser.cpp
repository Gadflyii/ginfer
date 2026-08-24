#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string_view>

namespace ginfer::serve {
namespace {

using Json = nlohmann::json;

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-' && c != '.') { return false; }
    }
    return true;
}

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

bool parse_parameter(std::string_view inner, std::size_t& pos, Json& args) {
    constexpr std::string_view kParamOpen  = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    if (!starts_with_at(inner, pos, kParamOpen)) { return false; }
    const std::size_t name_begin = pos + kParamOpen.size();
    const std::size_t name_end   = inner.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
    pos                         = name_end + 1;
    const std::size_t value_end = inner.find(kParamClose, pos);
    if (value_end == std::string_view::npos) { return false; }
    const std::string raw_value = trim_ascii(inner.substr(pos, value_end - pos));
    Json parsed                 = Json::parse(raw_value, nullptr, false);
    args[key]                   = parsed.is_discarded() ? Json(raw_value) : parsed;
    pos                         = value_end + kParamClose.size();
    return true;
}

bool parse_one_tool_call(std::string_view block, std::size_t max_name_length, ToolCall& out) {
    constexpr std::string_view kFunctionOpen  = "<function=";
    constexpr std::string_view kFunctionClose = "</function>";
    std::size_t pos                           = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kFunctionOpen)) { return false; }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos = name_end + 1;

    const std::size_t function_end = block.find(kFunctionClose, pos);
    if (function_end == std::string_view::npos) { return false; }
    const std::string_view params = block.substr(pos, function_end - pos);
    Json args                     = Json::object();
    std::size_t param_pos         = 0;
    for (;;) {
        skip_ws(params, param_pos);
        if (param_pos >= params.size()) { break; }
        if (!parse_parameter(params, param_pos, args)) { return false; }
    }

    pos = function_end + kFunctionClose.size();
    skip_ws(block, pos);
    if (pos != block.size()) { return false; }

    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

ParsedToolCallOutput parse_qwen_format(const std::string& text,
                                       std::size_t max_tool_name_length) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }
        if (!starts_with_at(text, pos, kToolOpen)) { return fallback(text); }
        const std::size_t inner_begin = pos + kToolOpen.size();
        const std::size_t close       = text.find(kToolClose, inner_begin);
        if (close == std::string::npos) { return fallback(text); }
        ToolCall call;
        if (!parse_one_tool_call(std::string_view(text).substr(inner_begin, close - inner_begin),
                                 max_tool_name_length, call)) {
            return fallback(text);
        }
        out.tool_calls.push_back(std::move(call));
        pos = close + kToolClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

bool parse_atem_parameter(std::string_view block, std::size_t& pos, Json& args) {
    constexpr std::string_view kParameterOpen  = "<atem:parameter name=\"";
    constexpr std::string_view kAttributeClose = "\">";
    constexpr std::string_view kParameterClose = "</atem:parameter>";
    if (!starts_with_at(block, pos, kParameterOpen)) { return false; }
    const std::size_t name_begin = pos + kParameterOpen.size();
    const std::size_t name_end   = block.find(kAttributeClose, name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string key = std::string(block.substr(name_begin, name_end - name_begin));
    pos                   = name_end + kAttributeClose.size();
    const std::size_t value_end = block.find(kParameterClose, pos);
    if (value_end == std::string_view::npos) { return false; }

    // Muse explicitly preserves spaces in non-JSON scalar values. JSON itself permits
    // surrounding whitespace, so the same represented bytes can be tried as JSON first.
    const std::string raw_value(block.substr(pos, value_end - pos));
    Json parsed = Json::parse(raw_value, nullptr, false);
    args[key]   = parsed.is_discarded() ? Json(raw_value) : parsed;
    pos         = value_end + kParameterClose.size();
    return true;
}

bool parse_atem_invoke(std::string_view block, std::size_t& pos, std::size_t max_name_length,
                       ToolCall& out) {
    constexpr std::string_view kInvokeOpen     = "<atem:invoke name=\"";
    constexpr std::string_view kAttributeClose = "\">";
    constexpr std::string_view kInvokeClose    = "</atem:invoke>";
    if (!starts_with_at(block, pos, kInvokeOpen)) { return false; }
    const std::size_t name_begin = pos + kInvokeOpen.size();
    const std::size_t name_end   = block.find(kAttributeClose, name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos = name_end + kAttributeClose.size();

    Json args = Json::object();
    for (;;) {
        skip_ws(block, pos);
        if (starts_with_at(block, pos, kInvokeClose)) { break; }
        if (!parse_atem_parameter(block, pos, args)) { return false; }
    }
    pos += kInvokeClose.size();
    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

ParsedToolCallOutput parse_atem_format(const std::string& text,
                                       std::size_t max_tool_name_length) {
    constexpr std::string_view kCallsOpen  = "<atem:function_calls>";
    constexpr std::string_view kCallsClose = "</atem:function_calls>";
    const std::size_t first = text.find(kCallsOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));
    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }
        if (!starts_with_at(text, pos, kCallsOpen)) { return fallback(text); }
        pos += kCallsOpen.size();

        const std::size_t calls_before = out.tool_calls.size();
        for (;;) {
            skip_ws(text, pos);
            if (starts_with_at(text, pos, kCallsClose)) { break; }
            ToolCall call;
            if (!parse_atem_invoke(text, pos, max_tool_name_length, call)) {
                return fallback(text);
            }
            out.tool_calls.push_back(std::move(call));
        }
        if (out.tool_calls.size() == calls_before) { return fallback(text); }
        pos += kCallsClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

} // namespace

ParsedToolCallOutput parse_tool_call_output(const std::string& text,
                                            std::size_t max_tool_name_length) {
    constexpr std::string_view kQwenOpen = "<tool_call>";
    constexpr std::string_view kAtemOpen = "<atem:function_calls>";
    const std::size_t qwen = text.find(kQwenOpen);
    const std::size_t atem = text.find(kAtemOpen);
    if (atem != std::string::npos && (qwen == std::string::npos || atem < qwen)) {
        return parse_atem_format(text, max_tool_name_length);
    }
    if (qwen != std::string::npos) { return parse_qwen_format(text, max_tool_name_length); }
    return fallback(text);
}

std::string ToolCallStreamFilter::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    if (text.empty()) { return {}; }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    constexpr std::array<std::string_view, 2> kToolOpen = {
        "<tool_call>",
        "<atem:function_calls>",
    };
    pending_.append(text);
    std::size_t marker = std::string::npos;
    for (const std::string_view candidate : kToolOpen) {
        marker = std::min(marker, pending_.find(candidate));
    }
    if (marker != std::string::npos) {
        std::size_t safe_end = marker;
        while (safe_end != 0 &&
               std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
            --safe_end;
        }
        std::string visible = pending_.substr(0, safe_end);
        tool_region_        = pending_.substr(safe_end);
        pending_.clear();
        saw_tool_marker_ = true;
        emitted_bytes_ += visible.size();
        return visible;
    }

    std::size_t prefix = 0;
    for (const std::string_view candidate : kToolOpen) {
        prefix = std::max(prefix, longest_suffix_prefix(pending_, candidate));
    }
    std::size_t safe_end     = pending_.size() - prefix;
    while (safe_end != 0 && std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
        --safe_end;
    }
    std::string visible = pending_.substr(0, safe_end);
    pending_.erase(0, safe_end);
    emitted_bytes_ += visible.size();
    return visible;
}

std::string ToolCallStreamFilter::finish(bool is_tool_call_response) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    finished_ = true;
    if (is_tool_call_response) {
        pending_.clear();
        tool_region_.clear();
        return {};
    }
    std::string tail = std::move(pending_);
    tail += tool_region_;
    tool_region_.clear();
    emitted_bytes_ += tail.size();
    return tail;
}

} // namespace ginfer::serve
