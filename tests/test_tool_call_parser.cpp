#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_single_call() {
    const ginfer::serve::ParsedToolCallOutput parsed =
        ginfer::serve::parse_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const ginfer::serve::ParsedToolCallOutput parsed = ginfer::serve::parse_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const ginfer::serve::ParsedToolCallOutput parsed =
        ginfer::serve::parse_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const ginfer::serve::ParsedToolCallOutput parsed =
        ginfer::serve::parse_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ginfer::serve::ParsedToolCallOutput anthropic =
        ginfer::serve::parse_tool_call_output(text, 128);
    const ginfer::serve::ParsedToolCallOutput openai =
        ginfer::serve::parse_tool_call_output(text, 64);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ginfer::serve::ParsedToolCallOutput too_long =
        ginfer::serve::parse_tool_call_output(too_long_text, 128);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_incremental_filter_valid_tool() {
    ginfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish(true);
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    ginfer::serve::ToolCallStreamFilter malformed;
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish(false);

    ginfer::serve::ToolCallStreamFilter normal;
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish(false);

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

int test_muse_atem_calls() {
    const std::string text =
        "Using the weather tool.  \n"
        "<atem:function_calls>\n"
        "<atem:invoke name=\"weather.lookup\">\n"
        "<atem:parameter name=\"city\"> New York </atem:parameter>\n"
        "<atem:parameter name=\"days\">2</atem:parameter>\n"
        "<atem:parameter name=\"units\">[\"C\", \"F\"]</atem:parameter>\n"
        "</atem:invoke>\n"
        "<atem:invoke name=\"summarize\">\n"
        "<atem:parameter name=\"verbose\">true</atem:parameter>\n"
        "</atem:invoke>\n"
        "</atem:function_calls>";
    const ginfer::serve::ParsedToolCallOutput parsed =
        ginfer::serve::parse_tool_call_output(text, 64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "Muse ATEM calls were not recognized");
    failures += check(parsed.content == "Using the weather tool.",
                      "Muse ATEM content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 2, "Muse ATEM repeated invokes were not parsed");
    if (parsed.tool_calls.size() == 2) {
        failures += check(parsed.tool_calls[0].name == "weather.lookup" &&
                              parsed.tool_calls[1].name == "summarize",
                          "Muse ATEM function names were not preserved");
        const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
        const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
        failures += check(first.at("city") == " New York ",
                          "Muse ATEM plain-string spaces were stripped");
        failures += check(first.at("days") == 2 && first.at("units").at(1) == "F" &&
                              second.at("verbose") == true,
                          "Muse ATEM JSON scalar values were not parsed");
    }
    return failures;
}

int test_muse_atem_fallback_and_streaming() {
    const std::string malformed =
        "prefix\n<atem:function_calls><atem:invoke name=\"broken\">";
    const ginfer::serve::ParsedToolCallOutput parsed =
        ginfer::serve::parse_tool_call_output(malformed, 64);

    ginfer::serve::ToolCallStreamFilter valid;
    std::string visible;
    visible += valid.feed("Using it.  \n<atem:funct");
    visible += valid.feed("ion_calls>\n<atem:invoke name=\"f\">");
    visible += valid.feed("</atem:invoke>\n</atem:function_calls>");
    visible += valid.finish(true);

    ginfer::serve::ToolCallStreamFilter invalid;
    std::string restored;
    restored += invalid.feed(malformed.substr(0, 18));
    restored += invalid.feed(malformed.substr(18));
    restored += invalid.finish(false);

    int failures = 0;
    failures += check(!parsed.is_tool_call_response && parsed.content == malformed,
                      "malformed Muse ATEM output did not fall back verbatim");
    failures += check(visible == "Using it.",
                      "Muse ATEM stream filter leaked or lost the visible prefix");
    failures += check(restored == malformed,
                      "Muse ATEM stream fallback did not restore buffered bytes");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_configured_name_limit();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_muse_atem_calls();
    failures += test_muse_atem_fallback_and_streaming();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
