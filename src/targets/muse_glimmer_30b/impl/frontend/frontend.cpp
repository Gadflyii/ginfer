#include <ginfer/targets/muse_glimmer_30b/package.h>

#include "targets/muse_glimmer_30b/impl/config.h"
#include "targets/muse_glimmer_30b/impl/frontend/chat_template.h"
#include "targets/muse_glimmer_30b/impl/frontend/tokenizer.h"
#include "targets/muse_glimmer_30b/impl/load/bindings.h"
#include "text/unicode.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ginfer::targets::muse_glimmer_30b::detail {
namespace {

using Clock = std::chrono::steady_clock;
namespace fi = frontend_internal;

void check_control(const PreparationControl& control, const char* stage) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           std::string("prompt preparation cancelled during ") + stage);
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           std::string("prompt preparation exceeded its deadline during ") + stage);
    }
}

std::string current_date_utc() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char buffer[16];
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", &utc) == 0) {
        return "2026-08-19";
    }
    return buffer;
}

StopPolicy merge_stop(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                        std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)ginfer::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    return static_cast<std::uint32_t>(count);
}

constexpr TokenId kEomToken     = 200007;
constexpr TokenId kStartToken   = 200022;
constexpr TokenId kMessageToken = 200023;

void validate_protocol_tokens(const fi::Tokenizer& tokenizer) {
    constexpr std::array<std::pair<std::string_view, TokenId>, 6> expected{{
        {"<|begin_of_text|>", 200000},
        {"<|end_of_text|>", 200001},
        {"<|eom|>", kEomToken},
        {"<|eot|>", 200008},
        {"<|start|>", kStartToken},
        {"<|message|>", kMessageToken},
    }};
    for (const auto& [text, token] : expected) {
        const std::vector<int> encoded = tokenizer.encode(text);
        if (encoded.size() != 1 || encoded.front() != token ||
            !tokenizer.is_special_token(token)) {
            throw std::invalid_argument(
                "artifact tokenizer does not match registered Muse ATEM token IDs");
        }
    }
}

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

std::size_t valid_utf8_prefix_size(std::string_view bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto lead         = static_cast<unsigned char>(bytes[offset]);
        std::size_t length      = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum   = 0;
        if (lead <= 0x7fU) {
            length    = 1;
            codepoint = lead;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length    = 2;
            codepoint = lead & 0x1fU;
            minimum   = 0x80U;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length    = 3;
            codepoint = lead & 0x0fU;
            minimum   = 0x800U;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length    = 4;
            codepoint = lead & 0x07U;
            minimum   = 0x10000U;
        } else {
            throw std::invalid_argument("invalid UTF-8 leading byte in generated token stream");
        }
        if (offset + length > bytes.size()) { return offset; }
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<unsigned char>(bytes[offset + index]);
            if ((byte & 0xc0U) != 0x80U) {
                throw std::invalid_argument(
                    "invalid UTF-8 continuation byte in generated token stream");
            }
            codepoint = (codepoint << 6U) | (byte & 0x3fU);
        }
        if (codepoint < minimum || (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
            codepoint > 0x10ffffU) {
            throw std::invalid_argument("invalid UTF-8 codepoint in generated token stream");
        }
        offset += length;
    }
    return offset;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

enum class AtemStage : std::uint8_t {
    PlainContent,
    Header,
    BetweenMessages,
    Reasoning,
    Content,
};

struct DecoderState {
    std::string utf8_pending;
    std::string header_pending;
    std::array<std::string, 2> stop_pending;
    AtemStage stage                 = AtemStage::PlainContent;
    bool terminal                   = false;
    std::uint64_t decoded_bytes     = 0;
    std::uint32_t reasoning_tokens  = 0;
};

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            PublishedOutput candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

void close_channel(DecoderState& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
}

bool body_stage(AtemStage stage) noexcept {
    return stage == AtemStage::Reasoning || stage == AtemStage::Content ||
           stage == AtemStage::PlainContent;
}

OutputChannel stage_channel(AtemStage stage) noexcept {
    return stage == AtemStage::Reasoning ? OutputChannel::Reasoning : OutputChannel::Content;
}

void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       PublishedOutput& emitted, std::uint32_t committed_tokens,
                       StopMatch* best_match) {
    if (state.stage == AtemStage::Header || state.stage == AtemStage::BetweenMessages) {
        state.header_pending.append(text);
        return;
    }
    feed_channel(state, stage_channel(state.stage), text, policy, emitted, committed_tokens,
                 best_match);
}

void feed_token_bytes(DecoderState& state, std::string bytes, const StopPolicy& policy,
                      PublishedOutput& emitted, std::uint32_t committed_tokens,
                      StopMatch* best_match) {
    state.utf8_pending += bytes;
    const std::size_t valid = valid_utf8_prefix_size(state.utf8_pending);
    if (valid == 0) { return; }
    const std::string text = state.utf8_pending.substr(0, valid);
    state.utf8_pending.erase(0, valid);
    feed_decoded_text(state, text, policy, emitted, committed_tokens, best_match);
}

void flush_incomplete_utf8(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                           std::uint32_t committed_tokens) {
    if (state.utf8_pending.empty()) { return; }
    state.utf8_pending.clear();
    feed_decoded_text(state, "\xef\xbf\xbd", policy, emitted, committed_tokens, nullptr);
}

std::string trim_ascii(std::string text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return text.substr(begin, end - begin);
}

void open_atem_message(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                       std::uint32_t committed_tokens) {
    flush_incomplete_utf8(state, policy, emitted, committed_tokens);
    const std::string header = trim_ascii(std::move(state.header_pending));
    state.header_pending.clear();

    constexpr std::string_view kAssistant = "assistant";
    constexpr std::string_view kRecipient = " to=";
    if (header == kAssistant) {
        state.stage = AtemStage::Content;
        return;
    }
    if (header.starts_with(kAssistant) && header.size() > kAssistant.size() + kRecipient.size() &&
        std::string_view(header).substr(kAssistant.size(), kRecipient.size()) == kRecipient) {
        const std::string_view recipient =
            std::string_view(header).substr(kAssistant.size() + kRecipient.size());
        state.stage = recipient == "self" ? AtemStage::Reasoning : AtemStage::Content;
        return;
    }

    // A malformed header is data, not a reason to silently lose generated bytes.
    state.stage = AtemStage::Content;
    feed_channel(state, OutputChannel::Content, header, policy, emitted, committed_tokens, nullptr);
}

void close_atem_message(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                        std::uint32_t committed_tokens) {
    flush_incomplete_utf8(state, policy, emitted, committed_tokens);
    if (body_stage(state.stage)) { close_channel(state, stage_channel(state.stage), emitted); }
    state.stage = AtemStage::BetweenMessages;
    state.header_pending.clear();
}

void feed_protocol_token(DecoderState& state, TokenId token, std::string bytes,
                         const StopPolicy& policy, PublishedOutput& emitted,
                         std::uint32_t committed_tokens, StopMatch* best_match) {
    if (state.stage == AtemStage::PlainContent) {
        feed_token_bytes(state, std::move(bytes), policy, emitted, committed_tokens, best_match);
        return;
    }
    if (token == kStartToken) {
        close_atem_message(state, policy, emitted, committed_tokens);
        state.stage = AtemStage::Header;
        return;
    }
    if (token == kMessageToken &&
        (state.stage == AtemStage::Header || state.stage == AtemStage::BetweenMessages)) {
        open_atem_message(state, policy, emitted, committed_tokens);
        return;
    }
    if (token == kEomToken) {
        close_atem_message(state, policy, emitted, committed_tokens);
        return;
    }
    feed_token_bytes(state, std::move(bytes), policy, emitted, committed_tokens, best_match);
}

void terminalize(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                 std::uint32_t committed_tokens) {
    flush_incomplete_utf8(state, policy, emitted, committed_tokens);
    if (state.stage == AtemStage::Header || state.stage == AtemStage::BetweenMessages) {
        if (!state.header_pending.empty()) {
            feed_channel(state, OutputChannel::Content, state.header_pending, policy, emitted,
                         committed_tokens, nullptr);
            state.header_pending.clear();
        }
        close_channel(state, OutputChannel::Content, emitted);
    } else {
        close_channel(state, stage_channel(state.stage), emitted);
    }
    state.stop_pending = {};
    state.terminal     = true;
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.header_pending.clear();
    state.stop_pending = {};
    state.terminal     = true;
    return state;
}

} // namespace

PublishedOutput::PublishedOutput(PublishedOutput&& other) noexcept
    : values_(std::move(other.values_)), size_(std::exchange(other.size_, 0)) {}

PublishedOutput& PublishedOutput::operator=(PublishedOutput&& other) noexcept {
    if (this != &other) {
        values_ = std::move(other.values_);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void PublishedOutput::clear() noexcept {
    for (std::size_t index = 0; index < size_; ++index) { values_[index] = {}; }
    size_ = 0;
}

void PublishedOutput::push_back(OutputDelta value) {
    if (size_ == values_.size()) {
        throw std::logic_error("output decoder produced more than two channel transitions");
    }
    values_[size_++] = std::move(value);
}

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool parse_atem_output)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          preserve_special(output.raw || output.preserve_special_tokens) {
        if (parse_atem_output && !output.raw) {
            state.stage          = AtemStage::Header;
            state.header_pending = "assistant";
        }
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    bool preserve_special = false;
    DecoderState state;
    DecoderState preview_state;
    PublishedOutput preview_output;
    bool preview_ready = false;
};

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;
OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

runtime::OutputDecision OutputSession::preview(std::span<const TokenId> tokens,
                                               std::uint32_t budget_remaining,
                                               FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state = impl_->state;
    impl_->preview_output.clear();

    const auto complete = [&](std::uint32_t count, FinishReason reason) {
        impl_->preview_ready = true;
        return runtime::OutputDecision{.accepted_tokens = count, .finish_reason = reason};
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count = static_cast<std::uint32_t>(index + 1);
        const TokenId token       = tokens[index];
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("generated token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }

        if (impl_->preview_state.stage == AtemStage::Reasoning) {
            ++impl_->preview_state.reasoning_tokens;
        }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();
        if (stop_token && !impl_->policy.publish_stop_token) {
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            return complete(count, FinishReason::StopToken);
        }

        StopMatch match;
        const std::string bytes =
            impl_->tokenizer->decode_token_bytes(token, !impl_->preserve_special);
        feed_protocol_token(impl_->preview_state, token, bytes, impl_->policy,
                            impl_->preview_output, count, &match);

        if (match.found) {
            impl_->preview_state  = terminal_state(std::move(impl_->preview_state));
            impl_->preview_output = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            return complete(count, FinishReason::StopToken);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == budget_remaining) {
        terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
        return complete(count, limit_reason);
    }
    return complete(count, FinishReason::None);
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state = impl_->state;
    impl_->preview_output.clear();
    terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, 0);
    impl_->preview_ready = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() noexcept {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    using std::swap;
    swap(impl_->state, impl_->preview_state);
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->preview_ready = false;
    return output;
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

PreparedPrompt::PreparedPrompt() noexcept = default;
PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr) { throw std::logic_error("prepared prompt is empty"); }
    return PromptSummary{.prompt_tokens = checked_token_count(data_->token_ids.size()),
                         .has_media     = false};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    if (data_ == nullptr) { return {}; }
    return data_->prepare;
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

std::span<const TokenId> PreparedPrompt::token_ids() const noexcept {
    if (data_ == nullptr) { return {}; }
    return data_->token_ids;
}

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, const EngineOptions& options)
        : tokenizer(std::make_shared<fi::Tokenizer>(fi::TokenizerResources{
              .tokenizer_json          = resources.tokenizer_json,
              .tokenizer_config_json   = resources.tokenizer_config_json,
              .generation_config_json  = resources.generation_config_json,
          })),
          max_context(options.max_context),
          default_stop(merge_stop(*tokenizer, StopPolicy{})) {
        validate_protocol_tokens(*tokenizer);
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    std::uint32_t max_context = 0;
    StopPolicy default_stop;
};

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}
Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

Frontend make_frontend(const FrontendResources& resources, const EngineOptions& options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, options));
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    check_control(control, "start");
    const auto start = Clock::now();
    const bool parse_atem_output = input.options.add_generation_prompt;
    const std::string rendered =
        fi::render_chat(input.messages, input.options, current_date_utc());
    check_control(control, "render");
    const auto tokenize_started = Clock::now();
    std::vector<int> ids        = impl_->tokenizer->encode(rendered);
    auto prepared               = std::make_unique<PreparedPromptData>();
    prepared->token_ids.assign(ids.begin(), ids.end());
    prepared->parse_atem_output = parse_atem_output;
    prepared->prefix_identity_reusable = true;
    prepared->prepare.tokenize_seconds =
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    prepared->prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    check_control(control, "tokenization");
    if (prepared->token_ids.size() > impl_->max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prepared prompt has " + std::to_string(prepared->token_ids.size()) +
                               " tokens, exceeding Engine max_context " +
                               std::to_string(impl_->max_context));
    }
    return PreparedPrompt(std::move(prepared));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    return prepare(std::move(input), control).summary().prompt_tokens;
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    auto prepared         = std::make_unique<PreparedPromptData>();
    prepared->token_ids   = std::move(token_ids);
    prepared->prefix_identity_reusable = allow_prefix_identity;
    prepared->prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

PromptCapabilities Frontend::prompt_capabilities() const noexcept {
    return fi::prompt_capabilities();
}

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    StopPolicy policy = merge_stop(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(std::make_unique<OutputSession::Impl>(
        impl_->tokenizer, std::move(policy), output, prompt.data_->parse_atem_output));
}

} // namespace ginfer::targets::muse_glimmer_30b::detail
