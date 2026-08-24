#pragma once

#include <ginfer/types.h>

#include <string>
#include <string_view>
#include <vector>

namespace ginfer::targets::muse_glimmer_30b::frontend_internal {

[[nodiscard]] std::string render_chat(const std::vector<ChatMessage>& messages,
                                      const PromptOptions& options,
                                      std::string_view current_date);

[[nodiscard]] PromptCapabilities prompt_capabilities() noexcept;

} // namespace ginfer::targets::muse_glimmer_30b::frontend_internal
