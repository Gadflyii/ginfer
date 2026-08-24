#pragma once

#include <ginfer/targets/qwen3_6/frontend.h>
#include <ginfer/targets/qwen3_6/frontend_resources.h>
#include <ginfer/targets/qwen3_6/prepared_prompt.h>

namespace ginfer::targets::qwen3_6 {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ginfer::targets::qwen3_6
