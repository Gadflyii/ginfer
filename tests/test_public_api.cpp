#include "ginfer/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<ginfer::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<ginfer::PreparedPrompt>);
static_assert(std::is_move_constructible_v<ginfer::Engine>);
static_assert(!std::is_copy_constructible_v<ginfer::Engine>);

int main() {
    const ginfer::EngineOptions options;
    return options.enable_vision ? 1 : 0;
}
