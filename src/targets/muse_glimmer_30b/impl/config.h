#pragma once

#include <cstdint>

namespace ginfer::targets::muse_glimmer_30b::detail {

struct TextConfig {
    static constexpr int hidden       = 6656;
    static constexpr int layers       = 52;
    static constexpr int intermediate = 19968;
    static constexpr int vocab        = 202048;

    static constexpr int query_heads = 32;
    static constexpr int kv_heads    = 2;
    static constexpr int head_dim    = 128;
    static constexpr int rotary_dim  = 128;

    static constexpr int query_size = query_heads * head_dim;
    static constexpr int kv_size    = kv_heads * head_dim;

    static constexpr int sliding_window = 2048;
    static constexpr float rms_epsilon  = 1.0e-5F;
    static constexpr float post_norm_eps = 1.0e-8F;
    static constexpr float rope_theta   = 500000.0F;
    static constexpr float qk_scale_factor = 3.87F;
    static constexpr float output_multiplier       = 0.19611613513818404F;
    static constexpr float final_logit_softcapping = 20.0F;

    static constexpr int bos_token_id = 200000;
    static constexpr int eos_token_id = 200001;
    static constexpr int eot_token_id = 200008;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return layer >= 3 && (layer - 3) % 4 == 0;
    }

    [[nodiscard]] static constexpr int full_attention_layers() { return layers / 4; }

    [[nodiscard]] static constexpr int sliding_attention_layers() {
        return layers - full_attention_layers();
    }
};

static_assert(TextConfig::full_attention_layers() == 13);
static_assert(TextConfig::sliding_attention_layers() == 39);
static_assert(TextConfig::query_size == 4096);
static_assert(TextConfig::kv_size == 256);

struct DFlash2Config {
    static constexpr int layers                 = 5;
    static constexpr int kv_heads               = 8;
    static constexpr int block_size             = 16;
    static constexpr int conv_kernel            = 2;
    static constexpr int conv_group             = 16;
    static constexpr int selector_rank          = 256;
    static constexpr int selector_top_k         = 16;
    static constexpr int kernel_projection_rows = 1664;
    static constexpr int mask_token             = 201818;
    static constexpr int cyclic_capacity        = 2048;
    static constexpr int captured_rows          = TextConfig::hidden * 5;
    static constexpr int target_layer_ids[5]    = {1, 13, 25, 37, 49};
    static constexpr float attention_scale      = 0.08838834764831845F;
};

inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
inline constexpr std::uint32_t kNativeContext         = 131072;
inline constexpr std::uint32_t kKvQuantGroup          = 64;
// 1/sqrt(128). qk_scale_factor is applied as the Q RMS-norm weight, matching llama.cpp.
inline constexpr float kAttentionScale = 1.0F / 11.313708498984761F;

} // namespace ginfer::targets::muse_glimmer_30b::detail
