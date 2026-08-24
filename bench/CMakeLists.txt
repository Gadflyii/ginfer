function(ginfer_add_op_bench name)
  cmake_parse_arguments(arg "" "" "SOURCES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE ginfer_ops)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

ginfer_add_op_bench(ginfer_silu_mul_bench SOURCES ops/silu_mul_bench.cu)
ginfer_add_op_bench(ginfer_residual_add_bench SOURCES ops/residual_add_bench.cu)
ginfer_add_op_bench(ginfer_sigmoid_mul_bench SOURCES ops/sigmoid_mul_bench.cu)
ginfer_add_op_bench(ginfer_rmsnorm_bench SOURCES ops/rmsnorm_bench.cu)
ginfer_add_op_bench(ginfer_l2norm_bench SOURCES ops/l2norm_bench.cu)
ginfer_add_op_bench(ginfer_gdn_gating_bench SOURCES ops/gdn_gating_bench.cu)
ginfer_add_op_bench(ginfer_gdn_gating_proj_bench SOURCES ops/gdn_gating_proj_bench.cu)
ginfer_add_op_bench(ginfer_gated_delta_net_bench SOURCES ops/gated_delta_net_bench.cu)
ginfer_add_op_bench(ginfer_gdn_replay_bench SOURCES ops/gdn_replay_bench.cu)
ginfer_add_op_bench(ginfer_gdn_layer_bench SOURCES ops/gdn_layer_bench.cu)
ginfer_add_op_bench(ginfer_rope_bench SOURCES ops/rope_bench.cu)
ginfer_add_op_bench(ginfer_add_bias_bench SOURCES ops/add_bias_bench.cu)
ginfer_add_op_bench(ginfer_gelu_bench SOURCES ops/gelu_bench.cu)
ginfer_add_op_bench(ginfer_layer_norm_bench SOURCES ops/layer_norm_bench.cu)
ginfer_add_op_bench(ginfer_vision_pos_embed_bench SOURCES ops/vision_pos_embed_bench.cu)
ginfer_add_op_bench(ginfer_scatter_bench SOURCES ops/scatter_bench.cu)
ginfer_add_op_bench(ginfer_packed_softmax_attention_bench
  SOURCES ops/packed_softmax_attention_bench.cu)
ginfer_add_op_bench(ginfer_embedding_bench SOURCES ops/embedding_bench.cu)
ginfer_add_op_bench(ginfer_position_bench SOURCES ops/position_bench.cu)
ginfer_add_op_bench(ginfer_cast_bench SOURCES ops/cast_bench.cu)
ginfer_add_op_bench(ginfer_argmax_bench SOURCES ops/argmax_bench.cu)
ginfer_add_op_bench(ginfer_causal_conv1d_silu_bench SOURCES ops/causal_conv1d_silu_bench.cu)
ginfer_add_op_bench(ginfer_linear_bench SOURCES ops/linear_bench.cu)
ginfer_add_op_bench(ginfer_output_stage_bench SOURCES ops/output_stage_bench.cu)
ginfer_add_op_bench(ginfer_gdn_input_proj_bench SOURCES ops/gdn_input_proj_bench.cu)
ginfer_add_op_bench(ginfer_gdn_input_proj_conv_snapshot_bench
  SOURCES ops/gdn_input_proj_conv_snapshot_bench.cu)
ginfer_add_op_bench(ginfer_attn_input_proj_bench SOURCES ops/attn_input_proj_bench.cu)
ginfer_add_op_bench(ginfer_w8_linear_swiglu_bench SOURCES ops/w8_linear_swiglu_bench.cu)
ginfer_add_op_bench(ginfer_nvfp4_linear_swiglu_bench SOURCES ops/nvfp4_linear_swiglu_bench.cu)
ginfer_add_op_bench(ginfer_fp8_linear_swiglu_bench SOURCES ops/fp8_linear_swiglu_bench.cu)
ginfer_add_op_bench(ginfer_q4_linear_swiglu_bench SOURCES ops/q4_linear_swiglu_bench.cu)
ginfer_add_op_bench(ginfer_w8_linear_add_bench SOURCES ops/w8_linear_add_bench.cu)
ginfer_add_op_bench(ginfer_q5_linear_add_bench SOURCES ops/q5_linear_add_bench.cu)
ginfer_add_op_bench(ginfer_bf16_linear_add_bench SOURCES ops/bf16_linear_add_bench.cu)
ginfer_add_op_bench(ginfer_nvfp4_linear_add_bench SOURCES ops/nvfp4_linear_add_bench.cu)
ginfer_add_op_bench(ginfer_fp8_linear_add_bench SOURCES ops/fp8_linear_add_bench.cu)
ginfer_add_op_bench(ginfer_causal_softmax_attention_bench
  SOURCES ops/causal_softmax_attention_bench.cu)
ginfer_add_op_bench(ginfer_context_softmax_attention_bench
  SOURCES ops/context_softmax_attention_bench.cu)
ginfer_add_op_bench(ginfer_sliding_window_attention_bench
  SOURCES ops/sliding_window_attention_bench.cu)
ginfer_add_op_bench(ginfer_kv_cache_append_bench SOURCES ops/kv_cache_append_bench.cu)
ginfer_add_op_bench(ginfer_prepare_masked_block_bench
  SOURCES ops/prepare_masked_block_bench.cu)
ginfer_add_op_bench(ginfer_sampling_select_bench SOURCES ops/sampling_select_bench.cu)
ginfer_add_op_bench(ginfer_dflash2_accept_bench SOURCES ops/dflash2_accept_bench.cu)
ginfer_add_op_bench(ginfer_grouped_dynamic_conv_bench
  SOURCES ops/grouped_dynamic_conv_bench.cu)
ginfer_add_op_bench(ginfer_sparse_moe_bench SOURCES ops/sparse_moe_bench.cu)
ginfer_add_op_bench(ginfer_linear_pair_bench SOURCES ops/linear_pair_bench.cu)
ginfer_add_op_bench(ginfer_mtp_pack_bench SOURCES ops/mtp_pack_bench.cu)
ginfer_add_op_bench(ginfer_proposal_remap_bench SOURCES ops/proposal_remap_bench.cu)

# End-to-end product benchmark: public Engine API and native .ginfer artifacts only.
add_executable(ginfer_bench targets/qwen3_6_27b/ginfer_bench.cpp targets/qwen3_6_27b/ginfer_bench_support.cpp)
target_include_directories(ginfer_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_compile_definitions(ginfer_bench PRIVATE GINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ginfer_bench PRIVATE ginfer_engine CUDA::cudart)

# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Program facade and executes the production round schedule over a real .ginfer artifact.
add_executable(ginfer_qwen3_6_27b_mtp_round_bench
  targets/qwen3_6_27b/target_mtp_round_bench.cpp)
target_include_directories(ginfer_qwen3_6_27b_mtp_round_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ginfer_qwen3_6_27b_mtp_round_bench PRIVATE
  ginfer_engine ginfer_core)

# Complete production DFlash steady round: confirmed-context append, next proposal, target
# verification/acceptance, and host publication over a real 35B artifact.
add_executable(ginfer_qwen3_6_35b_a3b_dflash_round_bench
  targets/qwen3_6_35b_a3b/dflash_round_bench.cpp)
target_include_directories(ginfer_qwen3_6_35b_a3b_dflash_round_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ginfer_qwen3_6_35b_a3b_dflash_round_bench PRIVATE
  ginfer_engine ginfer_core)
