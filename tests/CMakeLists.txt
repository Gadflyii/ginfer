find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(ginfer_public_api_test test_public_api.cpp)
target_include_directories(ginfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ginfer_public_api_test COMMAND ginfer_public_api_test)

function(ginfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE ginfer_core)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      GINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      GINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

function(ginfer_add_op_test name)
  ginfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ginfer_add_linear_test name source)
  ginfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES ginfer_ops)
endfunction()

function(ginfer_add_fused_linear_test name source common)
  ginfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES ginfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
ginfer_add_test(ginfer_artifact_reader_test
  SOURCES test_ginfer_artifact_reader.cpp
  LIBRARIES ginfer_artifact)
ginfer_add_test(ginfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES ginfer_artifact)
ginfer_add_test(ginfer_media_decode_test
  SOURCES test_media_decode.cpp
  LIBRARIES ginfer_media_decode)
ginfer_add_test(ginfer_device_test       SOURCES test_device.cpp)
ginfer_add_test(ginfer_gpu_capability_test SOURCES test_gpu_capability.cpp)
ginfer_add_test(ginfer_artifact_allowlist_test
  SOURCES test_artifact_allowlist.cpp
  LIBRARIES ginfer_engine)
target_include_directories(ginfer_artifact_allowlist_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export
  ${PROJECT_SOURCE_DIR}/src/targets/muse_glimmer_30b/export)
ginfer_add_test(ginfer_bench_dram_roofline_test SOURCES test_bench_dram_roofline.cpp)
target_include_directories(ginfer_bench_dram_roofline_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/ops)
target_compile_definitions(ginfer_bench_dram_roofline_test PRIVATE
  GINFER_BENCH_DRAM_ROOFLINE_MAP_ONLY)
ginfer_add_test(ginfer_decode_graph_test SOURCES test_decode_graph.cpp)
ginfer_add_test(ginfer_tensor_test       SOURCES test_tensor.cpp)
ginfer_add_test(ginfer_arena_test        SOURCES test_arena.cpp)
ginfer_add_test(ginfer_request_memory_test SOURCES test_request_memory.cpp)
ginfer_add_test(ginfer_admission_policy_test SOURCES test_admission_policy.cpp)
ginfer_add_test(ginfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
ginfer_add_test(ginfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ginfer_engine)
target_include_directories(ginfer_sampling_defaults_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export
  ${PROJECT_SOURCE_DIR}/src/targets/muse_glimmer_30b/export)
ginfer_add_test(ginfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ginfer_core)
ginfer_add_test(ginfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ginfer_core)
ginfer_add_test(ginfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ginfer_core)
set_tests_properties(
  ginfer_artifact_materialization_test
  ginfer_device_test
  ginfer_decode_graph_test
  ginfer_arena_test
  ginfer_request_memory_test
  ginfer_kv_cache_test
  ginfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
ginfer_add_test(ginfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ginfer_engine ginfer_core)
target_include_directories(ginfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ginfer_add_test(ginfer_muse_glimmer_30b_frontend_test
  SOURCES targets/muse_glimmer_30b/test_frontend.cpp
  LIBRARIES ginfer_engine ginfer_core)
target_include_directories(ginfer_muse_glimmer_30b_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/muse_glimmer_30b/export
  ${PROJECT_SOURCE_DIR}/src/targets/muse_glimmer_30b/impl)
ginfer_add_test(ginfer_qwen3_6_runtime_mechanisms_test
  SOURCES targets/qwen3_6/test_runtime_mechanisms.cpp
  LIBRARIES ginfer_engine ginfer_core)
target_include_directories(ginfer_qwen3_6_runtime_mechanisms_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ginfer_add_test(ginfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ginfer_engine)
set_tests_properties(ginfer_qwen3_6_27b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_qwen3_6_27b_load_plan_test
  SOURCES targets/qwen3_6_27b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ginfer_engine)
target_include_directories(ginfer_qwen3_6_27b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ginfer_qwen3_6_27b_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ginfer_qwen3_6_27b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_qwen3_8_27b_dflash2_real_test
  SOURCES targets/qwen3_8_27b/test_engine_dflash2_real.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ginfer_engine)
set_tests_properties(ginfer_qwen3_8_27b_dflash2_real_test PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_qwen3_8_27b_dflash2_load_plan_test
  SOURCES targets/qwen3_8_27b/test_dflash2_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ginfer_engine)
target_include_directories(ginfer_qwen3_8_27b_dflash2_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ginfer_qwen3_8_27b_dflash2_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ginfer_qwen3_8_27b_dflash2_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_qwen3_6_35b_a3b_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_real.cpp
  LIBRARIES ginfer_engine)
set_tests_properties(ginfer_qwen3_6_35b_a3b_real_test PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_qwen3_6_35b_a3b_dflash_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp
  LIBRARIES ginfer_engine)
set_tests_properties(ginfer_qwen3_6_35b_a3b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_qwen3_6_35b_a3b_dflash_load_plan_test
  SOURCES targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ginfer_engine)
target_include_directories(ginfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_link_libraries(ginfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ginfer_qwen3_6_35b_a3b_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_muse_glimmer_30b_load_plan_test
  SOURCES targets/muse_glimmer_30b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ginfer_engine)
target_include_directories(ginfer_muse_glimmer_30b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/muse_glimmer_30b/export)
target_link_libraries(ginfer_muse_glimmer_30b_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ginfer_muse_glimmer_30b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ginfer_add_test(ginfer_muse_glimmer_30b_real_test
  SOURCES targets/muse_glimmer_30b/test_engine_real.cpp
  LIBRARIES ginfer_engine)
set_tests_properties(ginfer_muse_glimmer_30b_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Product protocol contracts.
ginfer_add_test(ginfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ginfer_serve)
ginfer_add_test(ginfer_responses_schema_test
  SOURCES test_responses_schema.cpp
  LIBRARIES ginfer_serve)
ginfer_add_test(ginfer_response_store_test
  SOURCES test_response_store.cpp
  LIBRARIES ginfer_serve)
ginfer_add_test(ginfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ginfer_serve)
ginfer_add_test(ginfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES ginfer_serve)
ginfer_add_test(ginfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES ginfer_serve)
ginfer_add_test(ginfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES ginfer_serve)
ginfer_add_test(ginfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES ginfer_serve)
target_include_directories(ginfer_http_error_handler_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ginfer_add_test(ginfer_bench_support_test
  SOURCES test_ginfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b/ginfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ginfer_engine)
target_include_directories(ginfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b)

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(ginfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  grouped_dynamic_conv
  rope
  vision_pos_embed
  vision_attention
  sampling
  scalar
  cast
  scatter
  position)
foreach(op IN LISTS ginfer_op_tests)
  ginfer_add_op_test(ginfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES ginfer_ops)
endforeach()
ginfer_add_test(ginfer_gdn_gating_proj_plan_test
  SOURCES ops/test_gdn_gating_proj_plan.cpp
  LIBRARIES ginfer_ops)

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
ginfer_add_op_test(ginfer_gqa_attention_test
  SOURCES ops/test_gqa_attention.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_gqa_attention_muse_test
  SOURCES ops/test_gqa_attention_muse.cpp
  LIBRARIES ginfer_ops)
set_tests_properties(ginfer_gqa_attention_muse_test PROPERTIES TIMEOUT 30)
ginfer_add_op_test(ginfer_dflash2_select_test
  SOURCES ops/test_dflash2_select.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_bidirectional_gqa_attention_test
  SOURCES ops/test_bidirectional_gqa_attention.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_swa_test
  SOURCES ops/test_swa.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_kv_cache_append_prefix_test
  SOURCES ops/test_kv_cache_append_prefix.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES ginfer_ops)
ginfer_add_linear_test(ginfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
ginfer_add_linear_test(ginfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
ginfer_add_linear_test(ginfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
ginfer_add_linear_test(ginfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
ginfer_add_linear_test(ginfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
ginfer_add_linear_test(ginfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
ginfer_add_linear_test(ginfer_linear_fp8_a16_test ops/linear/test_fp8_a16.cpp)
ginfer_add_linear_test(ginfer_linear_fp8_a8_test ops/linear/test_fp8_a8.cpp)
ginfer_add_op_test(ginfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES ginfer_ops)
ginfer_add_fused_linear_test(
  ginfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ginfer_add_fused_linear_test(
  ginfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ginfer_add_fused_linear_test(
  ginfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ginfer_add_op_test(ginfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES ginfer_ops)
ginfer_add_op_test(ginfer_linear_add_fp8_test
  SOURCES ops/linear_add/test_fp8.cpp
  LIBRARIES ginfer_ops)
ginfer_add_fused_linear_test(
  ginfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
ginfer_add_fused_linear_test(
  ginfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ginfer_add_fused_linear_test(
  ginfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ginfer_add_fused_linear_test(
  ginfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ginfer_add_fused_linear_test(
  ginfer_linear_swiglu_fp8_test
  ops/linear_swiglu/test_fp8.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

ginfer_add_test(ginfer_qwen3_6_27b_visual_scatter_test
  SOURCES targets/qwen3_6_27b/test_visual_scatter.cpp
  LIBRARIES ginfer_engine ginfer_core)
set_tests_properties(ginfer_qwen3_6_27b_visual_scatter_test PROPERTIES SKIP_RETURN_CODE 77)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ginfer_qwen3_6_27b_internal_tests
  ginfer_qwen3_6_27b_visual_scatter_test)
foreach(test_target IN LISTS ginfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()
