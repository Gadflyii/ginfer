target_sources(ginfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tokenizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/sequence.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/program.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/logit_softcap.cu)

target_include_directories(ginfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
