add_executable(ginfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ginfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ginfer PRIVATE
  ginfer_engine
  ginfer_product_load_progress
  ginfer_product_prompt_input)

add_executable(ginfer-serve serve/main.cpp)
target_include_directories(ginfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ginfer-serve PRIVATE ginfer_serve ginfer_product_load_progress)
