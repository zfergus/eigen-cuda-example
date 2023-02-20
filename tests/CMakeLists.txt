################################################################################
# Tests
################################################################################

add_executable(eigen_cuda_example_tests
  test.cpp
  benchmark.cpp
)

target_include_directories(eigen_cuda_example_tests PUBLIC ".")

################################################################################
# Required Libraries
################################################################################

target_link_libraries(eigen_cuda_example_tests PUBLIC eigen_cuda_example)

include(eigen_cuda_example_warnings)
target_link_libraries(eigen_cuda_example_tests PRIVATE eigen_cuda_example_warnings)

find_package(CUDAToolkit)
target_link_libraries(eigen_cuda_example_tests PRIVATE CUDA::cudart)

include(catch2)
target_link_libraries(eigen_cuda_example_tests PUBLIC Catch2::Catch2WithMain)

################################################################################
# Compiler options
################################################################################

target_compile_definitions(eigen_cuda_example_tests PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)

# Test data directory definition
if(WIN32)
  target_compile_definitions(eigen_cuda_example_tests PUBLIC TEST_DATA_DIR_CSTR="${CMAKE_CURRENT_SOURCE_DIR}\\\\data\\\\")
else()
  target_compile_definitions(eigen_cuda_example_tests PUBLIC TEST_DATA_DIR_CSTR="${CMAKE_CURRENT_SOURCE_DIR}/data/")
endif()

################################################################################
# Register tests
################################################################################

FetchContent_GetProperties(catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(eigen_cuda_example_tests)
