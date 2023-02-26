# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(EIGEN_CUDA_EXAMPLE_TOPLEVEL_PROJECT OFF)
else()
    set(EIGEN_CUDA_EXAMPLE_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14.0")
if(EIGEN_CUDA_EXAMPLE_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
    # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
    if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
        message(FATAL_ERROR "CMake required version to build IPC Toolkit is ${REQUIRED_CMAKE_VERSION}")
    endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/IPCToolkitOptions.cmake)
    message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/IPCToolkitOptions.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/IPCToolkitOptions.cmake)
endif()

################################################################################

project(IPCToolkit
        DESCRIPTION "A set of reusable functions to integrate IPC into an existing simulation."
        LANGUAGES CXX
        VERSION "0.0.1")

option(EIGEN_CUDA_EXAMPLE_BUILD_TESTS "Build unit-tests" ${EIGEN_CUDA_EXAMPLE_TOPLEVEL_PROJECT})

# Set default minimum C++ standard
if(EIGEN_CUDA_EXAMPLE_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

### Configuration
set(EIGEN_CUDA_EXAMPLE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src")
set(EIGEN_CUDA_EXAMPLE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/eigen_cuda_example/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")

################################################################################
# IPC Toolkit Library
################################################################################

# Add an empty library and fill in the list of sources in `src/CMakeLists.txt`.
add_library(eigen_cuda_example
  "${EIGEN_CUDA_EXAMPLE_SOURCE_DIR}/example.cu")

# Fill in configuration options
configure_file(
  "${EIGEN_CUDA_EXAMPLE_SOURCE_DIR}/config.hpp.in"
  "${EIGEN_CUDA_EXAMPLE_SOURCE_DIR}/config.hpp")

# Public include directory for IPC Toolkit
target_include_directories(eigen_cuda_example PUBLIC "${EIGEN_CUDA_EXAMPLE_INCLUDE_DIR}")

# CMake utils
include(eigen_cuda_example_target_link_system_libraries)

################################################################################
# Optional Definitions
################################################################################

# For MSVC, do not use the min and max macros.
target_compile_definitions(eigen_cuda_example PUBLIC NOMINMAX)

################################################################################
# Dependencies
################################################################################

# Eigen
include(eigen)
eigen_cuda_example_target_link_system_libraries(eigen_cuda_example PUBLIC Eigen3::Eigen)
# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(eigen_cuda_example PUBLIC "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

# spdlog
include(spdlog)
eigen_cuda_example_target_link_system_libraries(eigen_cuda_example PUBLIC spdlog::spdlog)

# Extra warnings (link last for highest priority)
include(eigen_cuda_example_warnings)
# target_link_libraries(eigen_cuda_example PRIVATE eigen_cuda_example_warnings)

################################################################################
# Compiler options
################################################################################

# Use C++17
target_compile_features(eigen_cuda_example PUBLIC cxx_std_17)

################################################################################
# CUDA
################################################################################

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "No CUDA support found!")
endif()

# We need to explicitly state that we need all CUDA files in the particle library
# to be built with -dc as the member functions could be called by other libraries
# and executables.
set_target_properties(eigen_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
  set(CMAKE_CUDA_ARCHITECTURES "native")
  set_target_properties(eigen_cuda_example PROPERTIES CUDA_ARCHITECTURES "native")
else()
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
  string(STRIP "${CUDA_ARCH_LIST}" CUDA_ARCH_LIST)
  string(REPLACE " " ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  set_target_properties(eigen_cuda_example PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif()

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static CUDA runtime can find it at runtime.
  set_property(TARGET eigen_cuda_example
                PROPERTY
                BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

find_package(CUDAToolkit)
target_link_libraries(eigen_cuda_example PRIVATE CUDA::cudart)

################################################################################
# Tests
################################################################################

if(EIGEN_CUDA_EXAMPLE_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()