cmake_minimum_required(VERSION 3.16)

project(
  axon
  VERSION 0.0.1
  DESCRIPTION "Lightweight deep learning framework from first principles"
  HOMEPAGE_URL "https://github.com/abidanBrito/axon"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AXON_BUILD_EXAMPLES "Build example programs" ON)
option(AXON_BUILD_TESTS "Build tests" ON)

# Core library
add_library(axon_core
  src/neuron.cpp
  src/network.cpp
)

add_library(axon::core ALIAS axon_core)

target_include_directories(axon_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(axon_core PUBLIC cxx_std_17)

# Examples
if(AXON_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Tests
if(AXON_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_subdirectory(tests)
endif()
