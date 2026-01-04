cmake_minimum_required(VERSION 3.16)

project(
  axon
  VERSION 0.0.1
  DESCRIPTION "Lightweight deep learning framework from first principles"
  HOMEPAGE_URL "https://github.com/abidanBrito/axon"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Static library
add_library(axon_core
  src/neuron.cpp
  src/network.cpp
)

target_include_directories(axon_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Tests
enable_testing()
add_subdirectory(tests)

# Main executable
add_subdirectory(src)
