# Minimum CMake version required
cmake_minimum_required(VERSION 3.16)

# Project name and version
project(SME2_BLAS VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler to system clang if not already set
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -mcpu=apple-m4 -march=native+sme2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(include/sme2_gemm)
include_directories(include/debug)

# Create a library target for your headers (header-only library)
add_library(sme2_gemm INTERFACE)
target_include_directories(sme2_gemm INTERFACE include/sme2_gemm)

add_library(gemm_truth INTERFACE)
target_include_directories(gemm_truth INTERFACE include/debug)

# If you later add source files, uncomment and modify this:
# file(GLOB_RECURSE SOURCES "src/*.cpp")
# add_library(sme2_gemm ${SOURCES})
# target_include_directories(sme2_gemm PUBLIC src/include)

# Test executable
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")
add_executable(test_program ${TEST_SOURCES})

# Benchmark executable
file(GLOB_RECURSE BENCHMARK_SOURCES "benchmark/*.cpp")
add_executable(benchmark_program ${BENCHMARK_SOURCES})

target_link_options(test_program PRIVATE "LINKER:-v")

target_link_options(benchmark_program PRIVATE "LINKER:-v")

# Link the library to the test executable
target_link_libraries(test_program sme2_gemm gemm_truth)

find_package(benchmark REQUIRED)

# Link the library to the test executable
target_link_libraries(benchmark_program sme2_gemm benchmark::benchmark)

# Optional: Add custom target for running tests
add_custom_target(run_tests
    COMMAND test_program
    DEPENDS test_program
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests..."
)
