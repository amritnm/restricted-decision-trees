cmake_minimum_required(VERSION 3.20)
project(restricted_decision_trees LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rdt_core INTERFACE)
target_include_directories(rdt_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
target_compile_features(rdt_core INTERFACE cxx_std_17)

# GCC defaults to -ffp-contract=fast: wherever the target has FMA (every
# aarch64 build, for one) it fuses a multiply and an add into one FMA -- one
# rounding instead of two -- even across statements. The exact and hist split
# searches then round the same sums differently, so hist stops equalling exact
# bit for bit on few-valued data (test_hist_honours_the_level_subsets failed
# on aarch64 Linux), and results drift from the x86_64 builds, which have no
# FMA at GCC's default -march. Clang (whose default contracts only within one
# expression) and MSVC are left alone.
target_compile_options(rdt_core INTERFACE $<$<CXX_COMPILER_ID:GNU>:-ffp-contract=off>)

# thread_pool.hpp uses std::thread; on Linux/macOS (GCC/Clang) that needs an
# explicit pthread link, unlike MSVC where it's implicit -- without this, the
# core builds fine (header-only) but linking anything against it fails on
# Unix with undefined references to pthread symbols.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(rdt_core INTERFACE Threads::Threads)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_rdt_native src/bindings.cpp)
target_link_libraries(_rdt_native PRIVATE rdt_core)

install(TARGETS _rdt_native LIBRARY DESTINATION rdt)
