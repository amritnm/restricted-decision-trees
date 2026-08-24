cmake_minimum_required(VERSION 3.20)
project(restricted_decision_trees LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rdt_core INTERFACE)
target_include_directories(rdt_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
target_compile_features(rdt_core INTERFACE cxx_std_17)

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
