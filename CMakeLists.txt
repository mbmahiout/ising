cmake_minimum_required(VERSION 3.27)
project(Ising)

set(CMAKE_CXX_STANDARD 20)

option(BUILD_TESTS "Build C++ testing/debugging binaries" OFF)

include_directories(include/)
include_directories(/opt/homebrew/include/eigen3)

set(Python_EXECUTABLE "/Users/mariusmahiout/miniforge3/envs/ising_env/bin/python")
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(extern/pybind11)

pybind11_add_module(ising MODULE
        src/pybindings.cpp
        src/models.cpp
        src/sample.cpp
        src/exact_infer.cpp
        src/utils.cpp
)

if(BUILD_TESTS)
    add_executable(ising_debug
        testing/ising_debug.cpp
        src/models.cpp
        src/sample.cpp
        src/exact_infer.cpp
        src/utils.cpp
    )
endif()