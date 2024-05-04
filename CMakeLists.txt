set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
cmake_minimum_required(VERSION 3.27)
project(Ising)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native") 

include(CheckIPOSupported)
check_ipo_supported(RESULT result)


option(BUILD_TESTS "Build C++ testing/debugging binaries" OFF)

include_directories(include/)
include_directories(/opt/homebrew/include/eigen3)

set(Python_EXECUTABLE "/Users/mariusmahiout/miniforge3/envs/ising_env/bin/python")
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)
find_package(Boost 1.65 REQUIRED) 
include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(extern/pybind11)

pybind11_add_module(ising MODULE
        src/pybindings.cpp
        src/models.cpp
        src/sample.cpp
        src/grad_ascent.cpp
        src/utils.cpp
)

if(result)
  set_target_properties(ising PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(BUILD_TESTS)
    add_executable(ising_debug
        testing/ising_debug.cpp
        src/models.cpp
        src/sample.cpp
        src/grad_ascent.cpp
        src/utils.cpp
    )

    set_target_properties(ising_debug PROPERTIES
        COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
        INTERPROCEDURAL_OPTIMIZATION FALSE
    )
endif()