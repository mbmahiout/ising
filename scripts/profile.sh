#!/bin/bash

# define build directory and ensure it exists
BUILD_DIR=./build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ensure BUILD_TESTS is enabled and build the project with CMake
echo "Building the project..."
cmake -DCMAKE_BUILD_TYPE=Profile -DBUILD_TESTS=ON ..
cmake --build .

# name of executable
EXECUTABLE=./ising_debug

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE does not exist."
    exit 1
fi

# run profiling
echo "Running profiling..."
CPUPROFILE=profile.prof DYLD_INSERT_LIBRARIES=/opt/homebrew/lib/libprofiler.dylib $EXECUTABLE

# generate profiling report
echo "Generating profiling report..."
pprof --text $EXECUTABLE profile.prof > profile_report.txt
pprof --pdf $EXECUTABLE profile.prof > profile_report.pdf

echo "Profiling complete. Text report saved to profile_report.txt and PDF report to profile_report.pdf."

# open pdf report automatically
open profile_report.txt
