# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(LaneKeepingInference LANGUAGES CXX)

# Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- ONNX Runtime Configuration ---
# Set the path to the ONNX Runtime installation
# IMPORTANT: Adjust this path to match your extracted folder name and exact location
# Assuming 'third_party' is in the project root and CMakeLists.txt is in src/cpp/
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime-osx-arm64-1.17.1")

# Add ONNX Runtime include directories
include_directories(
    "${ONNXRUNTIME_DIR}/include"
)

# Add ONNX Runtime library directories
link_directories(
    "${ONNXRUNTIME_DIR}/lib"
)

# --- End ONNX Runtime Configuration ---


# Add a subdirectory for source files (e.g., src/cpp/inference_test.cpp)
add_executable(lane_keeping_inference inference_test.cpp)

# Link the ONNX Runtime library to our executable
# The library name is typically 'onnxruntime' (CMake will add 'lib' and '.dylib/.so/.lib')
target_link_libraries(lane_keeping_inference PRIVATE onnxruntime)


# Specify output directories for binaries(optional)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
