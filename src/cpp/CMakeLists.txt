# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(LaneKeepingInference LANGUAGES CXX)

# Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- OpenCV Configuration ---
# Find OpenCV package installed on the system (e.g., via Homebrew)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")
    # Include OpenCV headers so the source files can find them
    include_directories(${OpenCV_INCLUDE_DIRS})
    # For some complex setups, we might need link_directories, but usually find_package is enough.
    # link_directories(${OpenCV_LIB_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found! Please ensure it's installed and findable by CMake.")
endif()
# --- End OpenCV Configuration ---

# --- ONNX Runtime Configuration ---
# Set the path to the ONNX Runtime installation
# IMPORTANT: Adjust this path to match our extracted folder name and exact location
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


# Add executables for source files (e.g., src/cpp/inference_test.cpp)
add_executable(lane_keeping_test_inference inference_test.cpp)
add_executable(lane_keeping_real_inference inference_real.cpp)

#Link the ONNX Runtime libraries to the TEST executable
target_link_libraries(lane_keeping_test_inference PRIVATE
    onnxruntime
)

# Link the ONNX Runtime and OpenCV libraries to the REAL IMAGE INFERENCE executable
target_link_libraries(lane_keeping_real_inference PRIVATE
    onnxruntime
    ${OpenCV_LIBS} # OpenCV is required for real image processing
)


# Specify output directories for binaries(optional)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
