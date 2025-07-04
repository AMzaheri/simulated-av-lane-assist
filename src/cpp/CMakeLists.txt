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
else()
    message(FATAL_ERROR "OpenCV not found! Please ensure it's installed and findable by CMake.")
endif()
# --- End OpenCV Configuration ---

# --- ONNX Runtime Configuration ---
# Set the base path to the ONNX Runtime binaries/headers folder (third_party/)
# This is the directory that *contains* either the extracted content directly,
# or a version-specific subfolder like onnxruntime-osx-arm64-1.17.1
set(ONNXRUNTIME_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")

# Define the typical version-specific subfolder name for local macOS downloads
# Adjust this name if your local ONNX Runtime folder is different!
set(ONNXRUNTIME_VERSION_SUBDIR "onnxruntime-osx-arm64-1.17.1")

# Determine the actual ONNX Runtime installation path
# Check if the version-specific subfolder exists
if (EXISTS "${ONNXRUNTIME_BASE_DIR}/${ONNXRUNTIME_VERSION_SUBDIR}/include/onnxruntime_cxx_api.h")
    # If the subfolder exists (likely local macOS setup)
    set(ONNXRUNTIME_INSTALL_PATH "${ONNXRUNTIME_BASE_DIR}/${ONNXRUNTIME_VERSION_SUBDIR}")
    message(STATUS "Found ONNX Runtime in version-specific subfolder: ${ONNXRUNTIME_VERSION_SUBDIR}")
else()
    # Otherwise, assume it's directly in the base directory (like in CI due to --strip-components=1)
    set(ONNXRUNTIME_INSTALL_PATH "${ONNXRUNTIME_BASE_DIR}")
    message(STATUS "Found ONNX Runtime directly in third_party folder (likely CI setup).")
endif()

# Find ONNX Runtime library
# This step needs to happen early to define ONNXRUNTIME_LIB
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime # CMake will look for libonnxruntime.so (Linux) or libonnxruntime.dylib (macOS)
    PATHS "${ONNXRUNTIME_INSTALL_PATH}/lib" # Look in the 'lib' sub-directory of the determined install path
    NO_DEFAULT_PATH # Only search the paths we explicitly provide
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_INSTALL_PATH}/lib")
else()
    message(STATUS "ONNX Runtime library found at: ${ONNXRUNTIME_LIB}")
endif()

# --- End ONNX Runtime Configuration ---


# Add executables for source files (e.g., src/cpp/inference_test.cpp)
add_executable(lane_keeping_test_inference inference_test.cpp)
add_executable(lane_keeping_real_inference inference_real.cpp)


# --- Configure and Link Targets (MUST BE AFTER add_executable) ---

# For lane_keeping_test_inference
target_include_directories(lane_keeping_test_inference PRIVATE
    "${ONNXRUNTIME_INSTALL_PATH}/include" # Use the determined install path
)
target_link_libraries(lane_keeping_test_inference PRIVATE
    ${ONNXRUNTIME_LIB} # Link using the found library path
)

# For lane_keeping_real_inference
target_include_directories(lane_keeping_real_inference PRIVATE
    "${ONNXRUNTIME_INSTALL_PATH}/include" # Use the determined install path
)
target_link_libraries(lane_keeping_real_inference PRIVATE
    ${ONNXRUNTIME_LIB} # Link using the found library path
    ${OpenCV_LIBS} # OpenCV is required for real image processing
)


# Specify output directories for binaries (optional)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
