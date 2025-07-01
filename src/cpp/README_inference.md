# C++ Inference Module for Lane Keeping Model

This module provides a C++ implementation for performing inference using the NVIDIA PilotNet model, leveraging ONNX Runtime for efficient model execution and OpenCV for image preprocessing. It supports both dummy input testing and real-time inference on actual image data.

## Table of Contents

* [Dependencies](#dependencies)
* [Setup Instructions](#setup-instructions)
    * [1. Download ONNX Runtime](#1-download-onnx-runtime)
    * [2. Install OpenCV](#2-install-opencv)
    * [3. Place the ONNX Model](#3-place-the-onnx-model)
    * [4. Prepare Test Images (for Real Data Inference)](#4-prepare-test-images-for-real-data-inference)
* [Build and Run](#build-and-run)
    * [Build Instructions](#build-instructions)
    * [Running Dummy Input Inference](#running-dummy-input-inference)
    * [Running Real Image Inference](#running-real-image-inference)

---

## Dependencies

* **CMake:** Version 3.10 or higher.
* **C++ Compiler:** A modern C++ compiler (e.g., AppleClang on macOS, GCC, MSVC) supporting C++17 standard).
* **ONNX Runtime:** Specifically, **v1.17.1 for macOS ARM64**. Other versions or platforms may require adjustments. This is the ONNX inference engine.
* **OpenCV:** A popular computer vision library, essential for image loading, manipulation, and preprocessing (required for Real Image Data Inference).

## Setup Instructions

These steps will guide you through setting up the necessary components to build and run the C++ inference module.

### 1. Download ONNX Runtime

Download the pre-built ONNX Runtime package for macOS ARM64 v1.17.1 (or the version appropriate for your system) from the [ONNX Runtime GitHub Releases page](https://github.com/microsoft/onnxruntime/releases).

Extract the downloaded `.zip` file. Place the extracted folder (e.g., `onnxruntime-osx-arm64-1.17.1`) into a `third_party/` directory at the **root of the `simulated-av-lane-assist` project**.

```bash
mkdir -p third_party
unzip /path/to/downloaded/onnxruntime-osx-arm64-1.17.1.zip -d third_party/
```

Ensure `third_party/` is listed in your project's `.gitignore` file.

### 2. Install OpenCV

```bash
# Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"

# Install OpenCV via Homebrew
brew install opencv
```

For other operating systems, please refer to the [official OpenCV installation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) guide or relevant package managers.

### 3. Place the ONNX Model

Ensure your ONNX model file (e.g., `nvidia_pilotnet.onnx`) is placed inside the `models/` directory at the root of `simulated-av-lane-assist` project.

### 4. Prepare Test Images (for Real Data Inference)

 * The `data/test_images/` directory contains two sample `.png` images, used for demonstrating the real image inference capabilities.
 * Sample images, simulating straight and curved road conditions, can be created using the `src/python/data_generator.py` script.
 
## Build and Run

### Build Instructions

1.  **Navigate to the `build` directory:**
    ```bash
    cd /path/to/your/simulated-av-lane-assist/src/cpp/build
    ```
    (If `build` directory doesn't exist, create it: `mkdir build`)

2.  **Run CMake to configure the project:**
    ```bash
    cmake ..
    ```
    Ensure you see messages confirming ONNX Runtime and OpenCV were found.

3.  **Build the executables:**
    ```bash
    make
    ```
    This will compile `inference_test.cpp` and `inference_real.cpp`, creating `lane_keeping_test_inference` and `lane_keeping_real_inference` executables (which will be placed in the `bin/` directory at your project root, as configured in `CMakeLists.txt`).

### Running Dummy Input Inference

This executable (`lane_keeping_test_inference`) uses a dummy, synthetic tensor as input to verify the ONNX Runtime setup and core model loading without image processing.

```bash
/path/to/your/simulated-av-lane-assist/bin/lane_keeping_test_inference
```

### Running Real Image Inference

 
This executable (`lane_keeping_real_inference`) loads and preprocesses `.png` images from the `data/test_images/` directory and performs inference on them.

```bash
/path/to/your/simulated-av-lane-assist/bin/lane_keeping_real_inference
```

You should see output for each image processed, including its filename and the predicted steering angle. Different angles are expected for images representing straight vs. curved roads.
