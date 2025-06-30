# simulated-av-lane-assist
A simulated autonomous driving project implementing a lane-keeping assistant using a hybrid Python-C++ architecture with MLOps principles.

## Project Structure

```
├── .gitignore
├── README.md
├── app/
├── data/
├── docs/
├── models/
└── src/
    ├── cpp/
    └── python/
```

## Dataset

This project utilises a custom-generated synthetic driving dataset, specifically designed for training machine learning models for autonomous vehicle lane-keeping.

The dataset comprises **34,379 grayscale images** captured from a simulated front-facing camera, covering driving scenarios on both **straight road segments** and a **curved (left-turn) arc**. Each image is paired with a corresponding steering angle label.

**Access the complete dataset on Kaggle:**
[Link to the Kaggle Dataset Here] 
(`https://www.kaggle.com/datasets/afsanehm/simulated-driving-data`)

For a detailed explanation of the data generation process, including the specific parameters used for diversification and the dataset's internal structure, please refer to the dedicated [`Data_README.md`](data/README.md).

### ML Model Training & Accessibility

The deep learning model was trained and evaluated on a dedicated Kaggle notebook: [Deep Learning for Simulated Driving](https://www.kaggle.com/code/afsanehm/deep-learning-for-simulated-driving). The trained model checkpoints (including the best-performing version) are accessible within this repository in the `models/` directory.


## Lane Keeping Inference (C++ with ONNX Runtime)

This section demonstrates how to perform neural network inference in a C++ console application using the ONNX Runtime library. It's configured to load and run an [NVIDIA PilotNet model] (https://www.kaggle.com/code/afsanehm/deep-learning-for-simulated-driving) for a lane-keeping task, taking dummy image data as input and predicting a steering angle.
- This setup has been successfully tested on macOS (ARM64 architecture) using ONNX Runtime v1.17.1.

### Project dtructur for this section

```
simulated-av-lane-assist/
├── models/
│   └── nvidia_pilotnet.onnx  <-- ONNX model
├── src/
│   └── cpp/
│       ├── CMakeLists.txt
│       └── inference_test.cpp
│       └── build/                 <-- CMake build directory 
│           └── lane_keeping_inference  <-- Executable will be here after build
├── third_party/
│   └── onnxruntime-osx-arm64-1.17.1/  <-- Extracted ONNX Runtime package goes here
│       ├── include/
│       └── lib/
└── README.md

```

## Dependencies

* **CMake:** Version 3.10 or higher.
* **C++ Compiler:** A modern C++ compiler (e.g., AppleClang on macOS, GCC, MSVC) supporting C++17 standard.
* **ONNX Runtime:** Specifically, **v1.17.1 for macOS ARM64**. Other versions or platforms may require adjustments.

## Setup Instructions

1.  **Download ONNX Runtime:**
    Download the pre-built ONNX Runtime package for macOS ARM64 v1.17.1 (or the version appropriate for your system). You can typically find this on the [ONNX Runtime GitHub Releases page](https://github.com/microsoft/onnxruntime/releases).

2.  **Extract ONNX Runtime:**
    Extract the downloaded `onnxruntime-osx-arm64-1.17.1.zip` file. Place the extracted folder (e.g., `onnxruntime-osx-arm64-1.17.1`) into a `third_party/` directory at the root directory of the `simulated-av-lane-assist` project.

    ```
    mkdir third_party
    unzip /path/to/downloaded/onnxruntime-osx-arm64-1.17.1.zip -d third_party/
    ```

3.  **Place the ONNX Model:**
    Ensure your ONNX model file (e.g., `nvidia_pilotnet.onnx`) is placed inside the `models/` directory at the root of your `simulated-av-lane-assist` project.

    ```
    mkdir models
    mv /path/to/your/nvidia_pilotnet.onnx models/
    ```

## Build Instructions

1.  **Create a build directory:**
    Navigate to the `src/cpp/` directory and create a `build` subdirectory if it doesn't already exist.

    ```bash
    cd simulated-av-lane-assist/src/cpp/
    mkdir -p build
    cd build
    ```

2.  **Configure CMake:**
    From inside the `build` directory, run CMake to configure the project.

    ```bash
    cmake ..
    ```
    * This step detects the compiler and generates the build system (e.g., Makefiles). You should see output indicating successful configuration.

3.  **Build the project:**
    From inside the `build` directory, trigger the compilation and linking process.

    ```bash
    make
    ```
    * You should see progress indicators (`[ 50%] Building CXX object...`, `[100%] Linking CXX executable...`) and finally `[100%] Built target lane_keeping_inference`.

## Running the Application

After a successful build, the executable `lane_keeping_inference` will be located in the `simulated-av-lane-assist/src/cpp/build/` directory.

1.  **Run the executable:**
    From inside the `build` directory:

    ```bash
    ./lane_keeping_inference
    ```

