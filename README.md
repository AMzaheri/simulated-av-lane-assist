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
│     ├── cpp/
│     └── python/
├── third_party/        <-- Contains external dependencies like ONNX Runtime
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


## C++ Inference Module

For high-performance model deployment, a C++ inference module has been developed. This module leverages **ONNX Runtime (v1.17.1)** to load and execute pre-trained ONNX models, such as the PilotNet for lane keeping.

Fo setup, build, and run instructions for the C++ inference application, please refer to its detailed documentation:
[**`src/cpp/README_inference.md`**](src/cpp/README_inference.md)

### Real Image Inference

The C++ module includes functionalities for:
* Loading images from file.
* Preprocessing steps such as resizing, grayscale conversion, and normalisation, tailored to the model's input requirements.
* Feeding the processed image data directly into the ONNX model for inference.

This allows for direct application of the trained models on actual visual inputs. **This enables a robust and performant pathway for integrating the AI model into real-time or embedded systems.**

## Continuous Integration / Continuous Deployment (CI/CD)

This project uses GitHub Actions to implement Continuous Integration (CI) for its C++ components, code quality and build reliability.

### Key Aspects:
* **Automated Builds:** Every push to key branches (like `main` , `dev` and `feature/cpp-inference`) automatically triggers a build process. This compiles the C++ inference executables (`lane_keeping_test_inference` and `lane_keeping_real_inference`) on an Ubuntu Linux environment.
* **Dependency Management:** The CI pipeline handles the automatic installation of important dependencies such as CMake, build-essential tools, OpenCV, and the ONNX Runtime for Linux (x64).
* **Automated Testing/Verification:** After a successful build, the pipeline attempts to run the `lane_keeping_test_inference` executable, providing immediate feedback on its runnability and basic functionality.
* **Cross-Platform Verification:** While development might occur on various operating systems (e.g., macOS), the CI environment consistently builds and tests on Linux, mimicking common deployment environments.

### Build Status:
You can view the current CI/CD pipeline status and detailed build logs by clicking on the badge below or visiting the [Actions tab](https://github.com/AMzaheri/simulated-av-lane-assist/actions) of this repository.

[![CI Build Status](https://github.com/AMzaheri/simulated-av-lane-assist/workflows/C%2B%2B%20Build%20and%20Test/badge.svg)](https://github.com/AMzaheri/simulated-av-lane-assist/actions)

---
