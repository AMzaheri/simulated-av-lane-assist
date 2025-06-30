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

For high-performance model deployment, a dedicated C++ inference module has been developed. This module leverages **ONNX Runtime (v1.17.1)** to efficiently load and execute pre-trained ONNX models, such as the PilotNet for lane keeping.

For comprehensive setup, build, and run instructions for the C++ inference application, please refer to its detailed documentation:
[**`src/cpp/README_inference.md`**](src/cpp/README_inference.md)

### Real Image Inference

The C++ module includes functionalities for:
* Loading images from file.
* Preprocessing steps such as resizing, grayscale conversion, and normalisation, tailored to the model's input requirements.
* Feeding the processed image data directly into the ONNX model for inference.

This allows for direct application of the trained models on actual visual inputs, moving beyond dummy data for testing and into practical inference scenarios.
