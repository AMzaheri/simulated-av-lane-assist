#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <array>

// --- OpenCV Headers ---
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // For cv::imread
#include <opencv2/imgproc.hpp>   // For cv::cvtColor, cv::resize
// #include <opencv2/highgui.hpp> // Optional: For displaying images, to debug visual output
// --------------------------------

int main() {
    std::cout << "Hello from C++ inference_real!" << std::endl;
    std::cout << "(With ONNX Runtime v1.17.1)" << std::endl;

    // --- 1. Initialise ONNX Runtime Environment ---
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "LaneKeepingInference"
    );

    // --- 2. Create Session Options ---
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    // --- 3. Define Model Path ---
    const std::string model_path = "../../../models/nvidia_pilotnet.onnx";

    // --- 4. Create an Inference Session ---
    Ort::Session session(env, model_path.c_str(), session_options);

    // --- 5. Create Ort::MemoryInfo for Tensors (Allocator is handled differently now) ---
    // We still need memory_info for creating Ort::Value tensors.
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, // Or OrtDeviceAllocator for default CPU allocator
            OrtMemType::OrtMemTypeDefault
        );

    // --- 6. Get Input and Output Node Info ---
    // Input: "input", shape (batch_size, 1, 66, 200)
    // Output: "output", shape (batch_size, 1)

    // *** IMPORTANT CHANGE: Use Ort::AllocatorWithDefaultOptions ***
    Ort::AllocatorWithDefaultOptions ort_alloc;

    // Get input node details
    size_t num_input_nodes = session.GetInputCount();
    // Use empty vectors and push_back, as TypeInfo and AllocatedStringPtr are not default-constructible
    std::vector<Ort::AllocatedStringPtr> input_node_names;
    input_node_names.reserve(num_input_nodes); // Optional: pre-allocate memory for efficiency

    std::vector<std::vector<int64_t>> input_node_shapes(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        // Use the ort_alloc object directly as the second argument
        input_node_names.push_back(session.GetInputNameAllocated(i, ort_alloc));

        // GetInputTypeInfo returns Ort::TypeInfo. We get it here and immediately extract shape.
        // We are no longer storing a vector of Ort::TypeInfo objects, mirroring the ResNet example.
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        input_node_shapes[i] = type_info.GetTensorTypeAndShapeInfo().GetShape();
        // The type_info object will be properly destructed when it goes out of scope in each iteration.
    }
    std::cout << "Input Name: "
              << input_node_names[0].get() << std::endl; // Use .get() to access raw const char*
    std::cout << "Input Shape: ["
              << input_node_shapes[0][0] << ", "
              << input_node_shapes[0][1] << ", "
              << input_node_shapes[0][2] << ", "
              << input_node_shapes[0][3] << "]"
              << std::endl;

    // Get output node details
    size_t num_output_nodes = session.GetOutputCount();
    // Use empty vectors and push_back
    std::vector<Ort::AllocatedStringPtr> output_node_names;
    output_node_names.reserve(num_output_nodes); // Optional: pre-allocate memory for efficiency

    std::vector<std::vector<int64_t>> output_node_shapes(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) {
        // Use the ort_alloc object directly as the second argument
        output_node_names.push_back(session.GetOutputNameAllocated(i, ort_alloc));

        // GetOutputTypeInfo returns Ort::TypeInfo.
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        output_node_shapes[i] = type_info.GetTensorTypeAndShapeInfo().GetShape();
    }
    std::cout << "Output Name: "
              << output_node_names[0].get() << std::endl; // Use .get() to access raw const char*
    std::cout << "Output Shape: ["
              << output_node_shapes[0][0] << ", "
              << output_node_shapes[0][1] << "]"
              << std::endl;

    // --- 7. Load and Preprocess Image for Input Tensor ---

    // Define model input dimensions (ensure these match your model's expected input)
    const int64_t input_width = 200;
    const int64_t input_height = 66;
    const int64_t input_channels = 1; // Assuming grayscale model input
    const int64_t batch_size = 1;

    // IMPORTANT: Define the path to your test image.
    // Place a test image (e.g., test_image.jpg) in your 'simulated-av-lane-assist/models/' directory.
    // The path is relative to where your executable runs (from 'src/cpp/build/').
    const std::string image_path = "../../../data/data_real_inference/images/frame_00028.png";

    // Load the image using OpenCV. cv::IMREAD_COLOR loads it as a 3-channel BGR image.
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "ERROR: Could not load image from " << image_path << std::endl;
        return 1; // Exit the program if image not found
    }
    std::cout << "Image loaded successfully: " << image.cols << "x" << image.rows << std::endl;

    // Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    std::cout << "Converted to grayscale." << std::endl;

    // Resize image to model's input dimensions (66x200)
    cv::Mat resized_image;
    cv::resize(gray_image, resized_image, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
    std::cout << "Image resized to: " << resized_image.cols << "x" << resized_image.rows << std::endl;

    // Normalize pixel values to range [-1, 1].
    // This is common for PilotNet models, where 0-255 pixels are scaled.
    // If your model expects 0-1 (divide by 255.0), adjust the scaling.
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC1, 1.0 / 127.5, -1.0); // Scales 0-255 to [-1, 1]
    std::cout << "Image normalized to [-1, 1] range." << std::endl;

    // Ensure the data is contiguous in memory.
    // This is often needed when passing data from cv::Mat to other libraries.
    if (!float_image.isContinuous()) {
        float_image = float_image.clone();
    }

    // Prepare the input tensor vector.
    // The model expects a single batch, 1 channel (grayscale), 66 height, 200 width.
    const size_t input_tensor_size = batch_size * input_channels * input_height * input_width;
    std::vector<float> input_tensor_values(input_tensor_size);

    // Copy the processed image data from cv::Mat to the std::vector<float>.
    // Since float_image is CV_32FC1 (single channel float), its data is already flat and ready to copy.
    std::memcpy(input_tensor_values.data(), float_image.data, input_tensor_size * sizeof(float));
    std::cout << "Image data copied to input tensor." << std::endl;

    std::vector<int64_t> input_shape = {
        batch_size, input_channels, input_height, input_width
    };


    // --- 8. Create Input Tensor ---
    // Create ONNX Runtime input tensor from our preprocessed image data
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(), // Pointer to the image data
        input_tensor_size,          // Total number of elements
        input_shape.data(),         // Pointer to the shape array
        input_shape.size()          // Number of dimensions in the shape
    );

    // Double-check if the input tensor is valid
    assert(input_tensor.IsTensor() && input_tensor.GetTensorTypeAndShapeInfo().GetElementType() == 
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::cout << "Input tensor created." << std::endl;

    // --- 9. Define Input and Output Names ---
    // The C API expects const char* arrays for input/output names.
    // Use the names retrieved from the model (input_node_names, output_node_names).
    const char* input_names_char[] = {input_node_names[0].get()};
    const char* output_names_char[] = {output_node_names[0].get()};

    // --- 10. Run Inference ---
    std::cout << "Running inference..." << std::endl;
    // Ort::RunOptions run_options{nullptr}; // Can set options here if needed
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},      // Run options (can be nullptr for default)
        input_names_char,              // Array of input names
        &input_tensor,                 // Array of input Ort::Value pointers
        1,                             // Number of inputs (only one in this case)
        output_names_char,             // Array of output names
        1                              // Number of outputs (only one in this case)
    );
    std::cout << "Inference complete!" << std::endl;

    // --- 11. Process Output ---
    // Get the output tensor (assuming there's only one output)
    Ort::Value& output_tensor = output_tensors[0];

    // Get the raw pointer to the output data
    float* output_data = output_tensor.GetTensorMutableData<float>();

    // Get the shape of the output tensor
    Ort::TensorTypeAndShapeInfo output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> actual_output_shape = output_tensor_info.GetShape();

    // The PilotNet model typically outputs a single float value (steering angle).
    // The output shape should be (1, 1) or just (1) if it's flattened.
    if (actual_output_shape.empty()) { // For scalar output, shape might be empty or {1}
        std::cout << "Predicted steering angle (scalar): " << output_data[0] << std::endl;
    } else if (actual_output_shape.size() == 2 && actual_output_shape[0] == 1 && actual_output_shape[1] == 1) {
        std::cout << "Predicted steering angle (batch 1, element 1): " << output_data[0] << std::endl;
    } else {
        std::cout << "Unexpected output shape. Printing first element: " << output_data[0] << std::endl;
        // You might want to print more elements or iterate if your model output is different
        // For example, if it was (1, N) for N classes:
        // for (size_t i = 0; i < output_tensor_info.GetElementCount(); ++i) {
        //     std::cout << "Output[" << i << "]: " << output_data[i] << std::endl;
        // }
    }

    // --- End of main function ---

    return 0;
}
