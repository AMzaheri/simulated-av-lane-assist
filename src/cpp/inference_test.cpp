/*
This code will:
 -Include the necessary ONNX Runtime headers.
 -Initialise the ONNX Runtime environment and session.
 -Define the path to exported ONNX model (nvidia_pilotnet.onnx).
 -Create a dummy input tensor that matches the 1x1x66x200 grayscale
 image input the model expects.
 -Perform inference.
 -Print some information about the output.
To build and run:
 cd src/cpp/build
 cmake ..
 cmake --build .
And then run the updated executable:
 ./lane_keeping_inference
*/
//---------------------------------------------------
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <array> // Include for std::array

// ONNX Runtime headers
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "Hello from C++ inference_test!" << std::endl;
    std::cout << "(Now with ONNX Runtime v1.17.1 - with fixes from ResNet example)" << std::endl;

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

    // --- 7. Create Dummy Input Data ---
    const int64_t input_width = 200;
    const int64_t input_height = 66;
    const int64_t input_channels = 1;
    const int64_t batch_size = 1;

    const size_t input_tensor_size = batch_size * input_channels *
        input_height * input_width;

    std::vector<float> input_tensor_values(input_tensor_size);
    for (size_t i = 0; i < input_tensor_size; ++i) {
        input_tensor_values[i] = static_cast<float>(i) / input_tensor_size;
    }
    std::vector<int64_t> input_shape = {
        batch_size, input_channels, input_height, input_width
    };

    // --- 8. Create Input Tensor (Ort::Value) ---
    // memory_info was created earlier and is reused here.
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_size,
            input_shape.data(),
            input_shape.size()
        );

    // --- 9. Perform Inference ---
    const char* input_names_c_str[] = {input_node_names[0].get()};
    const char* output_names_c_str[] = {output_node_names[0].get()};

    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr}, // runOptions in the example, but nullptr is fine for default
        input_names_c_str,
        &input_tensor,
        1, // Number of inputs
        output_names_c_str,
        num_output_nodes // Number of outputs
    );
    // --- 10. Process Output ---
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Inference successful! Predicted Steering Angle "
              << "(dummy input): " << output_data[0] << std::endl;

    // --- Cleanup ---
    // Ort::AllocatedStringPtr and Ort::TypeInfo objects manage their own memory.
    // Ort::AllocatorWithDefaultOptions also manages its internal OrtAllocator*.
    // No explicit cleanup needed for these.

    return 0;
}
