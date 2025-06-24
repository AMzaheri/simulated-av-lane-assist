import torch
import torch.nn as nn

# Define the NVIDIA_PilotNet class (copy-pasted from 
#(https://www.kaggle.com/code/afsanehm/deep-learning-for-simulated-driving)

class NVIDIA_PilotNet(nn.Module):
    def __init__(self):
        super(NVIDIA_PilotNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# --- Model Export Code ---

# 1. Instantiate model
model = NVIDIA_PilotNet()

# 2. Load the best trained weights
model_path = 'models/best_lane_keeping_model.pth' 

# It's good practice to map location to CPU if you trained on GPU
# and want to ensure it loads correctly on any system for export.
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval() 

# 3. Create a dummy input tensor
# This dummy input must match the expected input size of our model (Batch_Size, C, H, W)
# For the current model, it's (1, 1, 66, 200) for a single grayscale image
dummy_input = torch.randn(1, 1, 66, 200) # Batch size 1, 1 channel, 66 height, 200 width

# 4. Define the ONNX export path
onnx_model_path = "models/nvidia_pilotnet.onnx" # Will save in models/ directory

# 5. Export the model
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,        # Store the trained parameter weights inside the model file
    opset_version=11,          # The ONNX opset version to use (version 11 is common and stable)
    do_constant_folding=True,  # Whether to execute constant folding for optimization
    input_names=['input'],     # Name the input tensor
    output_names=['output'],   # Name the output tensor
    dynamic_axes={             # Define dynamic axes if the batch size can vary
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported to ONNX format at: {onnx_model_path}")

# Optional: Verify the ONNX model (requires onnx installed)
try:
    import onnx
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed!")
except Exception as e:
    print(f"ONNX model check failed: {e}")
