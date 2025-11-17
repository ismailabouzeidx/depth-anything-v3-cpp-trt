# Comparing Python and C++ Outputs

This guide explains how to compare the outputs from the Python (original) and C++ TensorRT implementations.

## Step 1: Run Python Model

First, run the Python Depth Anything 3 model to generate reference outputs:

```bash
cd /home/ismo/projects/Depth-Anything-3/src
python3 test.py
```

Or create a simple script to save outputs:

```python
import glob, os, torch, json, cv2, numpy as np
from safetensors import safe_open
from depth_anything_3.api import DepthAnything3, SAFETENSORS_NAME
from depth_anything_3.utils.model_loading import convert_general_state_dict

device = torch.device("cuda")
model_dir = "/home/ismo/projects/Depth-Anything-3/da3mono-large"

# Load model
with open(os.path.join(model_dir, "config.json"), "r") as f:
    config = json.load(f)
model_name = config.get("model_name", "da3mono-large")

model = DepthAnything3(model_name=model_name)
model_path = os.path.join(model_dir, SAFETENSORS_NAME)
state_dict = {}
with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

state_dict = convert_general_state_dict(state_dict)
model.load_state_dict(state_dict, strict=False)
model = model.to(device=device)
model.eval()

# Run inference
image_paths = [
    "../../assets/examples/SOH/000.png",
    "../../assets/examples/SOH/010.png"
]

# Force sequential processing
original_call = model.input_processor.__call__
import types
def sequential_wrapper(self, *args, **kwargs):
    kwargs['sequential'] = True
    kwargs['num_workers'] = 1
    return original_call(*args, **kwargs)
model.input_processor.__call__ = types.MethodType(sequential_wrapper, model.input_processor)

prediction = model.inference(image_paths)
model.input_processor.__call__ = original_call

# Save outputs
output_dir = "python_output"
os.makedirs(output_dir, exist_ok=True)

for i in range(prediction.depth.shape[0]):
    depth = prediction.depth[i].cpu().numpy()
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_16bit = (depth_normalized * 65535).astype(np.uint16)
    cv2.imwrite(f"{output_dir}/python_{i:03d}_depth.png", depth_16bit)
    
    depth_8bit = (depth_normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
    cv2.imwrite(f"{output_dir}/python_{i:03d}_depth_colored.png", colored)
```

## Step 2: Run C++ Model

```bash
cd /home/ismo/projects/depth_anything_v3_cpp_trt
./build/DepthAnything3TRT ../models/da3mono-large/model.engine \
  ../../Depth-Anything-3/assets/examples/SOH/000.png \
  ../../Depth-Anything-3/assets/examples/SOH/010.png
```

This will create outputs in `build/output/`:
- `000_depth.png`
- `010_depth.png`
- `000_depth_colored.png`
- `010_depth_colored.png`

## Step 3: Compare Outputs

Use the comparison script:

```bash
cd /home/ismo/projects/depth_anything_v3_cpp_trt
python3 scripts/compare_depths.py \
  --python_dir /path/to/python_output \
  --cpp_dir build/output \
  --output build/comparison
```

The script will:
1. Load depth maps from both implementations
2. Compute statistics (MAE, MSE, Correlation)
3. Generate comparison visualizations:
   - `diff_*.png` - Difference maps (hot colormap)
   - `comparison_*.png` - Side-by-side: Python | C++ | Difference

## Expected Results

If the implementations match correctly, you should see:
- **Correlation > 0.99**: Excellent match
- **Correlation > 0.95**: Good match
- **MAE < 0.01**: Very low mean absolute error

## Troubleshooting

- **Shape mismatch**: The script automatically resizes if needed
- **Different ranges**: Both are normalized to [0, 1] before comparison
- **Missing files**: Make sure both Python and C++ outputs exist

