# Depth Anything 3 C++ TensorRT

C++ implementation of Depth Anything 3 using TensorRT for optimized inference.

## Project Structure

```
depth_anything_v3_cpp_trt/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/                # Header files
│   ├── tensorrt_engine.h   # TensorRT engine wrapper
│   ├── preprocessing.h     # Image preprocessing functions
│   └── postprocessing.h    # Depth map postprocessing functions
├── src/                    # Source files
│   ├── main.cpp            # Main entry point
│   ├── tensorrt_engine.cpp # TensorRT engine implementation
│   ├── preprocessing.cpp   # Preprocessing implementation
│   └── postprocessing.cpp  # Postprocessing implementation
├── scripts/                # Conversion scripts
│   ├── export_onnx.py      # ONNX export script
│   └── export_tensorrt.py  # TensorRT engine conversion script
└── models/                 # Model files (place engine here)
```

## Dependencies

- CUDA (11.0+)
- TensorRT (10.0+)
- OpenCV (4.0+)
- CMake (3.18+)
- C++17 compiler

## Building

```bash
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/path/to/tensorrt
make
```

## Usage

```bash
# For engines exported with num_images=2 (default)
./depth_anything_3_cpp_trt model.engine image1.jpg image2.jpg -o output/

# For engines exported with num_images=1
./depth_anything_3_cpp_trt model.engine image1.jpg -o output/
```

The application automatically detects the expected number of images from the engine's input shape and validates your input accordingly. See [Using Engines with Different Input Sizes](#using-engines-with-different-input-sizes) for more details.

## Model Conversion

1. Export PyTorch model to ONNX:
```bash
python3 scripts/export_onnx.py --model_dir /path/to/model --output model.onnx
```

   **Note:** You can specify the number of images per batch:
   ```bash
   # For single image inference (num_images=1)
   python3 scripts/export_onnx.py --model_dir /path/to/model --output model.onnx --num_images 1
   
   # For multi-image inference (num_images=2, default)
   python3 scripts/export_onnx.py --model_dir /path/to/model --output model.onnx --num_images 2
   ```

2. Convert ONNX to TensorRT engine:
```bash
python3 scripts/export_tensorrt.py --onnx model.onnx --output model.engine --precision fp16
```

   **Note:** The `num_images` parameter must match the ONNX export:
   ```bash
   # For single image engine
   python3 scripts/export_tensorrt.py --onnx model.onnx --output model.engine --num_images 1
   
   # For multi-image engine (default)
   python3 scripts/export_tensorrt.py --onnx model.onnx --output model.engine --num_images 2
   ```

## Using Engines with Different Input Sizes

The engine's input shape is **fixed at export time** based on the `num_images` parameter. The application automatically detects the expected number of images from the engine and validates inputs accordingly.

- **Engine exported with `num_images=1`**: Requires exactly 1 image
- **Engine exported with `num_images=2`**: Requires exactly 2 images (default)

The application will:
- Automatically detect the expected number of images from the engine's input shape
- Validate that you provide the correct number of images
- Show an error if the count doesn't match

**Example:**
```bash
# Using a single-image engine
./depth_anything_3_cpp_trt model_single.engine image1.jpg

# Using a multi-image engine (default)
./depth_anything_3_cpp_trt model_multi.engine image1.jpg image2.jpg
```

## Python vs C++ TensorRT Output Comparison

The C++ TensorRT implementation has been validated against the original Python implementation. Here are the comparison results:

### Validation Results

**Test Configuration:**
- Model: `da3mono-large`
- Input: 2 images (280x504 resolution)
- Test Images: SOH example dataset

**Comparison Metrics:**
- **Correlation**: 0.999220 (Excellent match ✓)
- **Mean Absolute Error (MAE)**: 0.008120
- **Mean Squared Error (MSE)**: 0.000173
- **Max Difference**: ~0.24-0.29 (normalized)

### Key Differences

1. **Output Size**:
   - **Python**: Keeps model output size (280x504) by default
   - **C++**: Can keep model size (default) or resize to original image size (optional)
   - The C++ implementation matches Python by default (`resize_to_original = false`)

2. **Precision**:
   - Both implementations produce nearly identical results
   - Small differences (< 1% MAE) are due to:
     - Floating-point precision differences
     - Different resizing algorithms (if resizing is enabled)
     - Normalization differences

3. **Performance**:
   - **C++ TensorRT**: Optimized for inference speed with GPU acceleration
   - **Python**: More flexible but slower for production use

### Running Comparisons

To compare outputs yourself:

```bash
# 1. Generate Python outputs
cd /home/ismo/projects/Depth-Anything-3/src
python3 save_outputs_for_comparison.py

# 2. Run C++ inference
cd /home/ismo/projects/depth_anything_v3_cpp_trt
./build/depth_anything_3_cpp_trt model.engine img1.jpg img2.jpg

# 3. Compare outputs
python3 scripts/compare_depths.py \
  --python_dir python_output \
  --cpp_dir build/output \
  --output build/comparison
```

See `COMPARISON_GUIDE.md` and `scripts/README_COMPARISON.md` for detailed comparison instructions.

## Features

- **Preprocessing**: Image loading, resizing, normalization, patch size alignment
- **Inference**: TensorRT engine execution with GPU acceleration
- **Postprocessing**: Depth map normalization, colormap visualization, resizing
- **Validation**: Verified against Python implementation (99.9%+ correlation)

