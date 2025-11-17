# Comparing Python and C++ Outputs

## Quick Start

### Step 1: Generate Python Outputs

Run the Python model in an environment with dependencies:

```bash
cd /home/ismo/projects/depth_anything_v3_cpp_trt
python3 scripts/generate_python_outputs.py \
  --images ../../Depth-Anything-3/assets/examples/SOH/000.png \
          ../../Depth-Anything-3/assets/examples/SOH/010.png \
  --output_dir python_output
```

**Note:** This requires:
- Python 3.8+
- PyTorch
- depth-anything-3 package
- CUDA (for GPU inference)

If you get import errors, install dependencies:
```bash
pip install torch depth-anything-3 opencv-python numpy
```

### Step 2: Compare Outputs

Once Python outputs are generated:

```bash
cd /home/ismo/projects/depth_anything_v3_cpp_trt
python3 scripts/compare_depths.py \
  --python_dir python_output \
  --cpp_dir output \
  --output comparison_output
```

This will:
- Load depth maps from both implementations
- Compute comparison statistics (MAE, MSE, Correlation)
- Generate visualization files:
  - `diff_*.png` - Difference maps
  - `comparison_*.png` - Side-by-side: Python | C++ | Difference

## Expected Results

If implementations match correctly:
- **Correlation > 0.99**: Excellent match ✓
- **Correlation > 0.95**: Good match ✓
- **MAE < 0.01**: Very low error

## Troubleshooting

**Python dependencies not available?**
- Run `generate_python_outputs.py` in a different environment (conda, venv, etc.)
- Or manually run the Python model and save outputs to a directory

**Shape mismatch?**
- The comparison script automatically resizes if needed

**Different value ranges?**
- Both outputs are normalized to [0, 1] before comparison

## Files Generated

After running the comparison:
```
build/comparison/
├── diff_000.png          # Difference visualization
├── diff_010.png
├── comparison_000.png    # Side-by-side: Python | C++ | Difference
└── comparison_010.png
```

