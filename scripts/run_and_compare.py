#!/usr/bin/env python3
"""
Run Python model and compare with C++ outputs.
This script will:
1. Run Python Depth Anything 3 inference
2. Compare outputs with C++ TensorRT outputs
3. Generate comparison visualizations
"""
import os
import sys
import numpy as np
import cv2

# Try to import Python dependencies
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Depth-Anything-3/src'))
    import torch
    from safetensors import safe_open
    from depth_anything_3.api import DepthAnything3, SAFETENSORS_NAME
    from depth_anything_3.utils.model_loading import convert_general_state_dict
    PYTHON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Python dependencies not available: {e}")
    print("  Install with: pip install torch depth-anything-3")
    PYTHON_AVAILABLE = False


def run_python_inference(model_dir, image_paths, output_dir):
    """Run Python inference."""
    if not PYTHON_AVAILABLE:
        print("Error: Python dependencies not available")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    import json
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    model_name = config.get("model_name", "da3mono-large")
    
    print(f"Initializing model: {model_name}")
    model = DepthAnything3(model_name=model_name)
    
    model_path = os.path.join(model_dir, SAFETENSORS_NAME)
    print(f"Loading weights from {model_path}...")
    state_dict = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    state_dict = convert_general_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device)
    model.eval()
    
    print("\n=== Running Python Inference ===")
    
    # Force sequential processing
    original_call = model.input_processor.__call__
    import types
    def sequential_wrapper(self, *args, **kwargs):
        kwargs['sequential'] = True
        kwargs['num_workers'] = 1
        return original_call(*args, **kwargs)
    model.input_processor.__call__ = types.MethodType(sequential_wrapper, model.input_processor)
    
    try:
        prediction = model.inference(image_paths)
    finally:
        model.input_processor.__call__ = original_call
    
    print(f"Python output shape: {prediction.depth.shape}")
    
    # Save Python outputs
    python_output_dir = os.path.join(output_dir, "python_output")
    os.makedirs(python_output_dir, exist_ok=True)
    
    depth_maps = []
    for i in range(prediction.depth.shape[0]):
        depth = prediction.depth[i].cpu().numpy()
        depth_maps.append(depth)
        
        # Normalize and save
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_16bit = (depth_normalized * 65535).astype(np.uint16)
        depth_path = os.path.join(python_output_dir, f"python_{i:03d}_depth.png")
        cv2.imwrite(depth_path, depth_16bit)
        print(f"  Saved: {depth_path}")
        
        # Save colored
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
        colored_path = os.path.join(python_output_dir, f"python_{i:03d}_depth_colored.png")
        cv2.imwrite(colored_path, colored)
    
    return depth_maps


def load_cpp_outputs(cpp_dir):
    """Load C++ outputs."""
    if not os.path.exists(cpp_dir):
        return None
    
    depth_files = sorted([f for f in os.listdir(cpp_dir) if f.endswith("_depth.png")])
    if not depth_files:
        return None
    
    depth_maps = []
    for depth_file in depth_files:
        depth_path = os.path.join(cpp_dir, depth_file)
        depth_16bit = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_16bit is None:
            continue
        depth = depth_16bit.astype(np.float32) / 65535.0
        depth_maps.append(depth)
    
    return depth_maps


def compare_depths(py_depths, cpp_depths, output_dir):
    """Compare Python and C++ depth maps."""
    print("\n=== Comparing Outputs ===")
    
    if not py_depths:
        print("Error: No Python outputs to compare")
        return
    
    if not cpp_depths:
        print("Error: No C++ outputs to compare")
        return
    
    min_len = min(len(py_depths), len(cpp_depths))
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    all_stats = []
    
    for i in range(min_len):
        py_depth = py_depths[i]
        cpp_depth = cpp_depths[i]
        
        # Resize if needed
        if py_depth.shape != cpp_depth.shape:
            cpp_depth = cv2.resize(cpp_depth, (py_depth.shape[1], py_depth.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        py_norm = (py_depth - py_depth.min()) / (py_depth.max() - py_depth.min() + 1e-8)
        cpp_norm = (cpp_depth - cpp_depth.min()) / (cpp_depth.max() - cpp_depth.min() + 1e-8)
        
        # Statistics
        diff = np.abs(py_norm - cpp_norm)
        mae = np.mean(diff)
        mse = np.mean(diff ** 2)
        max_diff = np.max(diff)
        
        py_flat = py_norm.flatten()
        cpp_flat = cpp_norm.flatten()
        correlation = np.corrcoef(py_flat, cpp_flat)[0, 1] if len(py_flat) > 1 else 0.0
        
        stats = {'mae': mae, 'mse': mse, 'max_diff': max_diff, 'correlation': correlation}
        all_stats.append(stats)
        
        print(f"\n  Image {i}:")
        print(f"    MAE: {mae:.6f}, MSE: {mse:.6f}, Max Diff: {max_diff:.6f}")
        print(f"    Correlation: {correlation:.6f}")
        
        # Save visualizations
        diff_vis = (diff * 255).astype(np.uint8)
        diff_colored = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(comparison_dir, f"diff_{i:03d}.png"), diff_colored)
        
        py_colored = cv2.applyColorMap((py_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cpp_colored = cv2.applyColorMap((cpp_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        comparison = np.hstack([py_colored, cpp_colored, diff_colored])
        cv2.imwrite(os.path.join(comparison_dir, f"comparison_{i:03d}.png"), comparison)
    
    # Summary
    print(f"\n=== Summary ===")
    avg_mae = np.mean([s['mae'] for s in all_stats])
    avg_corr = np.mean([s['correlation'] for s in all_stats])
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average Correlation: {avg_corr:.6f}")
    
    if avg_corr > 0.99:
        print("✓ Excellent match!")
    elif avg_corr > 0.95:
        print("✓ Good match!")
    elif avg_corr > 0.90:
        print("⚠ Moderate match")
    else:
        print("✗ Poor match")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/home/ismo/projects/Depth-Anything-3/da3mono-large")
    parser.add_argument("--images", nargs="+", 
                       default=["../../Depth-Anything-3/assets/examples/SOH/000.png",
                               "../../Depth-Anything-3/assets/examples/SOH/010.png"])
    parser.add_argument("--cpp_dir", default="build/output")
    parser.add_argument("--output", default="build/comparison_output")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Resolve paths
    image_paths = [os.path.join(project_root, img) if not os.path.isabs(img) else img 
                   for img in args.images]
    cpp_dir = os.path.join(project_root, args.cpp_dir)
    output_dir = os.path.join(project_root, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Python inference
    py_depths = None
    if PYTHON_AVAILABLE:
        py_depths = run_python_inference(args.model_dir, image_paths, output_dir)
    else:
        print("\nSkipping Python inference (dependencies not available)")
        print("  To compare, install: pip install torch depth-anything-3")
    
    # Load C++ outputs
    cpp_depths = load_cpp_outputs(cpp_dir)
    if not cpp_depths:
        print(f"\nError: No C++ outputs found in {cpp_dir}")
        return 1
    
    # Compare
    if py_depths:
        compare_depths(py_depths, cpp_depths, output_dir)
    else:
        print("\nCannot compare - Python outputs not available")
        return 1
    
    print(f"\n✓ Complete! Results in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

