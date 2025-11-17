#!/usr/bin/env python3
"""
Compare Python (original) and C++ TensorRT outputs for Depth Anything 3.
"""
import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Add Depth-Anything-3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Depth-Anything-3/src'))

from safetensors import safe_open
from depth_anything_3.api import DepthAnything3, SAFETENSORS_NAME
from depth_anything_3.utils.model_loading import convert_general_state_dict


def load_python_model(model_dir):
    """Load the Python model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Read config
    import json
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    model_name = config.get("model_name", "da3mono-large")
    
    print(f"Initializing model: {model_name}")
    model = DepthAnything3(model_name=model_name)
    
    # Load weights
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
    
    return model, device


def run_python_inference(model, device, image_paths, output_dir):
    """Run Python inference and save outputs."""
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
    
    print(f"Python output shapes:")
    print(f"  depth: {prediction.depth.shape}")
    if prediction.processed_images is not None:
        print(f"  processed_images: {prediction.processed_images.shape}")
    
    # Save Python outputs
    python_output_dir = os.path.join(output_dir, "python_output")
    os.makedirs(python_output_dir, exist_ok=True)
    
    depth_maps = []
    for i in range(prediction.depth.shape[0]):
        depth = prediction.depth[i].cpu().numpy()
        depth_maps.append(depth)
        
        # Save raw depth (16-bit PNG)
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_16bit = (depth_normalized * 65535).astype(np.uint16)
        depth_path = os.path.join(python_output_dir, f"python_{i:03d}_depth.png")
        cv2.imwrite(depth_path, depth_16bit)
        print(f"  Saved: {depth_path}")
        
        # Save colored depth
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
        colored_path = os.path.join(python_output_dir, f"python_{i:03d}_depth_colored.png")
        cv2.imwrite(colored_path, colored)
        print(f"  Saved: {colored_path}")
    
    return depth_maps, prediction


def load_cpp_outputs(output_dir):
    """Load C++ outputs."""
    print("\n=== Loading C++ Outputs ===")
    
    cpp_output_dir = os.path.join(output_dir, "..", "output")
    if not os.path.exists(cpp_output_dir):
        print(f"Error: C++ output directory not found: {cpp_output_dir}")
        return None
    
    # Find depth files
    depth_files = sorted([f for f in os.listdir(cpp_output_dir) if f.endswith("_depth.png")])
    print(f"Found {len(depth_files)} C++ depth files")
    
    depth_maps = []
    for depth_file in depth_files:
        depth_path = os.path.join(cpp_output_dir, depth_file)
        depth_16bit = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_16bit is None:
            print(f"  Warning: Failed to load {depth_path}")
            continue
        
        # Convert to float32
        depth = depth_16bit.astype(np.float32) / 65535.0
        depth_maps.append(depth)
        print(f"  Loaded: {depth_file} (shape: {depth.shape})")
    
    return depth_maps


def compare_outputs(python_depths, cpp_depths, output_dir):
    """Compare Python and C++ outputs."""
    print("\n=== Comparing Outputs ===")
    
    if cpp_depths is None:
        print("Error: Cannot compare - C++ outputs not available")
        return
    
    min_len = min(len(python_depths), len(cpp_depths))
    if min_len == 0:
        print("Error: No outputs to compare")
        return
    
    print(f"Comparing {min_len} depth maps...")
    
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    all_stats = []
    
    for i in range(min_len):
        py_depth = python_depths[i]
        cpp_depth = cpp_depths[i]
        
        # Resize if needed
        if py_depth.shape != cpp_depth.shape:
            print(f"  Image {i}: Resizing C++ output from {cpp_depth.shape} to {py_depth.shape}")
            cpp_depth = cv2.resize(cpp_depth, (py_depth.shape[1], py_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Normalize both to [0, 1] for comparison
        py_norm = (py_depth - py_depth.min()) / (py_depth.max() - py_depth.min() + 1e-8)
        cpp_norm = (cpp_depth - cpp_depth.min()) / (cpp_depth.max() - cpp_depth.min() + 1e-8)
        
        # Compute statistics
        diff = np.abs(py_norm - cpp_norm)
        mae = np.mean(diff)
        mse = np.mean(diff ** 2)
        max_diff = np.max(diff)
        
        # Compute correlation
        py_flat = py_norm.flatten()
        cpp_flat = cpp_norm.flatten()
        correlation = np.corrcoef(py_flat, cpp_flat)[0, 1]
        
        stats = {
            'image_idx': i,
            'mae': mae,
            'mse': mse,
            'max_diff': max_diff,
            'correlation': correlation,
            'py_shape': py_depth.shape,
            'cpp_shape': cpp_depth.shape,
            'py_range': (py_depth.min(), py_depth.max()),
            'cpp_range': (cpp_depth.min(), cpp_depth.max()),
        }
        all_stats.append(stats)
        
        print(f"\n  Image {i}:")
        print(f"    MAE (Mean Absolute Error): {mae:.6f}")
        print(f"    MSE (Mean Squared Error): {mse:.6f}")
        print(f"    Max Difference: {max_diff:.6f}")
        print(f"    Correlation: {correlation:.6f}")
        print(f"    Python range: [{py_depth.min():.4f}, {py_depth.max():.4f}]")
        print(f"    C++ range: [{cpp_depth.min():.4f}, {cpp_depth.max():.4f}]")
        
        # Save difference visualization
        diff_vis = (diff * 255).astype(np.uint8)
        diff_colored = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)
        diff_path = os.path.join(comparison_dir, f"diff_{i:03d}.png")
        cv2.imwrite(diff_path, diff_colored)
        print(f"    Saved difference map: {diff_path}")
        
        # Save side-by-side comparison
        py_vis = (py_norm * 255).astype(np.uint8)
        cpp_vis = (cpp_norm * 255).astype(np.uint8)
        py_colored = cv2.applyColorMap(py_vis, cv2.COLORMAP_INFERNO)
        cpp_colored = cv2.applyColorMap(cpp_vis, cv2.COLORMAP_INFERNO)
        
        comparison = np.hstack([py_colored, cpp_colored, diff_colored])
        comp_path = os.path.join(comparison_dir, f"comparison_{i:03d}.png")
        cv2.imwrite(comp_path, comparison)
        print(f"    Saved comparison: {comp_path}")
    
    # Summary statistics
    print(f"\n=== Summary ===")
    avg_mae = np.mean([s['mae'] for s in all_stats])
    avg_mse = np.mean([s['mse'] for s in all_stats])
    avg_corr = np.mean([s['correlation'] for s in all_stats])
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Correlation: {avg_corr:.6f}")
    
    if avg_corr > 0.99:
        print("✓ Excellent match! Outputs are very similar.")
    elif avg_corr > 0.95:
        print("✓ Good match! Outputs are similar.")
    elif avg_corr > 0.90:
        print("⚠ Moderate match. Some differences detected.")
    else:
        print("✗ Poor match. Significant differences detected.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Python and C++ outputs")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/ismo/projects/Depth-Anything-3/da3mono-large",
        help="Directory containing config.json and model.safetensors",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=[
            "../../Depth-Anything-3/assets/examples/SOH/000.png",
            "../../Depth-Anything-3/assets/examples/SOH/010.png",
        ],
        help="Input image paths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="build/comparison_output",
        help="Output directory for comparison results",
    )
    
    args = parser.parse_args()
    
    # Resolve image paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_paths = [os.path.join(project_root, img) if not os.path.isabs(img) else img for img in args.images]
    
    # Check images exist
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            return 1
    
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Python model
    model, device = load_python_model(args.model_dir)
    
    # Run Python inference
    python_depths, prediction = run_python_inference(model, device, image_paths, output_dir)
    
    # Load C++ outputs
    cpp_depths = load_cpp_outputs(output_dir)
    
    # Compare
    compare_outputs(python_depths, cpp_depths, output_dir)
    
    print(f"\n✓ Comparison complete! Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

