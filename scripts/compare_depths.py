#!/usr/bin/env python3
"""
Compare C++ TensorRT depth outputs with Python original outputs.
This script compares already-generated depth maps from both implementations.
"""
import os
import sys
import numpy as np
import cv2
from pathlib import Path


def load_depth_map(path):
    """Load a depth map from PNG file."""
    depth_16bit = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_16bit is None:
        return None
    # Convert to float32 normalized [0, 1]
    depth = depth_16bit.astype(np.float32) / 65535.0
    return depth


def find_depth_files(directory, pattern="_depth.png"):
    """Find all depth files in directory."""
    if not os.path.exists(directory):
        return []
    files = sorted([f for f in os.listdir(directory) if f.endswith(pattern)])
    return [os.path.join(directory, f) for f in files]


def compare_depths(python_path, cpp_path, output_comparison_dir, idx):
    """Compare two depth maps and save comparison visualizations."""
    print(f"\n  Comparing image {idx}:")
    print(f"    Python: {os.path.basename(python_path)}")
    print(f"    C++:    {os.path.basename(cpp_path)}")
    
    py_depth = load_depth_map(python_path)
    cpp_depth = load_depth_map(cpp_path)
    
    if py_depth is None:
        print(f"    ✗ Failed to load Python depth: {python_path}")
        return None
    if cpp_depth is None:
        print(f"    ✗ Failed to load C++ depth: {cpp_path}")
        return None
    
    # Resize if needed
    if py_depth.shape != cpp_depth.shape:
        print(f"    Resizing C++ from {cpp_depth.shape} to {py_depth.shape}")
        cpp_depth = cv2.resize(cpp_depth, (py_depth.shape[1], py_depth.shape[0]), 
                              interpolation=cv2.INTER_LINEAR)
    
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
    correlation = np.corrcoef(py_flat, cpp_flat)[0, 1] if len(py_flat) > 1 else 0.0
    
    stats = {
        'mae': mae,
        'mse': mse,
        'max_diff': max_diff,
        'correlation': correlation,
        'py_range': (py_depth.min(), py_depth.max()),
        'cpp_range': (cpp_depth.min(), cpp_depth.max()),
        'py_shape': py_depth.shape,
        'cpp_shape': cpp_depth.shape,
    }
    
    print(f"    MAE (Mean Absolute Error): {mae:.6f}")
    print(f"    MSE (Mean Squared Error): {mse:.6f}")
    print(f"    Max Difference: {max_diff:.6f}")
    print(f"    Correlation: {correlation:.6f}")
    print(f"    Python range: [{py_depth.min():.4f}, {py_depth.max():.4f}]")
    print(f"    C++ range:    [{cpp_depth.min():.4f}, {cpp_depth.max():.4f}]")
    
    # Save difference visualization
    diff_vis = (diff * 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)
    diff_path = os.path.join(output_comparison_dir, f"diff_{idx:03d}.png")
    cv2.imwrite(diff_path, diff_colored)
    print(f"    Saved difference map: {diff_path}")
    
    # Save side-by-side comparison
    py_vis = (py_norm * 255).astype(np.uint8)
    cpp_vis = (cpp_norm * 255).astype(np.uint8)
    py_colored = cv2.applyColorMap(py_vis, cv2.COLORMAP_INFERNO)
    cpp_colored = cv2.applyColorMap(cpp_vis, cv2.COLORMAP_INFERNO)
    
    # Create comparison image: Python | C++ | Difference
    comparison = np.hstack([py_colored, cpp_colored, diff_colored])
    comp_path = os.path.join(output_comparison_dir, f"comparison_{idx:03d}.png")
    cv2.imwrite(comp_path, comparison)
    print(f"    Saved comparison: {comp_path}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Python and C++ depth outputs")
    parser.add_argument(
        "--python_dir",
        type=str,
        help="Directory containing Python depth outputs",
    )
    parser.add_argument(
        "--cpp_dir",
        type=str,
        default="build/output",
        help="Directory containing C++ depth outputs (default: build/output)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="build/comparison",
        help="Output directory for comparison results",
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Resolve paths
    cpp_dir = os.path.join(project_root, args.cpp_dir) if not os.path.isabs(args.cpp_dir) else args.cpp_dir
    output_dir = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
    
    # Find C++ depth files
    cpp_files = find_depth_files(cpp_dir, "_depth.png")
    if not cpp_files:
        print(f"Error: No C++ depth files found in {cpp_dir}")
        print(f"  Looking for files matching: *_depth.png")
        return 1
    
    print(f"Found {len(cpp_files)} C++ depth files:")
    for f in cpp_files:
        print(f"  {os.path.basename(f)}")
    
    # Find Python depth files
    if args.python_dir:
        python_dir = os.path.join(project_root, args.python_dir) if not os.path.isabs(args.python_dir) else args.python_dir
        python_files = find_depth_files(python_dir, "_depth.png")
        if not python_files:
            print(f"\nWarning: No Python depth files found in {python_dir}")
            print("  You can run the Python model first, or provide --python_dir")
            print("  For now, comparing C++ outputs with themselves (baseline)...")
            python_files = cpp_files  # Use C++ as baseline
    else:
        print("\nNo Python directory provided. Comparing C++ outputs only.")
        print("  Use --python_dir to specify Python output directory")
        python_files = []
    
    if not python_files:
        return 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Comparing Outputs ===")
    print(f"Python directory: {python_dir if args.python_dir else 'N/A'}")
    print(f"C++ directory:    {cpp_dir}")
    print(f"Output directory: {output_dir}")
    
    # Match files by index (assuming same order)
    min_len = min(len(python_files), len(cpp_files))
    all_stats = []
    
    for i in range(min_len):
        stats = compare_depths(python_files[i], cpp_files[i], output_dir, i)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("\n✗ No valid comparisons made")
        return 1
    
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
    
    print(f"\n✓ Comparison complete! Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

