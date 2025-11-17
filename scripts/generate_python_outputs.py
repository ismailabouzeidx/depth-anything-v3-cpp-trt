#!/usr/bin/env python3
"""
Generate Python Depth Anything 3 outputs for comparison with C++ implementation.
Run this script in an environment with Python dependencies installed.
"""
import os
import sys
import json
import cv2
import numpy as np
import torch
from safetensors import safe_open

# Add Depth-Anything-3 to path
script_dir = os.path.dirname(os.path.abspath(__file__))
depth_anything_path = os.path.join(script_dir, '../../Depth-Anything-3/src')
sys.path.insert(0, depth_anything_path)

try:
    from depth_anything_3.api import DepthAnything3, SAFETENSORS_NAME
    from depth_anything_3.utils.model_loading import convert_general_state_dict
except ImportError as e:
    print(f"Error: Failed to import Depth Anything 3 modules: {e}")
    print(f"Make sure you're in an environment with depth-anything-3 installed")
    print(f"  Install with: pip install depth-anything-3")
    sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python Depth Anything 3 outputs")
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
        default="python_output",
        help="Output directory for Python depth maps",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    image_paths = []
    for img in args.images:
        if os.path.isabs(img):
            image_paths.append(img)
        else:
            # Try relative to project root first
            full_path = os.path.join(project_root, img)
            if os.path.exists(full_path):
                image_paths.append(full_path)
            elif os.path.exists(img):
                image_paths.append(img)
            else:
                print(f"Warning: Image not found: {img}")
    
    if not image_paths:
        print("Error: No valid images found")
        return 1
    
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Python Depth Anything 3 Output Generator")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Images: {len(image_paths)}")
    for img in image_paths:
        print(f"  - {img}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return 1
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    try:
        with open(os.path.join(args.model_dir, "config.json"), "r") as f:
            config = json.load(f)
        model_name = config.get("model_name", "da3mono-large")
        
        print(f"Model name: {model_name}")
        model = DepthAnything3(model_name=model_name)
        
        model_path = os.path.join(args.model_dir, SAFETENSORS_NAME)
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return 1
        
        print(f"Loading weights from {model_path}...")
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        state_dict = convert_general_state_dict(state_dict)
        missed, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missed keys: {len(missed)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        model = model.to(device=device)
        model.eval()
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Force sequential processing
    print("\nConfiguring for sequential processing...")
    original_call = model.input_processor.__call__
    import types
    def sequential_wrapper(self, *args, **kwargs):
        kwargs['sequential'] = True
        kwargs['num_workers'] = 1
        return original_call(*args, **kwargs)
    model.input_processor.__call__ = types.MethodType(sequential_wrapper, model.input_processor)
    
    # Run inference
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)
    try:
        prediction = model.inference(image_paths)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        model.input_processor.__call__ = original_call
        return 1
    finally:
        model.input_processor.__call__ = original_call
    
    print(f"\nInference completed!")
    print(f"  Depth shape: {prediction.depth.shape}")
    if prediction.processed_images is not None:
        print(f"  Processed images shape: {prediction.processed_images.shape}")
    
    # Save outputs
    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)
    
    for i in range(prediction.depth.shape[0]):
        depth = prediction.depth[i].cpu().numpy()
        
        # Get original image name for output filename
        if i < len(image_paths):
            img_name = os.path.basename(image_paths[i])
            base_name = os.path.splitext(img_name)[0]
        else:
            base_name = f"image_{i:03d}"
        
        # Normalize depth to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Save 16-bit depth map
        depth_16bit = (depth_normalized * 65535).astype(np.uint16)
        depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
        cv2.imwrite(depth_path, depth_16bit)
        print(f"  Saved: {depth_path} (shape: {depth.shape}, range: [{depth_min:.4f}, {depth_max:.4f}])")
        
        # Save colored depth map
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
        colored_path = os.path.join(output_dir, f"{base_name}_depth_colored.png")
        cv2.imwrite(colored_path, colored)
        print(f"  Saved: {colored_path}")
    
    print(f"\n✓ All outputs saved to: {output_dir}")
    print(f"\nNext step: Compare with C++ outputs using:")
    print(f"  python3 scripts/compare_depths.py --python_dir {output_dir} --cpp_dir build/output")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

