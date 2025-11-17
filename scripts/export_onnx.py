#!/usr/bin/env python3
"""
Export Depth Anything 3 model to ONNX format.
"""
import os
import json
import torch
import torch.onnx
from safetensors import safe_open
from depth_anything_3.api import DepthAnything3, SAFETENSORS_NAME
from depth_anything_3.utils.model_loading import convert_general_state_dict


def export_to_onnx(
    model_dir: str,
    output_path: str,
    input_height: int = 280,
    input_width: int = 504,
    batch_size: int = 1,
    num_images: int = 2,
    opset_version: int = 18,
    include_extrinsics: bool = False,
    include_intrinsics: bool = False,
):
    """
    Export DepthAnything3 model to ONNX format.
    
    Args:
        model_dir: Directory containing config.json and model.safetensors
        output_path: Path to save the ONNX model
        input_height: Input image height
        input_width: Input image width
        batch_size: Batch size (B)
        num_images: Number of images per batch (N)
        opset_version: ONNX opset version
        include_extrinsics: Whether to include extrinsics as input
        include_intrinsics: Whether to include intrinsics as input
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Read config to get model name
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    model_name = config.get("model_name", "da3mono-large")
    
    print(f"Initializing model: {model_name}")
    # Initialize model with the correct model name
    model = DepthAnything3(model_name=model_name)
    
    # Load weights from safetensors file
    model_path = os.path.join(model_dir, SAFETENSORS_NAME)
    print(f"Loading weights from {model_path}...")
    state_dict = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    # Convert state dict and load into model
    state_dict = convert_general_state_dict(state_dict)
    missed, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missed keys: {len(missed)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    # Move model to device and set to eval mode
    model = model.to(device=device)
    model.eval()
    
    # Get the underlying model (DepthAnything3Net)
    core_model = model.model
    core_model.eval()
    
    # Create dummy input tensors
    # Input shape: (B, N, 3, H, W)
    dummy_image = torch.randn(batch_size, num_images, 3, input_height, input_width, device=device)
    
    # Prepare inputs for ONNX export
    # ONNX doesn't handle None well, so we'll create a wrapper
    class ONNXModelWrapper(torch.nn.Module):
        def __init__(self, model, include_extrinsics, include_intrinsics):
            super().__init__()
            self.model = model
            self.include_extrinsics = include_extrinsics
            self.include_intrinsics = include_intrinsics
        
        def forward(self, image, extrinsics=None, intrinsics=None):
            # Convert None to actual None for the model
            if not self.include_extrinsics:
                extrinsics = None
            if not self.include_intrinsics:
                intrinsics = None
            
            # Call the model
            output = self.model(image, extrinsics, intrinsics, export_feat_layers=[], infer_gs=False)
            
            # Extract main outputs (depth is the primary output)
            # Output is an addict.Dict which supports both attribute and dict access
            depth = output.depth if hasattr(output, 'depth') else output.get("depth", None)
            if depth is not None:
                # Reshape from (B, N, 1, H, W) to (B, N, H, W) if needed
                if depth.dim() == 5 and depth.shape[2] == 1:
                    depth = depth.squeeze(2)
                return depth
            else:
                # If depth not in output, try to get the first tensor value
                if hasattr(output, 'keys'):
                    first_key = list(output.keys())[0] if output.keys() else None
                    if first_key:
                        val = output[first_key]
                        if val.dim() == 5 and val.shape[2] == 1:
                            val = val.squeeze(2)
                        return val
                # Fallback - create zero tensor with same shape as input (without channel dim)
                B, N = image.shape[0], image.shape[1]
                H, W = image.shape[3], image.shape[4]
                return torch.zeros(B, N, H, W, device=image.device, dtype=image.dtype)
    
    wrapped_model = ONNXModelWrapper(core_model, include_extrinsics, include_intrinsics)
    wrapped_model.eval()
    
    # Prepare input arguments for ONNX export
    if include_extrinsics and include_intrinsics:
        dummy_extrinsics = torch.randn(batch_size, num_images, 4, 4, device=device)
        dummy_intrinsics = torch.randn(batch_size, num_images, 3, 3, device=device)
        dummy_inputs = (dummy_image, dummy_extrinsics, dummy_intrinsics)
        input_names = ["image", "extrinsics", "intrinsics"]
        dynamic_axes = {
            "image": {0: "batch_size", 1: "num_images"},
            "extrinsics": {0: "batch_size", 1: "num_images"},
            "intrinsics": {0: "batch_size", 1: "num_images"},
        }
    elif include_extrinsics:
        dummy_extrinsics = torch.randn(batch_size, num_images, 4, 4, device=device)
        dummy_inputs = (dummy_image, dummy_extrinsics)
        input_names = ["image", "extrinsics"]
        dynamic_axes = {
            "image": {0: "batch_size", 1: "num_images"},
            "extrinsics": {0: "batch_size", 1: "num_images"},
        }
    elif include_intrinsics:
        dummy_intrinsics = torch.randn(batch_size, num_images, 3, 3, device=device)
        dummy_inputs = (dummy_image, dummy_intrinsics)
        input_names = ["image", "intrinsics"]
        dynamic_axes = {
            "image": {0: "batch_size", 1: "num_images"},
            "intrinsics": {0: "batch_size", 1: "num_images"},
        }
    else:
        dummy_inputs = (dummy_image,)
        input_names = ["image"]
        dynamic_axes = {
            "image": {0: "batch_size", 1: "num_images"},
        }
    
    output_names = ["depth"]
    
    # Add dynamic axes for spatial dimensions (channel dim 2 is fixed at 3, so skip it)
    dynamic_axes["image"].update({3: "height", 4: "width"})  # Skip dim 2 (channels=3)
    dynamic_axes["depth"] = {0: "batch_size", 1: "num_images", 2: "height", 3: "width"}
    
    print(f"\nExporting to ONNX...")
    print(f"Input shape: {dummy_image.shape}")
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    print(f"Opset version: {opset_version}")
    
    # Test forward pass first
    print("\nTesting forward pass...")
    with torch.no_grad():
        test_output = wrapped_model(*dummy_inputs)
        print(f"Test output shape: {test_output.shape}")
    
    # Export to ONNX
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Try with static shapes first (simpler, more compatible)
    # Remove dynamic axes for initial export
    static_dynamic_axes = None  # Use None for static shapes
    
    try:
        print("Attempting ONNX export with static shapes...")
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=static_dynamic_axes,  # Static shapes
            verbose=False,
        )
        print("✓ Static shape export succeeded")
    except Exception as e:
        print(f"Static shape export failed: {e}")
        print("Attempting with dynamic shapes...")
        # Try with dynamic shapes
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
    
    print(f"\n✓ ONNX model exported successfully to: {output_path}")
    print(f"  Model size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
    except ImportError:
        print("⚠ onnx package not installed, skipping verification")
    except Exception as e:
        print(f"⚠ ONNX model verification failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Depth Anything 3 to ONNX")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/ismo/projects/Depth-Anything-3/da3mono-large",
        help="Directory containing config.json and model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ismo/projects/Depth-Anything-3/da3mono-large/model.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=280,
        help="Input image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=504,
        help="Input image width",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=2,
        help="Number of images per batch",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--include_extrinsics",
        action="store_true",
        help="Include extrinsics as input",
    )
    parser.add_argument(
        "--include_intrinsics",
        action="store_true",
        help="Include intrinsics as input",
    )
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output,
        input_height=args.height,
        input_width=args.width,
        batch_size=args.batch_size,
        num_images=args.num_images,
        opset_version=args.opset,
        include_extrinsics=args.include_extrinsics,
        include_intrinsics=args.include_intrinsics,
    )

