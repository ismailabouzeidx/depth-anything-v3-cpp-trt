#!/usr/bin/env python3
"""
Export ONNX model to TensorRT engine.
"""
import os
import argparse
import numpy as np
import torch

try:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    print("ERROR: TensorRT not installed. Install with: pip install nvidia-tensorrt")
    exit(1)


def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB
    input_shape: tuple = (1, 2, 3, 280, 504),
    dynamic_shapes: bool = False,
):
    """
    Build a TensorRT engine from an ONNX model.
    
    Args:
        onnx_file_path: Path to the ONNX model file
        engine_file_path: Path to save the TensorRT engine
        precision: Precision mode - 'fp32', 'fp16', or 'int8'
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes
        input_shape: Input shape tuple (B, N, C, H, W)
        dynamic_shapes: Whether to enable dynamic shapes
    """
    print(f"Building TensorRT engine from: {onnx_file_path}")
    print(f"Output: {engine_file_path}")
    print(f"Precision: {precision}")
    print(f"Input shape: {input_shape}")
    
    # Initialize TensorRT
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network with explicit batch flag
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    try:
        with open(onnx_file_path, 'rb') as model_file:
            model_data = model_file.read()
        
        if not parser.parse(model_data):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(f"  Error {error}: {parser.get_error(error)}")
            return None
    except Exception as e:
        print(f"ERROR: Exception while parsing ONNX file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"✓ ONNX model parsed successfully")
    print(f"  Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"  Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
    
    # Configure builder
    config = builder.create_builder_config()
    # TensorRT 10.x uses set_memory_pool_limit instead of max_workspace_size
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    except AttributeError:
        # Fallback for older TensorRT versions
        config.max_workspace_size = max_workspace_size
    
    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 precision enabled")
        else:
            print("⚠ FP16 not supported on this platform, using FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✓ INT8 precision enabled")
        else:
            print("⚠ INT8 not supported on this platform, using FP32")
    else:
        print("✓ FP32 precision (default)")
    
    # Configure input shapes
    if dynamic_shapes:
        print("Configuring dynamic shapes...")
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        
        # Set min, opt, max shapes
        B, N, C, H, W = input_shape
        profile.set_shape(
            input_tensor.name,
            (1, 1, C, H//2, W//2),  # min
            (B, N, C, H, W),        # opt
            (B*2, N*2, C, H*2, W*2) # max
        )
        config.add_optimization_profile(profile)
        print(f"  Dynamic shape range: min=(1,1,{C},{H//2},{W//2}), opt={input_shape}, max=({B*2},{N*2},{C},{H*2},{W*2})")
    else:
        # Set static input shape
        input_tensor = network.get_input(0)
        input_tensor.shape = input_shape
        print(f"  Static input shape: {input_shape}")
    
    # Build engine
    print("\nBuilding TensorRT engine (this may take a while)...")
    try:
        # TensorRT 10.x uses build_serialized_network instead of build_engine
        try:
            # TensorRT 10.x API
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                print("ERROR: Failed to build engine")
                return None
            
            print("✓ Engine built successfully")
            
            # Save engine
            print(f"Saving engine to: {engine_file_path}")
            os.makedirs(os.path.dirname(engine_file_path) if os.path.dirname(engine_file_path) else ".", exist_ok=True)
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)
            
            engine_size = os.path.getsize(engine_file_path) / (1024**2)
            print(f"✓ Engine saved successfully")
            print(f"  Engine size: {engine_size:.2f} MB")
            
            # Load engine to get info
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            if engine:
                print(f"\nEngine Information:")
                # TensorRT 10.x uses tensor names, not indices
                try:
                    inputs = [name for i in range(engine.num_io_tensors) 
                             for name in [engine.get_tensor_name(i)]
                             if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
                    outputs = [name for i in range(engine.num_io_tensors) 
                              for name in [engine.get_tensor_name(i)]
                              if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
                    print(f"  Inputs: {inputs}")
                    print(f"  Outputs: {outputs}")
                except Exception as e:
                    # Fallback: just list all tensor names
                    try:
                        all_tensors = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
                        print(f"  All tensors: {all_tensors}")
                    except:
                        print(f"  (Could not retrieve tensor information)")
            
            return engine
            
        except AttributeError:
            # Fallback for older TensorRT versions (< 10.x)
            engine = builder.build_engine(network, config)
            if engine is None:
                print("ERROR: Failed to build engine")
                return None
            
            print("✓ Engine built successfully")
            
            # Save engine
            print(f"Saving engine to: {engine_file_path}")
            os.makedirs(os.path.dirname(engine_file_path) if os.path.dirname(engine_file_path) else ".", exist_ok=True)
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
            
            engine_size = os.path.getsize(engine_file_path) / (1024**2)
            print(f"✓ Engine saved successfully")
            print(f"  Engine size: {engine_size:.2f} MB")
            
            # Print engine info
            print(f"\nEngine Information:")
            print(f"  Inputs: {[engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]}")
            print(f"  Outputs: {[engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]}")
            print(f"  Max batch size: {engine.max_batch_size}")
            
            return engine
        
    except Exception as e:
        print(f"ERROR: Failed to build engine: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_engine(engine_file_path: str, input_shape: tuple = (1, 2, 3, 280, 504)):
    """
    Test the TensorRT engine with dummy input.
    """
    print(f"\nTesting engine: {engine_file_path}")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("⚠ pycuda not installed, skipping engine test")
        print("  Install with: pip install pycuda")
        return
    
    # Load engine
    with open(engine_file_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        print("ERROR: Failed to load engine")
        return
    
    # Create context
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    # Prepare input
    B, N, C, H, W = input_shape
    dummy_input = np.random.randn(B, N, C, H, W).astype(np.float32)
    np.copyto(inputs[0]['host'], dummy_input.ravel())
    
    # Transfer input to GPU
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer output from GPU
    output_shape = (B, N, H, W)
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    
    output = outputs[0]['host'].reshape(output_shape)
    
    print(f"✓ Engine test successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine")
    parser.add_argument(
        "--onnx",
        type=str,
        default="/home/ismo/projects/Depth-Anything-3/da3mono-large/model.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ismo/projects/Depth-Anything-3/da3mono-large/model.engine",
        help="Path to save TensorRT engine",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Precision mode (default: fp16)",
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
        "--workspace",
        type=int,
        default=4096,
        help="Workspace size in MB (default: 4096)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic shapes",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the engine after building",
    )
    
    args = parser.parse_args()
    
    # Check if ONNX file exists
    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX file not found: {args.onnx}")
        exit(1)
    
    input_shape = (args.batch_size, args.num_images, 3, args.height, args.width)
    max_workspace_size = args.workspace * (1024**2)  # Convert MB to bytes
    
    engine = build_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.output,
        precision=args.precision,
        max_batch_size=args.batch_size,
        max_workspace_size=max_workspace_size,
        input_shape=input_shape,
        dynamic_shapes=args.dynamic,
    )
    
    if engine is not None and args.test:
        test_engine(args.output, input_shape)

