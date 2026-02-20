#!/usr/bin/env python3
"""
Convert YOLOv8 Segmentation model to NCNN using PNNX
"""
import subprocess
import os
import sys

# Try to import pnnx
try:
    import pnnx
except ImportError:
    print("Installing pnnx...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pnnx"], check=True)
    import pnnx

# Paths
onnx_model = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/best.onnx"
output_dir = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted"

if not os.path.exists(onnx_model):
    print(f"Error: ONNX model not found at {onnx_model}")
    sys.exit(1)

print("Converting ONNX to NCNN via PNNX...")
print(f"Input:  {onnx_model}")
print(f"Output directory: {output_dir}")

try:
    # Convert ONNX to PNNX and then to NCNN
    param_path = os.path.join(output_dir, "best.param")
    bin_path = os.path.join(output_dir, "best.bin")
    
    # Use pnnx to convert
    pnnx.convert(onnx_model, param_path, bin_path, use_fp16=False)
    
    print("\n✅ Conversion successful!")
    print(f"NCNN model files created:")
    print(f"  .param file: {param_path}")
    print(f"  .bin file:   {bin_path}")
    
except Exception as e:
    print(f"Error during conversion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
