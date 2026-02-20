#!/usr/bin/env python3
"""
Complete YOLOv8 Segmentation Model Conversion Pipeline
Converts: best.pt → best.onnx → best.param + best.bin
"""
from ultralytics import YOLO
import subprocess
import os
import sys

def export_to_onnx(model_path):
    """Export PyTorch model to ONNX format"""
    print("=" * 70)
    print("STEP 1: Export PyTorch to ONNX Format")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None
    
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("🔄 Exporting to ONNX...")
    try:
        success = model.export(
            task="segment",
            format="onnx",
            opset=12,
            imgsz=640,
            simplify=True
        )
        
        if success:
            onnx_path = model_path.replace('.pt', '.onnx')
            if os.path.exists(onnx_path):
                size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"✅ ONNX export successful!")
                print(f"   Output: {onnx_path} ({size_mb:.2f} MB)")
                return onnx_path
        return None
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def convert_onnx_to_ncnn(onnx_path):
    """Convert ONNX model to NCNN format"""
    print("\n" + "=" * 70)
    print("STEP 2: Convert ONNX to NCNN Format")
    print("=" * 70)
    
    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX model not found at {onnx_path}")
        return False
    
    # Find onnx2ncnn tool
    onnx2ncnn_paths = [
        "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/ultralytics/ncnn/build/tools/onnx/onnx2ncnn",
        "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/ultralytics/ncnn/build/tools/onnx2ncnn"
    ]
    
    onnx2ncnn_path = None
    for path in onnx2ncnn_paths:
        if os.path.exists(path):
            onnx2ncnn_path = path
            break
    
    if not onnx2ncnn_path:
        print("❌ Error: onnx2ncnn tool not found")
        print("Please ensure NCNN is built with ONNX support")
        return False
    
    output_dir = os.path.dirname(onnx_path)
    param_path = os.path.join(output_dir, "best.param")
    bin_path = os.path.join(output_dir, "best.bin")
    
    print(f"🔄 Converting ONNX to NCNN...")
    print(f"   Input:  {onnx_path}")
    print(f"   Output: {param_path}")
    print(f"           {bin_path}")
    
    try:
        result = subprocess.run(
            [onnx2ncnn_path, onnx_path, param_path, bin_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Conversion failed")
            print(f"   Error: {result.stderr}")
            return False
        
        if os.path.exists(param_path) and os.path.exists(bin_path):
            param_size = os.path.getsize(param_path) / (1024 * 1024)
            bin_size = os.path.getsize(bin_path) / (1024 * 1024)
            print(f"✅ NCNN conversion successful!")
            print(f"   .param file: {param_path} ({param_size:.2f} MB)")
            print(f"   .bin file:   {bin_path} ({bin_size:.2f} MB)")
            return True
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def optimize_ncnn_model(param_path, bin_path):
    """Optimize NCNN model"""
    print("\n" + "=" * 70)
    print("STEP 3: Optimize NCNN Model")
    print("=" * 70)
    
    ncnnoptimize_path = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/ultralytics/ncnn/build/tools/ncnnoptimize"
    
    if not os.path.exists(ncnnoptimize_path):
        print("⚠️  ncnnoptimize tool not found, skipping optimization")
        return False
    
    output_dir = os.path.dirname(param_path)
    opt_param = os.path.join(output_dir, "best_opt.param")
    opt_bin = os.path.join(output_dir, "best_opt.bin")
    
    print(f"🔄 Optimizing NCNN model...")
    
    try:
        result = subprocess.run(
            [ncnnoptimize_path, param_path, bin_path, opt_param, opt_bin],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and os.path.exists(opt_param):
            opt_param_size = os.path.getsize(opt_param) / (1024 * 1024)
            opt_bin_size = os.path.getsize(opt_bin) / (1024 * 1024)
            print(f"✅ Optimization successful!")
            print(f"   Optimized .param: {opt_param} ({opt_param_size:.2f} MB)")
            print(f"   Optimized .bin:   {opt_bin} ({opt_bin_size:.2f} MB)")
            return True
        else:
            print(f"⚠️  Optimization skipped or failed")
            return False
    except Exception as e:
        print(f"⚠️  Optimization error: {e}")
        return False

def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "YOLOv8 Segmentation Model Conversion Pipeline" + " " * 8 + "║")
    print("║" + " " * 20 + "PyTorch → ONNX → NCNN" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Paths
    model_path = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"📍 Working directory: /home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/")
    print(f"📍 Input model: {model_path}")
    print()
    
    # Step 1: Export to ONNX
    onnx_path = export_to_onnx(model_path)
    if not onnx_path:
        print("\n❌ Conversion failed at ONNX export step")
        sys.exit(1)
    
    # Step 2: Convert ONNX to NCNN
    if not convert_onnx_to_ncnn(onnx_path):
        print("\n❌ Conversion failed at NCNN conversion step")
        sys.exit(1)
    
    param_path = onnx_path.replace('.onnx', '.param')
    bin_path = onnx_path.replace('.onnx', '.bin')
    
    # Step 3: Optimize NCNN model
    optimize_ncnn_model(param_path, bin_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE! ✨")
    print("=" * 70)
    
    output_dir = os.path.dirname(onnx_path)
    print(f"\n📦 Output files in: {output_dir}")
    print()
    
    # List all output files
    for filename in ["best.pt", "best.onnx", "best.param", "best.bin", "best_opt.param", "best_opt.bin"]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✅ {filename:20s} ({size_mb:7.2f} MB)")
    
    print()
    print("📱 Next steps for Android integration:")
    print("   1. Use best.param and best.bin with the Android app")
    print("   2. Reference: https://github.com/Digital2Slave/ncnn-android-yolov8-seg")
    print("   3. Copy files to Android project and update model paths")
    print()
    print("🎯 Model is ready for deployment!")
    print()

if __name__ == "__main__":
    main()
