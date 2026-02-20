#!/usr/bin/env python3
"""
Convert YOLOv8 Segmentation model to NCNN
Using direct compilation approach
"""
import subprocess
import os
import sys

def build_onnx2ncnn():
    """Build onnx2ncnn tool from source"""
    ncnn_root = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/ultralytics/ncnn"
    build_dir = os.path.join(ncnn_root, "build")
    
    # Create build directory if it doesn't exist
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    print("Building onnx2ncnn tool...")
    print(f"Build directory: {build_dir}")
    
    # Configure with CMake
    try:
        result = subprocess.run(
            ["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DNCNN_BUILD_TOOLS=ON", ".."],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("CMake configuration output:")
            print(result.stdout)
            print(result.stderr)
        
        # Build just the onnx tools
        print("Compiling onnx2ncnn...")
        result = subprocess.run(
            ["make", "-j$(nproc)"],
            cwd=build_dir,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("Build errors:")
            print(result.stderr[-2000:])  # Last 2000 chars
        
    except Exception as e:
        print(f"Error during build: {e}")
        return False
    
    # Check if onnx2ncnn exists
    onnx2ncnn_path = os.path.join(build_dir, "tools/onnx/onnx2ncnn")
    if not os.path.exists(onnx2ncnn_path):
        # Try alternative path
        onnx2ncnn_path = os.path.join(build_dir, "tools/onnx2ncnn")
    
    if os.path.exists(onnx2ncnn_path):
        print(f"✅ onnx2ncnn built successfully at {onnx2ncnn_path}")
        return True
    else:
        print(f"❌ onnx2ncnn not found at {onnx2ncnn_path}")
        # List what's in tools directory
        tools_dir = os.path.join(build_dir, "tools")
        if os.path.exists(tools_dir):
            print(f"Available tools in {tools_dir}:")
            for root, dirs, files in os.walk(tools_dir):
                for f in files:
                    if "onnx2ncnn" in f or (os.access(os.path.join(root, f), os.X_OK) and '.' not in f):
                        full_path = os.path.join(root, f)
                        print(f"  - {full_path}")
        return False

def convert_onnx_to_ncnn():
    """Convert ONNX to NCNN"""
    onnx_path = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/best.onnx"
    output_dir = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted"
    
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        return False
    
    # Check both possible locations
    onnx2ncnn_path = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/ultralytics/ncnn/build/tools/onnx/onnx2ncnn"
    if not os.path.exists(onnx2ncnn_path):
        onnx2ncnn_path = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/ultralytics/ncnn/build/tools/onnx2ncnn"
    
    if not os.path.exists(onnx2ncnn_path):
        print(f"⚠️  onnx2ncnn tool not found, attempting to build...")
        if not build_onnx2ncnn():
            print("❌ Failed to build onnx2ncnn tool")
            return False
    
    param_path = os.path.join(output_dir, "best.param")
    bin_path = os.path.join(output_dir, "best.bin")
    
    print(f"\n🔄 Converting ONNX to NCNN...")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {param_path}")
    print(f"           {bin_path}")
    
    try:
        result = subprocess.run(
            [onnx2ncnn_path, onnx_path, param_path, bin_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Conversion failed")
            print(f"Error: {result.stderr}")
            return False
        
        print(f"✅ Conversion successful!")
        
        if os.path.exists(param_path) and os.path.exists(bin_path):
            param_size = os.path.getsize(param_path) / (1024 * 1024)
            bin_size = os.path.getsize(bin_path) / (1024 * 1024)
            print(f"\n📦 NCNN model files created:")
            print(f"   .param file: {param_path} ({param_size:.2f} MB)")
            print(f"   .bin file:   {bin_path} ({bin_size:.2f} MB)")
            
            # Try to optimize
            optimize_model(param_path, bin_path, output_dir)
            return True
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def optimize_model(param_path, bin_path, output_dir):
    """Optimize NCNN model"""
    ncnnoptimize_path = "/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/ultralytics/ncnn/build/tools/ncnnoptimize"
    
    if not os.path.exists(ncnnoptimize_path):
        print("⚠️  ncnnoptimize tool not found, skipping optimization")
        return
    
    opt_param = os.path.join(output_dir, "best_opt.param")
    opt_bin = os.path.join(output_dir, "best_opt.bin")
    
    print(f"\n🔄 Optimizing NCNN model...")
    
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
        else:
            print(f"⚠️  Optimization warning: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Optimization skipped: {e}")

def main():
    print("=" * 60)
    print("YOLOv8 Segmentation to NCNN Converter")
    print("=" * 60)
    
    if convert_onnx_to_ncnn():
        print("\n" + "=" * 60)
        print("✨ Conversion complete!")
        print("=" * 60)
        print("\n📱 You can now use these NCNN files with the Android app:")
        print("   https://github.com/Digital2Slave/ncnn-android-yolov8-seg")
        print("\nNext steps:")
        print("1. Copy best.param and best.bin to your Android project")
        print("2. Update the model path in the Android app code")
        print("3. Build and run the app on your device")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Conversion failed")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
