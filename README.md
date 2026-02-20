# YOLOv8 Segmentation Model to NCNN Conversion Pipeline

Complete toolkit for converting YOLOv8 segmentation models from PyTorch format to NCNN format for optimized inference on edge devices and mobile platforms.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Output Files](#output-files)
- [Android Integration](#android-integration)
- [FAQ](#faq)

## 🎯 Overview

This repository provides a complete pipeline to convert YOLOv8 segmentation models through the following transformation:

```
PyTorch Model (.pt)
        ↓
    ONNX Model (.onnx)
        ↓
    NCNN Model (.param + .bin)
        ↓
    Ready for Mobile/Edge Deployment
```

### What is NCNN?

NCNN is a high-performance neural network inference framework optimized for mobile platforms. It provides:
- ✅ Lightweight (~500KB library)
- ✅ Fast inference on CPUs
- ✅ Support for ARM, x86, and other architectures
- ✅ Easy integration with Android apps

## ✨ Features

- **Automated Conversion Pipeline**: One-command conversion from PyTorch to NCNN
- **ONNX Intermediate Format**: Uses ONNX for compatibility and debugging
- **Model Optimization**: Optional NCNN model optimization for smaller file sizes
- **Detailed Logging**: Clear, color-coded output for easy tracking
- **Error Handling**: Graceful error reporting with helpful diagnostics
- **Pre-configured Settings**: Optimized default parameters for YOLOv8-seg models

## 📦 Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL2)
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for larger models)
- **Disk Space**: 500MB minimum free space

### Required Software

1. **Python 3.8+** with pip
2. **Git** (for cloning repositories)
3. **Build Tools** (for NCNN compilation):
   - CMake 3.10+
   - C++ compiler (GCC/Clang/MSVC)
   - Protocol Buffers compiler

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ncnn_conveted.git
cd ncnn_conveted
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
# Install Ultralytics YOLOv8
pip install ultralytics

# Install ONNX and optimization tools
pip install onnx onnxruntime onnxslim

# (Optional) Install additional dependencies
pip install opencv-python numpy torch torchvision
```

### Step 4: Build NCNN with ONNX Support

The NCNN library with ONNX tools is already included in the `ultralytics/ncnn/` directory.

**If you need to rebuild NCNN:**

```bash
cd ultralytics/ncnn
mkdir -p build
cd build

# Configure CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_OPENMP=ON \
      -DNCNN_VULKAN=OFF \
      -DNCNN_PLATFORM_API=ON \
      -DNCNN_BUILD_TOOLS=ON \
      ..

# Build
make -j$(nproc)
cd ../../..
```

## ⚡ Quick Start

### The Easiest Way: One Command

```bash
python3 convert_complete_pipeline.py
```

This command will:
1. ✅ Load your `best.pt` model
2. ✅ Export to ONNX format
3. ✅ Convert ONNX to NCNN
4. ✅ Optimize the NCNN model
5. ✅ Generate all output files

**Expected Output:**
```
╔════════════════════════════════════════════════════════════════════╗
║               YOLOv8 Segmentation Model Conversion Pipeline        ║
║                    PyTorch → ONNX → NCNN                          ║
╚════════════════════════════════════════════════════════════════════╝

✅ ONNX export successful!
✅ NCNN conversion successful!
✅ Model is ready for deployment!
```

## 📖 Detailed Usage

### Method 1: Using the Complete Pipeline Script (Recommended)

**What it does:**
- Converts `best.pt` → `best.onnx` → `best.param` + `best.bin`
- Validates each step
- Displays detailed progress information
- Lists all generated files

**How to use:**

```bash
# Ensure your best.pt is in the current directory
python3 convert_complete_pipeline.py
```

**Output:**
- `best.onnx` - ONNX intermediate format
- `best.param` - NCNN parameter file (network architecture)
- `best.bin` - NCNN weight file (model parameters)
- `best_opt.param` & `best_opt.bin` - Optimized versions (if optimization succeeds)

### Method 2: Step-by-Step Manual Conversion

If you prefer more control, convert each step manually:

#### Step 1: Export to ONNX

```python
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Export to ONNX
model.export(
    task='segment',
    format='onnx',
    opset=12,
    imgsz=640,
    simplify=True
)
```

#### Step 2: Convert ONNX to NCNN

```bash
# Using the onnx2ncnn tool
./ultralytics/ncnn/build/tools/onnx/onnx2ncnn best.onnx best.param best.bin
```

#### Step 3: (Optional) Optimize NCNN Model

```bash
# Optimize for faster inference
./ultralytics/ncnn/build/tools/ncnnoptimize best.param best.bin best_opt.param best_opt.bin
```

### Method 3: Using Custom Model Paths

Edit `convert_complete_pipeline.py` to use a custom model path:

```python
# Change this line in the main() function:
model_path = "/path/to/your/custom_model.pt"
```

Then run:
```bash
python3 convert_complete_pipeline.py
```

## 📁 Project Structure

```
ncnn_conveted/
├── README.md                          # This file
├── best.pt                            # Your YOLOv8-seg model
├── best.onnx                          # ONNX format (generated)
├── best.param                         # NCNN architecture (generated)
├── best.bin                           # NCNN weights (generated)
├── convert_complete_pipeline.py       # Main conversion script
├── convert_onnx_to_ncnn.py           # ONNX to NCNN converter
├── convert_with_pnnx.py              # Alternative PNNX method
├── ultralytics/                       # Ultralytics YOLOv8 library
│   ├── ultralytics/                   # Core YOLOv8 code
│   │   ├── nn/
│   │   │   └── modules/
│   │   │       └── head.py            # Fixed Segment/Pose classes
│   │   └── models/
│   └── ncnn/                          # NCNN library and tools
│       └── build/
│           └── tools/
│               ├── onnx/
│               │   └── onnx2ncnn      # ONNX to NCNN converter
│               └── ncnnoptimize       # Model optimizer
└── requirements.txt                   # Python dependencies
```

## ❓ Troubleshooting

### Issue 1: "best.pt not found"

**Error:**
```
❌ Error: Model not found at /path/to/best.pt
```

**Solution:**
- Ensure `best.pt` is in the correct directory
- Check the path in `convert_complete_pipeline.py` matches your setup
- Use absolute paths to avoid directory confusion

```bash
# Check file exists
ls -lh best.pt
```

### Issue 2: "onnx2ncnn tool not found"

**Error:**
```
❌ Error: onnx2ncnn tool not found
Please ensure NCNN is built with ONNX support
```

**Solution:**
Rebuild NCNN with ONNX support:

```bash
cd ultralytics/ncnn/build
cmake -DNCNN_BUILD_TOOLS=ON ..
make -j$(nproc)
cd ../../..
```

### Issue 3: ONNX Export Fails

**Error:**
```
❌ ONNX export failed: [error message]
```

**Solutions:**
- Update Ultralytics: `pip install -U ultralytics`
- Install required tools: `pip install onnx onnxruntime onnxslim`
- Try with different opset version (change `opset=12` to `opset=11` or `opset=13`)

### Issue 4: Out of Memory During Conversion

**Error:**
```
RuntimeError: CUDA out of memory or Memory error
```

**Solutions:**
- Use CPU-only: Delete CUDA devices before running
- Reduce batch size in export parameters
- Use a smaller model (YOLOv8n instead of YOLOv8l)

```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python3 convert_complete_pipeline.py
```

### Issue 5: Optimization Fails

**Error:**
```
⚠️ Optimization skipped or failed
```

**Solution:**
- This is non-critical; your model files are still usable
- Optimization is optional for faster inference
- Skip this step if optimization fails

## 📊 Output Files Explained

After successful conversion, you'll have:

### `best.pt` (Original PyTorch)
- **Size**: Typically 5-20 MB
- **Format**: PyTorch tensor format
- **Use**: Training and benchmarking
- **Not needed** for mobile deployment

### `best.onnx` (ONNX Intermediate)
- **Size**: Typically 8-15 MB
- **Format**: Open Neural Network Exchange
- **Use**: Format conversion, debugging, model inspection
- **Tool**: View with [Netron.app](https://netron.app)
- **Keep** for future conversions or format changes

### `best.param` (NCNN Architecture)
- **Size**: Typically 50KB - 1MB
- **Format**: NCNN network definition
- **Use**: Defines model architecture and operations
- **Required** for Android app with `best.bin`
- **Keep**: Always needed for deployment

### `best.bin` (NCNN Weights)
- **Size**: Typically 5-15 MB
- **Format**: NCNN binary weights
- **Use**: Model parameters and weights
- **Required** for Android app with `best.param`
- **Keep**: Always needed for deployment

### `best_opt.param` & `best_opt.bin` (Optimized)
- **Optional** files created by optimization step
- **Size**: Usually 20-30% smaller than original
- **Use**: Faster inference on mobile devices
- **Benefit**: Reduced model size, slightly faster execution
- **Keep**: Use these files if optimization succeeds

## 📱 Android Integration

### Using the Converted Model in Android

The NCNN model files (`best.param` and `best.bin`) are ready for Android deployment.

#### Step 1: Copy Model Files

```bash
# Copy to Android project
cp best.param /path/to/android/project/app/src/main/assets/
cp best.bin /path/to/android/project/app/src/main/assets/
```

#### Step 2: Update Android Code

In your Android app (Java/Kotlin):

```java
// Load NCNN model
Net net = new Net();
net.load_param("best.param");    // Load architecture
net.load_model("best.bin");      // Load weights
```

#### Step 3: Use Reference Implementation

Use the [Digital2Slave/ncnn-android-yolov8-seg](https://github.com/Digital2Slave/ncnn-android-yolov8-seg) repository as a reference for complete Android integration.

**Key Points:**
- Input: (1, 3, 640, 640) BCHW tensor
- Output: Detection and mask predictions
- Preprocessing: Normalization and resizing
- Postprocessing: NMS, mask decoding

## ❓ FAQ

### Q1: What's the difference between the original model and NCNN model?

**A:** The model architecture and weights remain the same. The format is optimized for mobile inference:
- PyTorch: Full framework overhead, not optimized for mobile
- NCNN: Lightweight, fast CPU inference, minimal dependencies

### Q2: Will the conversion affect model accuracy?

**A:** No. The model accuracy remains the same. Conversion is format-only; weights and architecture are preserved.

### Q3: Can I use the same NCNN files on different devices?

**A:** Yes, NCNN is cross-platform. The same `.param` and `.bin` files work on:
- Android phones and tablets
- iOS devices
- Linux systems
- Windows systems
- Raspberry Pi and other ARM devices

### Q4: How much faster will my model run with NCNN?

**A:** Speed improvement depends on:
- Hardware (CPU type and cores)
- Model size (YOLOv8n is faster than YOLOv8m)
- Typical improvement: 2-5x faster than PyTorch on CPU

### Q5: Can I convert other YOLO variants?

**A:** Yes! This pipeline works for:
- ✅ YOLOv8 Detect (detection)
- ✅ YOLOv8 Segment (segmentation)
- ✅ YOLOv8 Pose (pose estimation)
- ✅ YOLOv8 Classify (classification)

Just change the `task` parameter:
```python
model.export(task='detect', ...)  # For detection
model.export(task='pose', ...)    # For pose estimation
```

### Q6: What if I have a different model name (not best.pt)?

**A:** Edit the script to use your model path:

```python
# In convert_complete_pipeline.py, change:
model_path = "/path/to/your/model.pt"
```

### Q7: How do I verify the conversion was successful?

**A:** Check for these indicators:
```bash
# All files should exist and have reasonable sizes
ls -lh best.pt best.onnx best.param best.bin

# Expected sizes:
# best.pt:     5-20 MB
# best.onnx:   8-15 MB
# best.param:  50KB-1MB
# best.bin:    5-15 MB
```

### Q8: Can I undo the conversion?

**A:** No need - your original `best.pt` is preserved. You can keep both PyTorch and NCNN formats.

## 🔍 Model Architecture Details

For your reference, here's what happens during conversion:

### Input Specification
- **Format**: BCHW (Batch, Channel, Height, Width)
- **Shape**: (1, 3, 640, 640)
- **Data Type**: Float32
- **Normalization**: ImageNet standard

### Output Specification
**Detection Head:**
- Shape: (1, 8400, 85) for detection
- Contains: Bounding boxes + class probabilities

**Segmentation Head:**
- Mask Prototypes: (1, 32, 25600)
- Mask Coefficients: (1, 32, 8400)

## 📚 References

- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **NCNN Framework**: https://github.com/Tencent/ncnn
- **NCNN Android Example**: https://github.com/Digital2Slave/ncnn-android-yolov8-seg
- **ONNX Format**: https://onnx.ai/
- **Model Visualization**: https://netron.app

## 🛠️ Advanced Usage

### Custom Export Parameters

Modify the export settings in `convert_complete_pipeline.py`:

```python
model.export(
    task="segment",
    format="onnx",
    opset=13,                    # ONNX opset version
    imgsz=640,                   # Input image size
    simplify=True,               # Simplify model
    dynamic=False,               # Dynamic input shapes
    batch=1,                     # Batch size
)
```

### Batch Processing Multiple Models

Create a loop in the script:

```python
models = ["model1.pt", "model2.pt", "model3.pt"]
for model_file in models:
    model_path = f"/path/to/{model_file}"
    export_to_onnx(model_path)
    convert_onnx_to_ncnn(model_path.replace('.pt', '.onnx'))
```

## 📝 License

This conversion pipeline uses:
- **Ultralytics YOLOv8**: AGPL-3.0 License
- **NCNN**: BSD 3-Clause License

## 💬 Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [FAQ](#faq)
3. Check Ultralytics GitHub: https://github.com/ultralytics/ultralytics/issues
4. Check NCNN GitHub: https://github.com/Tencent/ncnn/issues

## ✅ Checklist Before Deployment

- [ ] `best.pt` successfully exported to `best.onnx`
- [ ] `best.onnx` successfully converted to `best.param` and `best.bin`
- [ ] File sizes are reasonable (not suspiciously small)
- [ ] `best.param` and `best.bin` are in your Android project assets
- [ ] Android app code updated with correct model paths
- [ ] Tested on target Android device

## 🚀 Getting Started Now

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Ensure best.pt is in current directory
ls -lh best.pt

# 3. Run conversion
python3 convert_complete_pipeline.py

# 4. Check output files
ls -lh best.param best.bin

# 5. Copy to Android project
cp best.param best.bin /path/to/android/assets/
```

That's it! Your model is ready for mobile deployment! 🎉

---

**Last Updated**: February 2026  
**Tested with**: YOLOv8n-seg, YOLOv8-seg variants  
**Python Version**: 3.8+  
**Status**: ✅ Production Ready
