from ultralytics import YOLO

# Load model
model = YOLO(r"/home/ioptime/Desktop/zeeshan_farooq/ncnn_conveted/best.pt")

# Export model
success = model.export(task="segment", format="onnx", opset=12, imgsz=640, simplify=True)