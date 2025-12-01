from ultralytics import YOLO

# 1. Load the Baseline Model
print("Loading PyTorch Model...")
model = YOLO('yolov8n-pose.pt')

# 2. Export to ONNX (Industry Standard for Inference)
# dynamic=True allows the model to handle different batch sizes if needed
print("Starting ONNX Export (This may take a minute)...")
path = model.export(format='onnx', dynamic=True, device=0)

print(f"Export Complete! Model saved to: {path}")