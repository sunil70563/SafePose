import cv2
import time
import os
import sys
import site
from ultralytics import YOLO

# --- DLL FIX (Keep this if it helped you resolve the issue) ---
def add_dll_paths():
    if os.name != 'nt': return
    try:
        site_packages = site.getsitepackages()[1]
        paths = [
            os.path.join(site_packages, "onnxruntime", "capi"),
            os.path.join(site_packages, "nvidia", "cudnn", "bin"),
            os.path.join(site_packages, "nvidia", "cublas", "bin")
        ]
        for p in paths:
            if os.path.exists(p): os.add_dll_directory(p)
    except Exception: pass

add_dll_paths()
# -----------------------------------------------------------

# FORCE TCP CONNECTION
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("Loading Optimized ONNX Model...")
# Load the ONNX model
model = YOLO('yolov8n-pose.onnx', task='pose') 

RTSP_URL = "rtsp://localhost:8554/mystream"
print(f"Connecting to Stream: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("SafePose (Optimized ONNX)", cv2.WINDOW_NORMAL) 

if not cap.isOpened():
    print("Error: Could not connect to RTSP stream.")
    exit()
else:
    print("SUCCESS: Stream Connected! Running Optimization Mode...")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        # If stream ends (FFmpeg restarting), wait briefly
        time.sleep(0.1)
        cap.open(RTSP_URL)
        continue

    # Inference using ONNX
    results = model(frame, verbose=False)

    for result in results:
        annotated_frame = result.plot()
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {int(fps)} (ONNX)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("SafePose (Optimized ONNX)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()