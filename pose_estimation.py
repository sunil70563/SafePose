import cv2
import time
import os
import sys
import site
from ultralytics import YOLO
import geometry_utils # Import our custom math module

# --- DLL FIX (Required for Windows ONNX) ---
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
# -------------------------------------------

# FORCE TCP CONNECTION
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("------------------------------------------------")
print("SafePose Phase 4: Geometric Logic Integration")
print("------------------------------------------------")

print("Loading Optimized ONNX Model...")
try:
    model = YOLO('yolov8n-pose.onnx', task='pose')
except Exception as e:
    print(f"Error loading ONNX: {e}")
    exit()

RTSP_URL = "rtsp://localhost:8554/mystream"
print(f"Connecting to Stream: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("SafePose Analytics", cv2.WINDOW_NORMAL) 

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        cap.open(RTSP_URL)
        continue

    # Inference
    results = model(frame, verbose=False)

    for result in results:
        # 1. Get Standard Skeleton Plot
        annotated_frame = result.plot()
        
        # 2. Add Geometric Analysis Logic
        if result.keypoints is not None and result.keypoints.data is not None:
            # Move data to CPU for processing
            keypoints_list = result.keypoints.data.cpu().numpy()
            
            for person_kpts in keypoints_list:
                # Run Math Analysis
                status_text, color = geometry_utils.check_posture(person_kpts)
                
                # Draw Visuals (Elbow Point Index: 8)
                if person_kpts[8][2] > 0.5: # Confidence check
                    elbow_x, elbow_y = int(person_kpts[8][0]), int(person_kpts[8][1])
                    
                    # Draw Status Indicator Circle
                    cv2.circle(annotated_frame, (elbow_x, elbow_y), 12, color, -1)
                    # Draw Risk Text
                    cv2.putText(annotated_frame, status_text, (elbow_x - 60, elbow_y - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 3. Draw FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {int(fps)} (ONNX)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("SafePose Analytics", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()