import cv2
import time
import os
from ultralytics import YOLO

# FORCE TCP CONNECTION
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("Loading YOLOv8-Pose Model...")
model = YOLO('yolov8n-pose.pt') 

RTSP_URL = "rtsp://localhost:8554/mystream"
print(f"Connecting to Stream: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not connect.")
    exit()
else:
    print("SUCCESS: Stream Connected! Starting Inference...")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame drop or Stream disconnected. Reconnecting...")
        cap.open(RTSP_URL)
        time.sleep(1)
        continue

    # Inference
    results = model(frame, device=0, verbose=False)

    for result in results:
        annotated_frame = result.plot()
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Display FPS (Corrected Syntax)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("SafePose - Phase 2", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()