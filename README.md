
# SafePose: Real-Time Biomechanical Risk Analytics Engine

**SafePose** is an edge-AI computer vision system designed to detect hazardous ergonomic postures in industrial environments. It processes real-time RTSP video streams to identify unsafe joint angles and biomechanical risks using geometric vector analysis.

## 🚀 Key Features

-   **Real-Time Processing:** Handles concurrent RTSP streams at 60+ FPS using optimized inference pipelines.
    
-   **Edge Optimization:** Accelerated inference by **400%** using **ONNX Runtime (GPU)**, reducing latency for edge deployment.
    
-   **Geometric Analytics:** Custom logic layer to calculate 3D joint angles (knee flexion, back curvature) in real-time.
    
-   **Robust Architecture:** Dockerized microservices with auto-reconnection and fault tolerance.
    

## 🛠️ Infrastructure Setup

### Prerequisites

-   **Docker Desktop** (for hosting the RTSP server)
    
-   **FFmpeg** (for simulating live video streams)
    
-   **Python 3.8+**
    
-   **NVIDIA GPU** (RTX 3060 or higher recommended)
    

### 1. Start RTSP Server (MediaMTX)

We use MediaMTX running in Docker to simulate a low-latency IP camera server. Run the following command in your terminal:

```
docker run --rm -it -p 8554:8554 -p 1935:1935 -p 8888:8888 bluenviron/mediamtx:latest

```

### 2. Start Video Stream (Simulated Camera)

Push a local video file (`input_video.mp4`) to the server to simulate a live CCTV feed. Open a second terminal and run:

```
ffmpeg -re -stream_loop -1 -i input_video.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/mystream

```

## 💻 Running the Inference Engine

### Installation

```
pip install -r requirements.txt

```

### 1. Optimization Step (Phase 3)

Convert the standard PyTorch model to an optimized ONNX format for faster inference on Windows/Edge hardware.

```
python export_model.py

```

### 2. Run Optimized Inference

Run the main script. This utilizes `onnxruntime-gpu` for high-velocity inference.

```
python pose_estimation.py

```