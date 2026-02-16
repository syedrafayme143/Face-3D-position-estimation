# 3D Real-Time Face Tracking: Performance Comparison of Detection Methods

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive comparison of four leading face detection methods for real-time 3D position tracking with Kalman filtering optimization.

## 🎯 Project Overview

This project implements and benchmarks **four different face detection algorithms** for 3D position tracking:
- 🎭 **HaarCascade** (Classical CV)
- 🧠 **MediaPipe** (Google's ML Solution)
- 🚀 **YOLOv8-Face** (State-of-the-art DL)
- 🔬 **OpenCV DNN** (ResNet-based SSD)

Each method is enhanced with **Kalman filtering** for smooth, stable tracking and evaluated across **5 professional metrics**.

---

## 📊 Key Findings

### 🏆 Overall Performance Rankings

| Rank | Method | Detection | Speed (FPS) | Stability | Smoothness | Best For |
|------|--------|-----------|-------------|-----------|------------|----------|
| 🥇 | **MediaPipe** | 97.3% | 19.5 | ⭐ | ⭐⭐ | **Real-time applications** |
| 🥈 | **OpenCV DNN** | 97.3% | 9.5 | ⭐⭐ | ⭐⭐⭐ | **Balanced performance** |
| 🥉 | **YOLOv8** | 100% | 3.4 | ⭐⭐⭐ | ⭐⭐ | **Maximum accuracy** |
| 4️⃣ | **HaarCascade** | 71.0% | 5.0 | ⭐ | ⭐ | **Resource-constrained** |

### 📈 Detailed Metrics Comparison

#### 1️⃣ Detection Reliability
```
YOLOv8:      ████████████████████ 100.0%  (0 gaps)
OpenCV DNN:  ███████████████████▌ 97.3%   (3 gaps)
MediaPipe:   ███████████████████▌ 97.3%   (10 gaps)
HaarCascade: ██████████████▏      71.0%   (16 gaps)
```

#### 2️⃣ Processing Speed (FPS)
```
MediaPipe:   ███████████████████▌ 19.5 FPS  ⚡ Fastest
OpenCV DNN:  █████████▌            9.5 FPS
HaarCascade: █████                 5.0 FPS
YOLOv8:      ███▍                  3.4 FPS  🐢 Slowest but accurate
```

#### 3️⃣ Position Stability (Lower STD = Better)
```
YOLOv8:      237mm STD  ✅ Most stable
OpenCV DNN:  284mm STD
MediaPipe:   297mm STD
HaarCascade: 373mm STD  ⚠️ Least stable
```

#### 4️⃣ Tracking Smoothness (Lower Jitter = Better)
```
OpenCV DNN:  30mm jitter  ✅ Smoothest
YOLOv8:      44mm jitter
MediaPipe:   51mm jitter
HaarCascade: 101mm jitter ⚠️ Most jittery
```

#### 5️⃣ Kalman Filter Effectiveness
```
HaarCascade: 56% ✅ Best improvement
MediaPipe:   33%
OpenCV DNN:  24%
YOLOv8:      22%
```

---

## 🎬 Live Demonstrations

### MediaPipe - Fast & Reliable
![MediaPipe Demo](Results/mediapipe_tracking_demo.gif)
*Achieves 19.5 FPS with 97.3% detection rate - ideal for real-time applications*

### OpenCV DNN - Balanced Performance
![OpenCV DNN Demo](Results/dnn_tracking_demo.gif)
*Smoothest tracking (30mm jitter) with excellent 97.3% detection*

### YOLOv8 - Maximum Accuracy
![YOLOv8 Demo](Results/yolov8_tracking_demo.gif)
*Perfect 100% detection but slower at 3.4 FPS - best for accuracy-critical tasks*

**Legend:**
- 🔵 Blue: Raw measurement
- 🟢 Green: Kalman prediction
- 🔴 Red: Filtered position
- 🟡 Cyan: Detection bounding box

---

## 🛠️ Technical Architecture

### System Pipeline
```
Camera Feed → Undistortion → Detection → 3D Position Estimation → Kalman Filter → Visualization
     ↓            ↓              ↓              ↓                      ↓              ↓
  640×480    Calibrated    Face Bbox      Trigonometry          Noise Reduction   Real-time
             1920×1080                    + Camera Matrix                          Display
```

### 3D Position Calculation
The system calculates 3D position (x, y, z) in millimeters using:
1. **Known face width** (140mm average)
2. **Camera intrinsic matrix** (from calibration)
3. **Pinhole camera model** geometry
4. **Kalman filtering** for temporal smoothing

### Kalman Filter Configuration
- **State**: `[x, y, z]` position in mm
- **Model**: Constant position (no velocity)
- **Tuning**: Method-specific noise matrices for optimal performance

---

## 📦 Installation

### Prerequisites
```bash
Python 3.8+
OpenCV 4.0+
CUDA (optional, for GPU acceleration)
```

### Setup
```bash
# Clone repository
git clone https://github.com/syedrafayme143/Real-Time-Face-Tracking-Benchmark-with-Kalman-Filtering.git
cd Real-Time-Face-Tracking-Benchmark-with-Kalman-Filtering

# Create virtual environment
python -m venv face3d_env
source face3d_env/bin/activate  # Windows: face3d_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8-Face model
mkdir Models
curl -L -o Models/yolov8n-face.pt \
  https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt
```

### Camera Calibration
```bash
# Required before first use
python camera_calibration.py
# Follow on-screen instructions with checkerboard pattern
```

---

## 🚀 Usage

### Run Individual Methods
```bash
# MediaPipe (Fastest - 19.5 FPS)
python 02_face_track_mediapipe.py

# OpenCV DNN (Balanced - 9.5 FPS)
python 04_face_track_open_cv_dnn.py

# YOLOv8 (Most Accurate - 100% detection)
python 03_face_track_yolo.py

# HaarCascade (Lightweight - 5 FPS)
python 01_face_track_harrcascade.py
```

### Generate Performance Comparison
```bash
# Run all methods for 60 seconds each, then:
python 05_compare_runtime_results.py
```

### Interactive Controls
- **Q**: Quit and save results
- Real-time visualization in DearPyGUI window

---

## 📁 Project Structure

```
face-tracking-comparison/
├── 01_face_track_harrcascade.py   # HaarCascade implementation
├── 02_face_track_mediapipe.py     # MediaPipe implementation
├── 03_face_track_yolo.py          # YOLOv8-Face implementation
├── 04_face_track_open_cv_dnn.py   # OpenCV DNN implementation
├── 05_compare_runtime_results.py  # Benchmark comparison tool
├── Utils/
│   ├── lin_kalman.py              # Kalman filter implementation
│   ├── Dataplot.py                # Real-time plotting
│   └── performance_monitor.py     # Metrics collection
├── Models/
│   ├── yolov8n-face.pt            # YOLOv8 face detection model
│   ├── res10_300x300_ssd.caffemodel  # OpenCV DNN model
│   └── deploy.prototxt            # DNN configuration
├── Data/
│   └── calibration_data.pkl       # Camera calibration results
├── Results/
│   ├── Benchmarks/                # Performance JSON files
│   ├── *_demo.gif                 # Tracking demonstrations
│   └── *_plot.png                 # Position plots
└── requirements.txt
```

---

## 🔬 Methodology

### Evaluation Metrics

1. **Detection Reliability** (25% weight)
   - Detection rate percentage
   - Gap analysis (missed frame sequences)
   - Temporal consistency

2. **Processing Speed** (20% weight)
   - Average/Min/Max FPS
   - Frame time distribution (P95, P99)
   - Real-time capability assessment

3. **Position Stability** (20% weight)
   - Standard deviation per axis (X, Y, Z)
   - Overall 3D stability score
   - Noise characteristics

4. **Tracking Smoothness** (20% weight)
   - Frame-to-frame jitter
   - Motion prediction quality
   - Temporal coherence

5. **Kalman Filter Effectiveness** (15% weight)
   - Noise reduction per axis
   - Jitter reduction percentage
   - Filter performance score

### Test Conditions
- **Duration**: 60+ seconds per method
- **Environment**: Indoor, controlled lighting
- **Subject**: Static and dynamic head movements
- **Camera**: 1920×1080 resolution, calibrated
- **Hardware**: Consumer-grade CPU (no GPU required)

---

## 🎓 Use Case Recommendations

### 🚀 Real-Time Applications (VR/AR, Gaming)
**→ Choose: MediaPipe**
- ✅ Highest FPS (19.5)
- ✅ Excellent detection (97.3%)
- ✅ Low latency
- ⚠️ Moderate stability

### 🎯 High-Accuracy Requirements (Medical, Research)
**→ Choose: YOLOv8**
- ✅ Perfect detection (100%)
- ✅ Best stability (237mm STD)
- ⚠️ Lower FPS (3.4)
- ⚠️ Requires GPU for real-time

### ⚖️ Balanced Performance (Robotics, HCI)
**→ Choose: OpenCV DNN**
- ✅ Smoothest tracking (30mm jitter)
- ✅ High detection (97.3%)
- ✅ Moderate FPS (9.5)
- ✅ No external dependencies

### 💻 Resource-Constrained (Embedded, IoT)
**→ Choose: HaarCascade**
- ✅ Lightweight (5 FPS on CPU)
- ✅ No deep learning required
- ⚠️ Lower detection (71%)
- ⚠️ Higher noise

---

## 📊 Benchmark Results Summary

```
╔════════════════╦════════════╦═══════╦════════════╦═════════════╗
║ Method         ║ Detection  ║  FPS  ║ Stability  ║  Smoothness ║
╠════════════════╬════════════╬═══════╬════════════╬═════════════╣
║ MediaPipe      ║   97.3% ⭐  ║ 19.5 ⚡║   0.9%     ║    49%      ║
║ OpenCV DNN     ║   97.3% ⭐  ║  9.5  ║   5.4%     ║    70% ⭐    ║
║ YOLOv8         ║  100.0% ⭐⭐ ║  3.4  ║  21.0% ⭐   ║    56%      ║
║ HaarCascade    ║   71.0%    ║  5.0  ║   0.0%     ║     0%      ║
╚════════════════╩════════════╩═══════╩════════════╩═════════════╝
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add GPU acceleration benchmarks
- [ ] Support for multiple face tracking
- [ ] Integration with ROS/ROS2
- [ ] Mobile deployment (iOS/Android)
- [ ] Real-time performance optimization
- [ ] Additional detection methods (RetinaFace, MTCNN)

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenCV Team** - Core computer vision library
- **Google MediaPipe** - Efficient face detection solution
- **Ultralytics** - YOLOv8 implementation
- **OpenCV DNN Module** - Deep learning inference

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ and Python

</div>
