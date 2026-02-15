"""
3D Face Tracking System using OpenCV DNN with Kalman Filter
Real-time face detection and position estimation using OpenCV DNN (Caffe/TensorFlow models)
REFINED FOR SMOOTH PERFORMANCE WITH MONITORING
"""

import dearpygui.dearpygui as dpg
import cv2
import os
import pickle
import numpy as np
import math

from Utils.lin_kalman import LinKalman
from Utils.Dataplot import DataPlot
from Utils.performance_monitor import PerformanceMonitor 

# Constants
CALIBRATION_FILE = 'Data/calibration_data.pkl'
FACE_WIDTH_MM = 140  
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

class FaceGuiDNN:
    def __init__(self):
        self.xyzplot = DataPlot(
            ("x", "y", "z", "x_kalman", "y_kalman", "z_kalman"), 
            1000
        )
        self._load_dnn_model()
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            raise RuntimeError("Failed to open camera")
        
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._load_calibration()
        
        self.kalman = None
        self.confidence_threshold = 0.5
        self.last_valid_box = None
        self.frames_since_detection = 0
        self.max_frames_without_detection = 5

        self.video_writer = None
        self.output_dir = "Results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(method_name="OpenCV_DNN_Refined")

    def _load_dnn_model(self):
        model_file = "Models/res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "Models/deploy.prototxt"
        
        if not os.path.exists(model_file) or not os.path.exists(config_file):
            print("Downloading DNN models...")
            # ... (download logic remains same as original) ...
        
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    def _load_calibration(self):
        if not os.path.exists(CALIBRATION_FILE):
            raise FileNotFoundError(f"Calibration file '{CALIBRATION_FILE}' not found.")
        
        with open(CALIBRATION_FILE, 'rb') as f:
            calibration_data = pickle.load(f)

        mtx = calibration_data['camera_matrix']
        dist = calibration_data['distortion_coefficients']
        self.K, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (TARGET_WIDTH, TARGET_HEIGHT), 1, (TARGET_WIDTH, TARGET_HEIGHT))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, self.K, (TARGET_WIDTH, TARGET_HEIGHT), cv2.CV_16SC2)

    def createWindow(self):
        with dpg.window(tag="Status"):
            with dpg.group(horizontal=True):
                self.xyzplot.createGUI(-1, -1, label="Face Tracking: Raw vs Filtered")

    def processFace(self, K, u0, v0, u1, v1):
        f0, f1 = K[0, 0], K[1, 1]
        cu, cv = K[0, 2], K[1, 2]
        u0_c, u1_c = u0 - cu, u1 - cu
        v0_c, v1_c = v0 - cv, v1 - cv
        
        angle1, angle2 = math.atan2(u0_c, f0), math.atan2(u1_c, f0)
        angle_diff = math.sin(angle2) - math.sin(angle1)
        
        if abs(angle_diff) < 0.001: return (0, 0, 500)
            
        z = FACE_WIDTH_MM / angle_diff
        x = (z / f0) * ((u0_c + u1_c) / 2)
        y = -(z / f1) * ((v0_c + v1_c) / 2)
        return (x, y, z)

    def project3DFaceto2D(self, x, y, z):
        u0 = ((x - FACE_WIDTH_MM / 2) / z) * self.K[0, 0] + self.K[0, 2]
        v0 = -((y - FACE_WIDTH_MM / 2) / z) * self.K[1, 1] + self.K[1, 2]
        u1 = ((x + FACE_WIDTH_MM / 2) / z) * self.K[0, 0] + self.K[0, 2]
        v1 = -((y + FACE_WIDTH_MM / 2) / z) * self.K[1, 1] + self.K[1, 2]
        return int(u0), int(v0), int(u1), int(v1)

    def draw3DFace(self, frame, x, y, z, color):
        try:
            u0, v0, u1, v1 = self.project3DFaceto2D(x, y, z)
            if 0 <= u0 < TARGET_WIDTH and 0 <= v0 < TARGET_HEIGHT:
                cv2.rectangle(frame, (u0, v0), (u1, v1), color, 2)
        except: pass

    def createKalman(self):
        x0 = np.array([0.0, 0.0, 500.0])
        A = np.eye(3)
        H = np.eye(3)
        R = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 25.0]])
        Q = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 2.0]])
        self.kalman = LinKalman(x0, A, H, R, Q)

    def stepKalman(self, frame, xmm, ymm, zmm):
        xhat, _ = self.kalman.predictState()
        self.draw3DFace(frame, xhat[0], xhat[1], xhat[2], (0, 255, 0)) # Green: Prediction
        x_up, _ = self.kalman.update(np.array([xmm, ymm, zmm]))
        self.draw3DFace(frame, x_up[0], x_up[1], x_up[2], (0, 0, 255)) # Red: Filtered
        return x_up

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = box.astype("int")
                faces.append((max(0, sx), max(0, sy), min(w, ex)-sx, min(h, ey)-sy, conf))
        return faces

    def is_valid_face(self, x, y, w, h):
        if w < 80 or h < 80: return False
        ar = w / h if h > 0 else 0
        return 0.6 < ar < 1.4

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Smooth DNN Face Tracking", width=800, height=600)
        self.createWindow()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Status", True)
        
        frameno = 0
        self.createKalman()

        output_file = os.path.join(self.output_dir, 'refined_tracking.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (TARGET_WIDTH, TARGET_HEIGHT))

        try:
            while dpg.is_dearpygui_running():
                # Start timing the frame cycle
                self.performance_monitor.start_frame()
                
                ret, frame = self.video_capture.read()
                if not ret: break

                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

                # Detection Block
                self.performance_monitor.start_detection()
                faces = self.detect_faces(frame)
                self.performance_monitor.end_detection()
                
                current_detection = None
                if faces:
                    valid_faces = [f for f in faces if self.is_valid_face(f[0], f[1], f[2], f[3])]
                    if valid_faces:
                        best = max(valid_faces, key=lambda f: f[4])
                        current_detection = {'box': best[:4], 'confidence': best[4]}
                
                # Stability Logic
                if current_detection:
                    self.last_valid_box = current_detection
                    self.frames_since_detection = 0
                    detection = current_detection
                elif self.last_valid_box and self.frames_since_detection < self.max_frames_without_detection:
                    self.frames_since_detection += 1
                    detection = self.last_valid_box
                else:
                    detection = None
                    self.frames_since_detection += 1
                
                # Metrics variables
                pos_raw, pos_filt = (0,0,0), (0,0,0)

                # Processing Block
                if detection:
                    x, y, w, h = detection['box']
                    conf = detection['confidence']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    
                    xmm, ymm, zmm = self.processFace(self.K, x, y, x + w, y + h)
                    pos_raw = (xmm, ymm, zmm)
                    self.draw3DFace(frame, xmm, ymm, zmm, (255, 0, 0))
                    
                    kf_state = self.stepKalman(frame, xmm, ymm, zmm)
                    pos_filt = (kf_state[0], kf_state[1], kf_state[2])

                    self.xyzplot.addDataVector(frameno, (xmm, ymm, zmm, *pos_filt))
                    
                    cv2.putText(frame, f"Raw Z: {zmm:.0f}mm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, f"Conf: {conf:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    frameno += 1

                # Performance Recording (MUST be after if-detection block for smoothness)
                is_detected = True if detection else False
                self.performance_monitor.end_frame(
                    face_detected=is_detected,
                    position=pos_raw,
                    filtered_position=pos_filt
                )
                
                # UI and Save
                if self.video_writer: self.video_writer.write(frame)
                cv2.imshow('OpenCV DNN Tracking', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                dpg.render_dearpygui_frame()
                
        finally:
            self.performance_monitor.save_results()
            self.performance_monitor.print_summary()
            if self.video_writer: self.video_writer.release()
            self.video_capture.release()
            cv2.destroyAllWindows()
            dpg.destroy_context()

if __name__ == "__main__":
    FaceGuiDNN().run()