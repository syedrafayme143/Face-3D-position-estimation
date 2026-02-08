"""
3D Face Tracking System using OpenCV DNN with Kalman Filter
Real-time face detection and position estimation using OpenCV DNN (Caffe/TensorFlow models)

Model files will be downloaded automatically on first run.
"""

import dearpygui.dearpygui as dpg
import cv2
import os
import pickle
import numpy as np
import math

from Utils.lin_kalman import LinKalman
from Utils.Dataplot import DataPlot

# Constants
CALIBRATION_FILE = 'Data/calibration_data.pkl'
FACE_WIDTH_MM = 140  # Average width of a human face in mm
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


class FaceGuiDNN:
    """Face tracking GUI using OpenCV DNN with Kalman filtering."""
    
    def __init__(self):
        """Initialize the Face GUI with camera, calibration, DNN, Kalman filter, and plotting setup."""
        # Create data plot for raw and filtered measurements (6 curves)
        self.xyzplot = DataPlot(
            ("x", "y", "z", "x_kalman", "y_kalman", "z_kalman"), 
            1000
        )

        # Initialize DNN Face Detector
        print("Loading OpenCV DNN model...")
        self._load_dnn_model()
        
        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            raise RuntimeError("Failed to open camera")
        
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Original video resolution: {self.width}x{self.height}")

        # Load calibration data
        self._load_calibration()
        
        # Kalman filter will be initialized in createKalman()
        self.kalman = None
        
        # Detection confidence threshold
        self.confidence_threshold = 0.5
        
        # Track last valid detection for stability
        self.last_valid_box = None
        self.frames_since_detection = 0
        self.max_frames_without_detection = 5

        # --- NEW FEATURE: Initialize Video Writer variables ---
        self.video_writer = None
        self.output_dir = "Results"
        os.makedirs(self.output_dir, exist_ok=True) # Create Results folder if it doesn't exist

    def _load_dnn_model(self):
        """Load OpenCV DNN face detection model - downloads automatically if needed."""
        # Model file paths
        model_file = "Models/res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "Models/deploy.prototxt"
        
        # Check if model files exist
        if not os.path.exists(model_file) or not os.path.exists(config_file):
            print("DNN model files not found. Downloading...")
            
            # URLs for model files
            config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            try:
                import urllib.request
                
                # Download config file
                if not os.path.exists(config_file):
                    print("Downloading deploy.prototxt...")
                    urllib.request.urlretrieve(config_url, config_file)
                    print("✓ Config file downloaded")
                
                # Download model file (larger, ~10MB)
                if not os.path.exists(model_file):
                    print("Downloading model file (~10MB, please wait)...")
                    
                    def download_progress(block_num, block_size, total_size):
                        if total_size > 0:
                            downloaded = block_num * block_size
                            percent = min(100, (downloaded / total_size) * 100)
                            print(f"\rProgress: {percent:.1f}%", end='')
                    
                    urllib.request.urlretrieve(model_url, model_file, download_progress)
                    print("\n✓ Model file downloaded successfully!")
                
            except Exception as e:
                print(f"\n✗ Automatic download failed: {e}")
                print("\nPlease download manually:")
                print(f"1. Config: {config_url}")
                print(f"   Save as: {config_file}")
                print(f"2. Model: {model_url}")
                print(f"   Save as: {model_file}")
                raise RuntimeError("Failed to download DNN model files. Please download manually.")
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("✓ OpenCV DNN model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load DNN model: {e}")

    def _load_calibration(self):
        """Load camera calibration data and initialize undistortion maps."""
        if not os.path.exists(CALIBRATION_FILE):
            raise FileNotFoundError(
                f"Calibration file '{CALIBRATION_FILE}' not found. "
                "Please run camera calibration first."
            )
        
        try:
            with open(CALIBRATION_FILE, 'rb') as f:
                calibration_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration data: {e}")

        mtx = calibration_data['camera_matrix']
        dist = calibration_data['distortion_coefficients']
        
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")
        
        # Create optimal camera matrix for TARGET resolution
        self.K, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist,
            (TARGET_WIDTH, TARGET_HEIGHT),
            1,
            (TARGET_WIDTH, TARGET_HEIGHT)
        )
        
        # Initialize undistortion maps at TARGET resolution
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            mtx, dist, None, self.K,
            (TARGET_WIDTH, TARGET_HEIGHT),
            cv2.CV_16SC2
        )

    def createWindow(self):
        """Create the DearPyGUI window with data plots."""
        with dpg.window(tag="Status"):
            with dpg.group(horizontal=True):
                self.xyzplot.createGUI(-1, -1, label="OpenCV DNN: Face Position Raw vs Kalman (mm)")

    def processFace(self, K, u0, v0, u1, v1):
        """
        Calculate 3D position of face center from bounding box.
        
        Args:
            K: Camera matrix (3x3)
            u0, v0: Top-left corner of face bounding box
            u1, v1: Bottom-right corner of face bounding box
            
        Returns:
            tuple: (x, y, z) position in mm relative to camera center
        """
        # Extract camera parameters
        f0 = K[0, 0]  # Focal length in x (pixels)
        f1 = K[1, 1]  # Focal length in y (pixels)
        cu = K[0, 2]  # Principal point x (pixels)
        cv = K[1, 2]  # Principal point y (pixels)
        
        # Convert to centered coordinates (origin at principal point)
        u0_centered = u0 - cu
        u1_centered = u1 - cu
        v0_centered = v0 - cv
        v1_centered = v1 - cv
        
        # Calculate angles to left and right edges of face
        angle1 = math.atan2(u0_centered, f0)
        angle2 = math.atan2(u1_centered, f0)
        
        # Calculate distance using known face width
        angle_diff = math.sin(angle2) - math.sin(angle1)
        if abs(angle_diff) < 0.001:  # Avoid division by very small numbers
            return (0, 0, 500)  # Return default distance
            
        z = FACE_WIDTH_MM / angle_diff
        
        # Calculate x, y position of face center
        u_center = (u0_centered + u1_centered) / 2
        v_center = (v0_centered + v1_centered) / 2
        
        x = (z / f0) * u_center
        y = -(z / f1) * v_center  # Negative because image y-axis points down
        
        return (x, y, z)

    def project3DFaceto2D(self, x, y, z):
        """
        Project 3D face position back to 2D image coordinates.
        
        Args:
            x, y, z: 3D position in mm
            
        Returns:
            tuple: (u0, v0, u1, v1) bounding box corners in pixels
        """
        if z < 10:  # Avoid division by very small z
            raise ValueError("Z value too small for projection")
            
        # Left edge of face
        u0 = ((x - FACE_WIDTH_MM / 2) / z) * self.K[0, 0] + self.K[0, 2]
        v0 = -((y - FACE_WIDTH_MM / 2) / z) * self.K[1, 1] + self.K[1, 2]
        
        # Right edge of face
        u1 = ((x + FACE_WIDTH_MM / 2) / z) * self.K[0, 0] + self.K[0, 2]
        v1 = -((y + FACE_WIDTH_MM / 2) / z) * self.K[1, 1] + self.K[1, 2]
        
        return int(u0), int(v0), int(u1), int(v1)

    def draw3DFace(self, frame, x, y, z, color):
        """
        Draw a bounding box for a 3D face position.
        
        Args:
            frame: Image to draw on
            x, y, z: 3D position in mm
            color: BGR color tuple
        """
        try:
            u0, v0, u1, v1 = self.project3DFaceto2D(x, y, z)
            
            # Check if coordinates are within frame bounds
            if 0 <= u0 < TARGET_WIDTH and 0 <= v0 < TARGET_HEIGHT:
                cv2.rectangle(frame, (u0, v0), (u1, v1), color, 2)
        except (OverflowError, ValueError, ZeroDivisionError):
            # Handle cases where projection fails
            pass

    def createKalman(self):
        """
        Initialize the Kalman filter for 3D position tracking.
        
        State: [x, y, z]
        Process model: constant position (A = I)
        Measurement model: direct observation (H = I)
        """
        # Initial state (will be updated with first measurement)
        x0 = np.array([0.0, 0.0, 500.0])  # Start at 500mm depth
        
        # State transition matrix (constant position model)
        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Observation matrix (direct measurement of position)
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Measurement noise covariance - DNN is quite accurate
        R = np.array([
            [4.0, 0.0, 0.0],       # DNN has good precision
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 25.0]       # z still has more noise
        ])
        
        # Process noise covariance
        Q = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        self.kalman = LinKalman(x0, A, H, R, Q)
        print("Kalman filter initialized for OpenCV DNN")

    def stepKalman(self, frame, xmm, ymm, zmm):
        """
        Run one step of Kalman filter: predict and update.
        
        Args:
            frame: Image to draw predictions on
            xmm, ymm, zmm: Measured position in mm
            
        Returns:
            tuple: (x_filtered, y_filtered, z_filtered) Kalman filtered position
        """
        # Predict step (draw predicted position in green)
        xhat, Phat = self.kalman.predictState()
        self.draw3DFace(frame, xhat[0], xhat[1], xhat[2], (0, 255, 0))
        
        # Update step with measurement
        z_measurement = np.array([xmm, ymm, zmm])
        x_updated, P_updated = self.kalman.update(z_measurement)
        
        # Draw filtered position in red
        self.draw3DFace(frame, x_updated[0], x_updated[1], x_updated[2], (0, 0, 255))
        
        return x_updated[0], x_updated[1], x_updated[2]

    def detect_faces(self, frame):
        """
        Detect faces using OpenCV DNN.
        
        Args:
            frame: Input image
            
        Returns:
            list: List of detected face bounding boxes [(x, y, w, h, confidence), ...]
        """
        h, w = frame.shape[:2]
        
        # Prepare blob for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Set input and perform forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within frame bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                width = endX - startX
                height = endY - startY
                
                faces.append((startX, startY, width, height, confidence))
        
        return faces

    def is_valid_face(self, x, y, w, h):
        """
        Validate if detection is a reasonable face.
        
        Args:
            x, y, w, h: Face bounding box
            
        Returns:
            bool: True if box seems valid for a face
        """
        # Minimum size
        if w < 80 or h < 80:
            return False
        
        # Maximum size
        if w > TARGET_WIDTH * 0.7 or h > TARGET_HEIGHT * 0.7:
            return False
        
        # Aspect ratio check
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False
        
        return True

    def run(self):
        """Main application loop."""
        dpg.create_context()
        dpg.create_viewport(title="OpenCV DNN Face Tracking with Kalman Filter", width=800, height=600)
        self.createWindow()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Status", True)
        
        frameno = 0
        
        # Initialize Kalman filter
        self.createKalman()

        # --- NEW FEATURE: Define Video Writer ---
        # Note: Saving as .mp4 because standard GIFs are too large for HD webcam video.
        # GitHub supports .mp4 uploads in issues and readmes.
        output_file = os.path.join(self.output_dir, 'dnn_face_tracking_result.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        # Warning: VideoWriter requires exact size match with frame
        self.video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (TARGET_WIDTH, TARGET_HEIGHT))
        print(f"Recording video to: {output_file}")
        print("Focus the video window and press 'q' to stop recording and exit.")

        try:
            while dpg.is_dearpygui_running():
                # Read frame from camera
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Resize frame to target resolution
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                # Undistort frame using pre-computed maps
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

                # Detect faces with DNN
                faces = self.detect_faces(frame)
                
                # Find best valid face detection
                current_detection = None
                if faces:
                    # Filter for valid faces
                    valid_faces = [(x, y, w, h, conf) for (x, y, w, h, conf) in faces 
                                   if self.is_valid_face(x, y, w, h)]
                    
                    if valid_faces:
                        # Select face with highest confidence
                        best_face = max(valid_faces, key=lambda f: f[4])
                        x, y, w, h, conf = best_face
                        current_detection = {
                            'box': (x, y, w, h),
                            'confidence': conf
                        }
                
                # Use current detection or last valid one (for stability)
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
                
                # Process detection if available
                if detection:
                    x, y, w, h = detection['box']
                    conf = detection['confidence']
                    
                    # Draw DNN detection box (cyan)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    
                    # Calculate 3D position
                    xmm, ymm, zmm = self.processFace(self.K, x, y, x + w, y + h)
                    
                    # Draw raw measurement (blue)
                    self.draw3DFace(frame, xmm, ymm, zmm, (255, 0, 0))
                    
                    # Apply Kalman filter
                    x_k, y_k, z_k = self.stepKalman(frame, xmm, ymm, zmm)

                    # Update plot with both raw and filtered data
                    self.xyzplot.addDataVector(
                        frameno,
                        (xmm, ymm, zmm, x_k, y_k, z_k)
                    )
                    
                    # Display raw measurements on frame
                    text = f"Raw: x={xmm:.0f} y={ymm:.0f} z={zmm:.0f}mm"
                    cv2.putText(
                        frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                    
                    # Display filtered measurements
                    text_filtered = f"Filtered: x={x_k:.0f} y={y_k:.0f} z={z_k:.0f}mm"
                    cv2.putText(
                        frame, text_filtered, (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )
                    
                    # Display confidence
                    text_conf = f"Conf: {conf:.2f}"
                    cv2.putText(
                        frame, text_conf, (x, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    
                    # Show if using cached detection
                    if self.frames_since_detection > 0:
                        text_cache = f"Cached ({self.frames_since_detection})"
                        cv2.putText(
                            frame, text_cache, (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1
                        )
                    
                    frameno += 1
                
                # Add legend to video
                cv2.putText(frame, "OpenCV DNN Method (ResNet SSD)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Cyan: Face | Blue: Raw | Green: Pred | Red: Filtered", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # --- NEW FEATURE: Save Frame to Video ---
                if self.video_writer is not None:
                    self.video_writer.write(frame)

                # Display frame
                cv2.imshow('OpenCV DNN Face Tracking', frame)
                
                # Allow OpenCV window events to process
                # --- MODIFIED: Ensure 'q' breaks the loop cleanly ---
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stop signal received...")
                    break

                # Render GUI
                dpg.render_dearpygui_frame()
                
        finally:
            # Cleanup
            print("Cleaning up resources...")
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"Video saved to {output_file}")

            self.video_capture.release()
            cv2.destroyAllWindows()
            dpg.destroy_context()


if __name__ == "__main__":
    try:
        gui = FaceGuiDNN()
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()