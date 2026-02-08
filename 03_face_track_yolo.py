"""
3D Face Tracking System using YOLOv8-Face Only with Kalman Filter
Real-time face detection and position estimation using YOLOv8-Face model

IMPORTANT: Download the model file manually before running:
1. Go to: https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt
2. Save the file as "yolov8n-face.pt" in this project directory
3. Run this script
"""

import dearpygui.dearpygui as dpg
import cv2
import os
import pickle
import numpy as np
import math
from ultralytics import YOLO

from Utils.lin_kalman import LinKalman
from Utils.Dataplot import DataPlot

# Constants
CALIBRATION_FILE = 'Data/calibration_data.pkl'
FACE_WIDTH_MM = 140  # Average width of a human face in mm
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


class FaceGuiYOLO:
    """Face tracking GUI using YOLOv8-Face with Kalman filtering."""
    
    def __init__(self):
        """Initialize the Face GUI with camera, calibration, YOLOv8-Face, Kalman filter, and plotting setup."""
        # Create data plot for raw and filtered measurements (6 curves)
        self.xyzplot = DataPlot(
            ("x", "y", "z", "x_kalman", "y_kalman", "z_kalman"), 
            1000
        )

        # Initialize YOLOv8-Face model
        print("Loading YOLOv8-Face model...")
        self._load_yolo_face_model()
        
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
        
        # Track last valid detection for stability
        self.last_valid_box = None
        self.frames_since_detection = 0
        self.max_frames_without_detection = 5

        # --- NEW FEATURE: Initialize Video Writer variables ---
        self.video_writer = None
        self.output_dir = "Results"
        os.makedirs(self.output_dir, exist_ok=True) # Create Results folder if it doesn't exist

    def _load_yolo_face_model(self):
        """Load YOLOv8-Face model."""
        model_path = "Models/yolov8n-face.pt"
        
        if not os.path.exists(model_path):
            print("\n" + "="*70)
            print("ERROR: YOLOv8-Face model file not found!")
            print("="*70)
            print("\nPlease download the model manually:")
            print("\nOption 1 (Recommended - HuggingFace):")
            print("  1. Open your browser and go to:")
            print("     https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
            print("  2. The file will download automatically")
            print("  3. Rename it to: yolov8n-face.pt")
            print("  4. Move it to your project folder:")
            print(f"     {os.path.abspath('.')}")
            print("\nOption 2 (Alternative - Direct Download):")
            print("  Run this command in your terminal:")
            print("  curl -L -o yolov8n-face.pt https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
            print("\nOption 3 (Using wget):")
            print("  wget -O yolov8n-face.pt https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
            print("\n" + "="*70)
            
            # Try automatic download one more time
            print("\nAttempting automatic download...")
            try:
                import urllib.request
                model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
                
                print(f"Downloading from: {model_url}")
                print("This may take a minute...")
                
                def download_progress(block_num, block_size, total_size):
                    if total_size > 0:
                        downloaded = block_num * block_size
                        percent = min(100, (downloaded / total_size) * 100)
                        print(f"\rProgress: {percent:.1f}%", end='')
                
                urllib.request.urlretrieve(model_url, model_path, download_progress)
                print("\n✓ Model downloaded successfully!")
                
            except Exception as e:
                print(f"\n✗ Automatic download failed: {e}")
                print("\nPlease download manually using the instructions above.")
                raise RuntimeError("YOLOv8-Face model not found. Please download manually.")
        
        try:
            print("Loading model...")
            self.model = YOLO(model_path)
            print("✓ YOLOv8-Face model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8-Face model: {e}")

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
                self.xyzplot.createGUI(-1, -1, label="YOLOv8-Face: Position Raw vs Kalman (mm)")

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
        
        # Measurement noise covariance - YOLOv8-Face is very accurate
        R = np.array([
            [2.0, 0.0, 0.0],       # x measurement noise
            [0.0, 2.0, 0.0],       # y measurement noise
            [0.0, 0.0, 15.0]       # z measurement noise (higher)
        ])
        
        # Process noise covariance
        Q = np.array([
            [1.0, 0.0, 0.0],       # x process noise
            [0.0, 1.0, 0.0],       # y process noise
            [0.0, 0.0, 2.0]        # z process noise
        ])
        
        self.kalman = LinKalman(x0, A, H, R, Q)
        print("Kalman filter initialized for YOLOv8-Face")

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

    def detect_faces_yolo(self, frame):
        """
        Detect faces using YOLOv8-Face model.
        
        Args:
            frame: Input image
            
        Returns:
            list: List of detected faces [(x, y, w, h, confidence), ...]
        """
        # Run inference
        results = self.model(frame, verbose=False, conf=0.5, device='cpu')
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get confidence
                conf = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(TARGET_WIDTH, x2)
                y2 = min(TARGET_HEIGHT, y2)
                
                # Convert to (x, y, w, h) format
                w = x2 - x1
                h = y2 - y1
                
                faces.append((x1, y1, w, h, conf))
        
        return faces

    def is_valid_face(self, x, y, w, h):
        """
        Validate if detection is a reasonable face.
        
        Args:
            x, y, w, h: Face bounding box
            
        Returns:
            bool: True if box seems valid for a face
        """
        # Minimum size (faces should be at least 80x80 pixels)
        if w < 80 or h < 80:
            return False
        
        # Maximum size (not more than 70% of frame - probably detection error)
        if w > TARGET_WIDTH * 0.7 or h > TARGET_HEIGHT * 0.7:
            return False
        
        # Aspect ratio check (faces are roughly square to slightly tall)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False
        
        # Position check (face should not be at extreme edges)
        if x < 10 or y < 10:
            return False
        if x + w > TARGET_WIDTH - 10 or y + h > TARGET_HEIGHT - 10:
            return False
        
        return True

    def run(self):
        """Main application loop."""
        dpg.create_context()
        dpg.create_viewport(title="YOLOv8-Face Tracking with Kalman Filter", width=800, height=600)
        self.createWindow()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Status", True)
        
        frameno = 0
        
        # Initialize Kalman filter
        self.createKalman()

        # --- NEW FEATURE: Define Video Writer ---
        # Note: Saving as .mp4 because standard GIFs are too large for HD webcam video.
        # GitHub supports .mp4 uploads.
        output_file = os.path.join(self.output_dir, 'face_tracking_result.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        # Warning: VideoWriter requires exact size match with frame
        self.video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (TARGET_WIDTH, TARGET_HEIGHT))
        print(f"Recording video to: {output_file}")
        print("Press 'q' to stop recording and exit.")

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

                # Detect faces using YOLOv8-Face
                faces = self.detect_faces_yolo(frame)
                
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
                    
                    # Draw YOLOv8-Face detection box (cyan)
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
                cv2.putText(frame, "YOLOv8-Face Detection Method", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Cyan: Face | Blue: Raw | Green: Pred | Red: Filtered", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # --- NEW FEATURE: Save Frame to Video ---
                if self.video_writer is not None:
                    self.video_writer.write(frame)

                # Display frame
                cv2.imshow('YOLOv8-Face Tracking', frame)
                
                # Allow OpenCV window events to process
                # --- MODIFIED: Ensure 'q' breaks the loop cleanly ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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
        gui = FaceGuiYOLO()
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()