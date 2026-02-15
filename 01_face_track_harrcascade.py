"""
3D Face Tracking System with Kalman Filter
Real-time face detection and position estimation with Kalman filtering for smooth tracking
WITH PERFORMANCE MONITORING
"""

import dearpygui.dearpygui as dpg
import cv2
import os
import pickle
import numpy as np
import math

from Utils.lin_kalman import LinKalman
from Utils.Dataplot import DataPlot
from Utils.performance_monitor import PerformanceMonitor  # ADDED

# Constants
CALIBRATION_FILE = 'Data/calibration_data.pkl'
FACE_WIDTH_MM = 140  # Average width of a human face in mm
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


class FaceGui:
    """Face tracking GUI with Kalman filtering."""
    
    def __init__(self):
        """Initialize the Face GUI with camera, calibration, Kalman filter, and plotting setup."""
        # Create data plot for raw and filtered measurements (6 curves)
        self.xyzplot = DataPlot(
            ("x", "y", "z", "x_kalman", "y_kalman", "z_kalman"), 
            1000
        )

        # Initialize face detector (Haar Cascade)
        cascPathface = os.path.join(
            os.path.dirname(cv2.__file__),
            "data",
            "haarcascade_frontalface_alt2.xml"
        )
        
        if not os.path.exists(cascPathface):
            raise FileNotFoundError(f"Face cascade file not found: {cascPathface}")
            
        self.faceCascade = cv2.CascadeClassifier(cascPathface)
        
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

        # --- NEW FEATURE: Initialize Video Writer variables ---
        self.video_writer = None
        self.output_dir = "Results"
        os.makedirs(self.output_dir, exist_ok=True) # Create Results folder if it doesn't exist
        
        # ADDED: Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(method_name="HaarCascade")

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
                self.xyzplot.createGUI(-1, -1, label="Face Position: Raw vs Kalman Filtered (mm)")

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
        except (OverflowError, ValueError, ZeroDivisionError) as e:
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
        
        # Measurement noise covariance (how much we trust measurements)
        # z-axis has more noise due to estimation method
        R = np.array([
            [10.0, 0.0, 0.0],      # x measurement noise
            [0.0, 10.0, 0.0],      # y measurement noise
            [0.0, 0.0, 50.0]       # z measurement noise (higher)
        ])
        
        # Process noise covariance (how much state can change between frames)
        Q = np.array([
            [1.0, 0.0, 0.0],       # x process noise
            [0.0, 1.0, 0.0],       # y process noise
            [0.0, 0.0, 2.0]        # z process noise
        ])
        
        self.kalman = LinKalman(x0, A, H, R, Q)
        print("Kalman filter initialized")

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

    def largest(self, faces):
        """
        Find the largest face from multiple detections.
        
        Args:
            faces: List of face bounding boxes
            
        Returns:
            tuple: Largest face bounding box (x, y, w, h)
        """
        if len(faces) == 0:
            return None
            
        largest = faces[0]
        for face in faces[1:]:
            if largest[2] * largest[3] < face[2] * face[3]:
                largest = face
        return largest

    def run(self):
        """Main application loop."""
        dpg.create_context()
        dpg.create_viewport(title="Face Tracking with Kalman Filter", width=800, height=600)
        self.createWindow()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Status", True)
        
        frameno = 0
        
        # Initialize Kalman filter
        self.createKalman()

        # --- NEW FEATURE: Define Video Writer ---
        # Note: Saving as .mp4 because standard GIFs are too large for HD webcam video.
        output_file = os.path.join(self.output_dir, 'haar_kalman_result.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        # Warning: VideoWriter requires exact size match with frame
        self.video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (TARGET_WIDTH, TARGET_HEIGHT))
        print(f"Recording video to: {output_file}")
        print("Focus the video window and press 'q' to stop recording and exit.")

        try:
            while dpg.is_dearpygui_running():
                # ADDED: Start frame timing
                self.performance_monitor.start_frame()
                
                # Read frame from camera
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Resize frame to target resolution
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                # Undistort frame using pre-computed maps
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

                # Convert to grayscale and detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # ADDED: Start detection timing
                self.performance_monitor.start_detection()
                
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # ADDED: End detection timing
                self.performance_monitor.end_detection()
                
                # Process largest detected face
                if len(faces) > 0:
                    face = self.largest(faces)
                    if face is not None:
                        (x, y, w, h) = face
                        
                        # Calculate raw 3D position
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
                            frame, text, (int(x), int(y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                        )
                        
                        # Display filtered measurements
                        text_filtered = f"Filtered: x={x_k:.0f} y={y_k:.0f} z={z_k:.0f}mm"
                        cv2.putText(
                            frame, text_filtered, (int(x), int(y - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                        )
                        
                        # ADDED: Record metrics for detected frame
                        self.performance_monitor.end_frame(
                            face_detected=True,
                            position=(xmm, ymm, zmm),
                            filtered_position=(x_k, y_k, z_k)
                        )
                        
                        frameno += 1
                else:
                    # ADDED: Record metrics for frame with no detection
                    self.performance_monitor.end_frame(face_detected=False)
                
                # Add legend to video
                cv2.putText(frame, "Model: Haar Cascade", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Blue: Raw measurement", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, "Green: Kalman prediction", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Red: Kalman filtered", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # --- NEW FEATURE: Save Frame to Video ---
                if self.video_writer is not None:
                    self.video_writer.write(frame)

                # Display frame
                cv2.imshow('Video', frame)
                
                # Allow OpenCV window events to process
                # --- MODIFIED: Ensure 'q' breaks the loop cleanly ---
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stop signal received...")
                    break

                # Render GUI
                dpg.render_dearpygui_frame()
                
                # ADDED: Print stats every 100 frames
                if frameno % 100 == 0 and frameno > 0:
                    stats = self.performance_monitor.get_realtime_stats()
                    print(f"[Frame {frameno}] FPS: {stats.get('avg_fps', 0):.1f}, "
                          f"Detection: {stats.get('detection_rate', 0):.1f}%")
                
        finally:
            # ADDED: Save benchmark results
            self.performance_monitor.save_results()
            self.performance_monitor.print_summary()
            
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
        gui = FaceGui()
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()