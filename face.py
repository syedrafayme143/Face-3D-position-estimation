import dearpygui.dearpygui as dpg
import cv2
import os
import pickle
import numpy as np
import math

from Dataplot import DataPlot

# Constants
CALIBRATION_FILE = 'calibration_data.pkl'
FACE_WIDTH_MM = 140  # Average width of a human face in mm
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

class FaceGui:
    def __init__(self):
        """Initialize the Face GUI with camera, calibration, and plotting setup."""
        # Create data plot to hold 1000 points before scrolling
        self.xyzplot = DataPlot(("x", "y", "z"), 1000)

        # Initialize face detector
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
        
        # Create optimal camera matrix and undistortion maps for TARGET resolution
        self.mtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, 
            (TARGET_WIDTH, TARGET_HEIGHT), 
            1, 
            (TARGET_WIDTH, TARGET_HEIGHT)
        )
        
        # Initialize undistortion maps at TARGET resolution
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            mtx, dist, None, self.mtx, 
            (TARGET_WIDTH, TARGET_HEIGHT), 
            cv2.CV_16SC2
        )

    def createWindow(self):
        """Create the DearPyGUI window with data plots."""
        with dpg.window(tag="Status"):
            with dpg.group(horizontal=True):
                self.xyzplot.createGUI(-1, -1, label="Face Position (mm)")

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
        
        # Calculate distance using known face width and angular subtense
        angle_diff = abs(angle2 - angle1)
        if angle_diff < 0.01:  # Avoid division by very small numbers
            return (0, 0, 0)
            
        z = FACE_WIDTH_MM / (2 * math.tan(angle_diff / 2))
        
        # Calculate x, y position of face center
        u_center = (u0_centered + u1_centered) / 2
        v_center = (v0_centered + v1_centered) / 2
        
        x = (z / f0) * u_center
        y = -(z / f1) * v_center  # Negative because image y-axis points down
        
        return (x, y, z)

    def run(self):
        """Main application loop."""
        dpg.create_context()
        dpg.create_viewport(title="Face Tracking System", width=800, height=600)
        self.createWindow()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Status", True)
        
        frameno = 0

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

                # Convert to grayscale and detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Process first detected face
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Calculate 3D position
                    u0, v0 = x, y
                    u1, v1 = x + w, y + h
                    (xmm, ymm, zmm) = self.processFace(self.mtx, u0, v0, u1, v1)
                    
                    # Update plot
                    self.xyzplot.addDataVector(frameno, (xmm, ymm, zmm))
                    
                    # Display position on frame
                    text = f"x={xmm:.0f}mm y={ymm:.0f}mm z={zmm:.0f}mm"
                    cv2.putText(
                        frame, text, (u0, v0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                    
                    frameno += 1
                
                # Display frame
                cv2.imshow('Video', frame)
                
                # Allow OpenCV window events to process
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Render GUI
                dpg.render_dearpygui_frame()
                
        finally:
            # Cleanup
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