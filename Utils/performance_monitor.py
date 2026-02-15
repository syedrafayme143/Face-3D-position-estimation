"""
Runtime Performance Monitor for Face Detection Methods
Professional benchmarking tool that captures real-time metrics during execution

This module can be imported into any face tracking script to collect performance data
without modifying the core functionality.

Usage:
    from Utils.performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor(method_name="MediaPipe")
    
    # In your main loop:
    monitor.start_frame()
    # ... detection code ...
    monitor.end_frame(face_detected=True, position=(x, y, z), 
                     filtered_position=(x_k, y_k, z_k))
    
    # At the end:
    monitor.save_results()
"""

import time
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict


class PerformanceMonitor:
    """
    Real-time performance monitoring for face detection methods.
    Collects metrics during runtime without affecting the tracking process.
    """
    
    def __init__(self, method_name, output_dir="Results/Benchmarks"):
        """
        Initialize performance monitor.
        
        Args:
            method_name: Name of the detection method
            output_dir: Directory to save benchmark results
        """
        self.method_name = method_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Timing metrics
        self.frame_times = []
        self.detection_times = []
        self.kalman_times = []
        
        # Detection metrics
        self.detection_flags = []  # True/False for each frame
        self.detection_gaps = []
        
        # Position data
        self.raw_positions = []     # [(x, y, z), ...]
        self.filtered_positions = []  # [(x, y, z), ...]
        
        # Frame tracking
        self.frame_count = 0
        self.detected_frame_count = 0
        self.current_frame_start = None
        self.current_detection_start = None
        
        # Session info
        self.start_time = time.time()
        self.session_start = datetime.now()
        
        print(f"[{method_name}] Performance Monitor initialized")
    
    def start_frame(self):
        """Mark the start of frame processing."""
        self.current_frame_start = time.time()
    
    def start_detection(self):
        """Mark the start of detection phase."""
        self.current_detection_start = time.time()
    
    def end_detection(self):
        """Mark the end of detection phase."""
        if self.current_detection_start is not None:
            detection_time = (time.time() - self.current_detection_start) * 1000  # ms
            self.detection_times.append(detection_time)
            self.current_detection_start = None
    
    def end_frame(self, face_detected, position=None, filtered_position=None):
        """
        Mark the end of frame processing and record metrics.
        
        Args:
            face_detected: Boolean indicating if face was detected
            position: Tuple (x, y, z) of raw position in mm
            filtered_position: Tuple (x, y, z) of Kalman filtered position in mm
        """
        if self.current_frame_start is not None:
            frame_time = (time.time() - self.current_frame_start) * 1000  # ms
            self.frame_times.append(frame_time)
            self.current_frame_start = None
        
        # Record detection status
        self.detection_flags.append(face_detected)
        self.frame_count += 1
        
        if face_detected:
            self.detected_frame_count += 1
            
            # Record positions
            if position is not None:
                self.raw_positions.append(position)
            if filtered_position is not None:
                self.filtered_positions.append(filtered_position)
    
    def get_realtime_stats(self):
        """
        Get current statistics during runtime.
        
        Returns:
            dict: Current performance metrics
        """
        if len(self.frame_times) == 0:
            return {}
        
        avg_fps = 1000 / np.mean(self.frame_times) if self.frame_times else 0
        detection_rate = (self.detected_frame_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'frames_detected': self.detected_frame_count,
            'detection_rate': detection_rate,
            'avg_fps': avg_fps,
            'avg_frame_time_ms': np.mean(self.frame_times),
            'current_runtime_sec': time.time() - self.start_time
        }
    
    def compute_final_metrics(self):
        """
        Compute comprehensive metrics after session ends.
        
        Returns:
            dict: Complete performance metrics
        """
        metrics = {}
        
        # === 1. DETECTION RELIABILITY ===
        metrics['detection_reliability'] = self._compute_detection_reliability()
        
        # === 2. PROCESSING SPEED ===
        metrics['processing_speed'] = self._compute_processing_speed()
        
        # === 3. POSITION STABILITY ===
        metrics['position_stability'] = self._compute_position_stability()
        
        # === 4. TRACKING SMOOTHNESS ===
        metrics['tracking_smoothness'] = self._compute_tracking_smoothness()
        
        # === 5. KALMAN FILTER EFFECTIVENESS ===
        metrics['kalman_effectiveness'] = self._compute_kalman_effectiveness()
        
        # Session metadata
        metrics['session_info'] = {
            'method_name': self.method_name,
            'start_time': self.session_start.isoformat(),
            'duration_sec': time.time() - self.start_time,
            'total_frames': self.frame_count
        }
        
        return metrics
    
    def _compute_detection_reliability(self):
        """Compute detection reliability metrics."""
        total = len(self.detection_flags)
        detected = sum(self.detection_flags)
        missed = total - detected
        
        # Calculate detection gaps
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, detected in enumerate(self.detection_flags):
            if not detected:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    gaps.append(i - gap_start)
                    in_gap = False
        
        return {
            'total_frames': total,
            'detected_frames': detected,
            'missed_frames': missed,
            'detection_rate_percent': (detected / total * 100) if total > 0 else 0,
            'num_gaps': len(gaps),
            'avg_gap_length': np.mean(gaps) if gaps else 0,
            'max_gap_length': max(gaps) if gaps else 0,
            'min_gap_length': min(gaps) if gaps else 0
        }
    
    def _compute_processing_speed(self):
        """Compute processing speed metrics."""
        if not self.frame_times:
            return {}
        
        frame_times_array = np.array(self.frame_times)
        
        return {
            'avg_fps': 1000 / np.mean(frame_times_array),
            'max_fps': 1000 / np.min(frame_times_array),
            'min_fps': 1000 / np.max(frame_times_array),
            'avg_frame_time_ms': np.mean(frame_times_array),
            'std_frame_time_ms': np.std(frame_times_array),
            'median_frame_time_ms': np.median(frame_times_array),
            'p95_frame_time_ms': np.percentile(frame_times_array, 95),
            'p99_frame_time_ms': np.percentile(frame_times_array, 99)
        }
    
    def _compute_position_stability(self):
        """Compute position stability metrics."""
        if len(self.raw_positions) < 2:
            return {'error': 'Insufficient position data'}
        
        raw_array = np.array(self.raw_positions)
        
        # Calculate standard deviations for each axis
        x_std = np.std(raw_array[:, 0])
        y_std = np.std(raw_array[:, 1])
        z_std = np.std(raw_array[:, 2])
        
        # Overall stability (lower std = more stable)
        overall_std = np.sqrt(x_std**2 + y_std**2 + z_std**2)
        
        # Stability score (0-100, higher is better)
        # Assuming std < 5mm is excellent (score 100), std > 50mm is poor (score 0)
        stability_score = max(0, min(100, 100 - (overall_std / 50 * 100)))
        
        return {
            'x_std_mm': float(x_std),
            'y_std_mm': float(y_std),
            'z_std_mm': float(z_std),
            'overall_std_mm': float(overall_std),
            'stability_score': float(stability_score),
            'x_mean_mm': float(np.mean(raw_array[:, 0])),
            'y_mean_mm': float(np.mean(raw_array[:, 1])),
            'z_mean_mm': float(np.mean(raw_array[:, 2]))
        }
    
    def _compute_tracking_smoothness(self):
        """Compute tracking smoothness (jitter) metrics."""
        if len(self.raw_positions) < 2:
            return {'error': 'Insufficient position data'}
        
        raw_array = np.array(self.raw_positions)
        
        # Calculate frame-to-frame differences (jitter)
        raw_diff = np.diff(raw_array, axis=0)
        raw_jitter = np.sqrt(np.sum(raw_diff**2, axis=1))
        
        # If we have filtered positions, calculate for them too
        filtered_jitter_metrics = {}
        if len(self.filtered_positions) >= 2:
            filtered_array = np.array(self.filtered_positions)
            filtered_diff = np.diff(filtered_array, axis=0)
            filtered_jitter = np.sqrt(np.sum(filtered_diff**2, axis=1))
            
            filtered_jitter_metrics = {
                'filtered_avg_jitter_mm': float(np.mean(filtered_jitter)),
                'filtered_std_jitter_mm': float(np.std(filtered_jitter)),
                'filtered_max_jitter_mm': float(np.max(filtered_jitter))
            }
        
        # Smoothness score (lower jitter = higher score)
        # Assuming jitter < 2mm/frame is excellent, > 20mm/frame is poor
        avg_jitter = np.mean(raw_jitter)
        smoothness_score = max(0, min(100, 100 - (avg_jitter / 20 * 100)))
        
        return {
            'raw_avg_jitter_mm': float(np.mean(raw_jitter)),
            'raw_std_jitter_mm': float(np.std(raw_jitter)),
            'raw_max_jitter_mm': float(np.max(raw_jitter)),
            'raw_min_jitter_mm': float(np.min(raw_jitter)),
            'smoothness_score': float(smoothness_score),
            **filtered_jitter_metrics
        }
    
    def _compute_kalman_effectiveness(self):
        """Compute Kalman filter effectiveness metrics."""
        if len(self.raw_positions) < 2 or len(self.filtered_positions) < 2:
            return {'error': 'Insufficient position data'}
        
        raw_array = np.array(self.raw_positions)
        filtered_array = np.array(self.filtered_positions)
        
        # Ensure same length
        min_len = min(len(raw_array), len(filtered_array))
        raw_array = raw_array[:min_len]
        filtered_array = filtered_array[:min_len]
        
        # Calculate noise reduction
        raw_std = np.std(raw_array, axis=0)
        filtered_std = np.std(filtered_array, axis=0)
        
        noise_reduction = (raw_std - filtered_std) / raw_std * 100
        
        # Calculate jitter reduction
        raw_diff = np.diff(raw_array, axis=0)
        filtered_diff = np.diff(filtered_array, axis=0)
        
        raw_jitter = np.sqrt(np.sum(raw_diff**2, axis=1))
        filtered_jitter = np.sqrt(np.sum(filtered_diff**2, axis=1))
        
        avg_raw_jitter = np.mean(raw_jitter)
        avg_filtered_jitter = np.mean(filtered_jitter)
        
        jitter_reduction = ((avg_raw_jitter - avg_filtered_jitter) / avg_raw_jitter * 100) if avg_raw_jitter > 0 else 0
        
        # Overall effectiveness score
        effectiveness_score = (np.mean(noise_reduction) + jitter_reduction) / 2
        effectiveness_score = max(0, min(100, effectiveness_score))
        
        return {
            'x_noise_reduction_percent': float(noise_reduction[0]),
            'y_noise_reduction_percent': float(noise_reduction[1]),
            'z_noise_reduction_percent': float(noise_reduction[2]),
            'avg_noise_reduction_percent': float(np.mean(noise_reduction)),
            'jitter_reduction_percent': float(jitter_reduction),
            'effectiveness_score': float(effectiveness_score),
            'raw_avg_std_mm': float(np.mean(raw_std)),
            'filtered_avg_std_mm': float(np.mean(filtered_std))
        }
    
    def save_results(self, filename=None):
        """
        Save benchmark results to JSON file.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.method_name}_benchmark_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        metrics = self.compute_final_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n[{self.method_name}] Benchmark results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of the current metrics."""
        stats = self.get_realtime_stats()
        
        print(f"\n{'='*60}")
        print(f"Performance Summary: {self.method_name}")
        print(f"{'='*60}")
        print(f"Frames Processed: {stats.get('frames_processed', 0)}")
        print(f"Detection Rate: {stats.get('detection_rate', 0):.2f}%")
        print(f"Average FPS: {stats.get('avg_fps', 0):.2f}")
        print(f"Runtime: {stats.get('current_runtime_sec', 0):.2f}s")
        print(f"{'='*60}\n")
