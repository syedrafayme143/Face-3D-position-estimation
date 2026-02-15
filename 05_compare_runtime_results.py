"""
Results Aggregator and Professional Comparison Report
Collects benchmark data from all methods and generates comprehensive comparison

Usage:
    python compare_runtime_results.py
"""

import json
import os
import glob
from datetime import datetime
import numpy as np


class BenchmarkComparator:
    """Professional comparison tool for runtime benchmark results."""
    
    def __init__(self, benchmark_dir="Results/Benchmarks"):
        """
        Initialize comparator.
        
        Args:
            benchmark_dir: Directory containing benchmark JSON files
        """
        self.benchmark_dir = benchmark_dir
        self.results = {}
        
    def load_latest_results(self):
        """Load the most recent benchmark file for each method."""
        if not os.path.exists(self.benchmark_dir):
            print(f"❌ Benchmark directory not found: {self.benchmark_dir}")
            return False
        
        # Find all benchmark files
        json_files = glob.glob(os.path.join(self.benchmark_dir, "*_benchmark_*.json"))
        
        if not json_files:
            print(f"❌ No benchmark files found in {self.benchmark_dir}")
            return False
        
        # Group by method name
        method_files = {}
        for filepath in json_files:
            filename = os.path.basename(filepath)
            method_name = filename.split('_benchmark_')[0]
            
            if method_name not in method_files:
                method_files[method_name] = []
            method_files[method_name].append(filepath)
        
        # Load most recent file for each method
        for method_name, files in method_files.items():
            # Sort by modification time, get most recent
            latest_file = max(files, key=os.path.getmtime)
            
            try:
                with open(latest_file, 'r') as f:
                    self.results[method_name] = json.load(f)
                print(f"✓ Loaded: {method_name} from {os.path.basename(latest_file)}")
            except Exception as e:
                print(f"❌ Error loading {latest_file}: {e}")
        
        return len(self.results) > 0
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        if not self.results:
            print("❌ No results to compare")
            return
        
        print("\n" + "="*90)
        print("PROFESSIONAL RUNTIME BENCHMARK COMPARISON REPORT")
        print("="*90)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Methods Compared: {len(self.results)}")
        print("="*90)
        
        # Metric 1: Detection Reliability
        self._print_detection_reliability()
        
        # Metric 2: Processing Speed
        self._print_processing_speed()
        
        # Metric 3: Position Stability
        self._print_position_stability()
        
        # Metric 4: Tracking Smoothness
        self._print_tracking_smoothness()
        
        # Metric 5: Kalman Filter Effectiveness
        self._print_kalman_effectiveness()
        
        # Overall Rankings
        self._print_rankings()
        
        # Save combined report
        self._save_comparison_report()
    
    def _print_detection_reliability(self):
        """Print detection reliability comparison."""
        print("\n" + "┌" + "─"*88 + "┐")
        print("│ 📊 METRIC 1: DETECTION RELIABILITY (Higher = Better)" + " "*34 + "│")
        print("├" + "─"*88 + "┤")
        print(f"│ {'Method':<20} {'Rate %':>10} {'Detected':>10} {'Missed':>8} {'Gaps':>6} {'Avg Gap':>9} {'Max Gap':>8} │")
        print("├" + "─"*88 + "┤")
        
        for method in sorted(self.results.keys()):
            dr = self.results[method].get('detection_reliability', {})
            if 'error' in dr:
                continue
            
            print(f"│ {method:<20} "
                  f"{dr.get('detection_rate_percent', 0):>9.2f}% "
                  f"{dr.get('detected_frames', 0):>9} "
                  f"{dr.get('missed_frames', 0):>7} "
                  f"{dr.get('num_gaps', 0):>5} "
                  f"{dr.get('avg_gap_length', 0):>8.1f} "
                  f"{dr.get('max_gap_length', 0):>7.0f} │")
        
        print("└" + "─"*88 + "┘")
    
    def _print_processing_speed(self):
        """Print processing speed comparison."""
        print("\n" + "┌" + "─"*88 + "┐")
        print("│ ⚡ METRIC 2: PROCESSING SPEED (Higher = Better)" + " "*40 + "│")
        print("├" + "─"*88 + "┤")
        print(f"│ {'Method':<20} {'Avg FPS':>10} {'Min FPS':>10} {'Max FPS':>10} {'Avg Time':>12} {'P95 Time':>10} │")
        print("├" + "─"*88 + "┤")
        
        for method in sorted(self.results.keys()):
            ps = self.results[method].get('processing_speed', {})
            if 'error' in ps:
                continue
            
            print(f"│ {method:<20} "
                  f"{ps.get('avg_fps', 0):>9.2f} "
                  f"{ps.get('min_fps', 0):>9.2f} "
                  f"{ps.get('max_fps', 0):>9.2f} "
                  f"{ps.get('avg_frame_time_ms', 0):>9.2f} ms "
                  f"{ps.get('p95_frame_time_ms', 0):>7.2f} ms │")
        
        print("└" + "─"*88 + "┘")
    
    def _print_position_stability(self):
        """Print position stability comparison."""
        print("\n" + "┌" + "─"*88 + "┐")
        print("│ 🎯 METRIC 3: POSITION STABILITY (Lower STD = More Stable)" + " "*28 + "│")
        print("├" + "─"*88 + "┤")
        print(f"│ {'Method':<20} {'Score':>8} {'X STD':>10} {'Y STD':>10} {'Z STD':>10} {'Overall':>12} │")
        print("├" + "─"*88 + "┤")
        
        for method in sorted(self.results.keys()):
            stab = self.results[method].get('position_stability', {})
            if 'error' in stab:
                continue
            
            print(f"│ {method:<20} "
                  f"{stab.get('stability_score', 0):>7.1f}% "
                  f"{stab.get('x_std_mm', 0):>8.2f} mm "
                  f"{stab.get('y_std_mm', 0):>8.2f} mm "
                  f"{stab.get('z_std_mm', 0):>8.2f} mm "
                  f"{stab.get('overall_std_mm', 0):>9.2f} mm │")
        
        print("└" + "─"*88 + "┘")
    
    def _print_tracking_smoothness(self):
        """Print tracking smoothness comparison."""
        print("\n" + "┌" + "─"*88 + "┐")
        print("│ 📈 METRIC 4: TRACKING SMOOTHNESS (Lower Jitter = Smoother)" + " "*28 + "│")
        print("├" + "─"*88 + "┤")
        print(f"│ {'Method':<20} {'Score':>8} {'Raw Jitter':>14} {'Filtered':>14} {'Reduction':>12} │")
        print("├" + "─"*88 + "┤")
        
        for method in sorted(self.results.keys()):
            smooth = self.results[method].get('tracking_smoothness', {})
            if 'error' in smooth:
                continue
            
            raw_jitter = smooth.get('raw_avg_jitter_mm', 0)
            filt_jitter = smooth.get('filtered_avg_jitter_mm', 0)
            reduction = ((raw_jitter - filt_jitter) / raw_jitter * 100) if raw_jitter > 0 and filt_jitter > 0 else 0
            
            print(f"│ {method:<20} "
                  f"{smooth.get('smoothness_score', 0):>7.1f}% "
                  f"{raw_jitter:>11.2f} mm "
                  f"{filt_jitter if filt_jitter > 0 else 0:>11.2f} mm "
                  f"{reduction:>10.1f}% │")
        
        print("└" + "─"*88 + "┘")
    
    def _print_kalman_effectiveness(self):
        """Print Kalman filter effectiveness comparison."""
        print("\n" + "┌" + "─"*88 + "┐")
        print("│ 🔧 METRIC 5: KALMAN FILTER EFFECTIVENESS (Higher = Better)" + " "*27 + "│")
        print("├" + "─"*88 + "┤")
        print(f"│ {'Method':<20} {'Score':>8} {'Noise Red.':>14} {'Jitter Red.':>15} {'Overall':>12} │")
        print("├" + "─"*88 + "┤")
        
        for method in sorted(self.results.keys()):
            kalman = self.results[method].get('kalman_effectiveness', {})
            if 'error' in kalman:
                continue
            
            print(f"│ {method:<20} "
                  f"{kalman.get('effectiveness_score', 0):>7.1f}% "
                  f"{kalman.get('avg_noise_reduction_percent', 0):>11.1f}% "
                  f"{kalman.get('jitter_reduction_percent', 0):>12.1f}% "
                  f"{kalman.get('effectiveness_score', 0):>10.1f}% │")
        
        print("└" + "─"*88 + "┘")
    
    def _print_rankings(self):
        """Print overall rankings."""
        print("\n" + "="*90)
        print("🏆 OVERALL RANKINGS")
        print("="*90)
        
        rankings = {
            'Detection Reliability': lambda x: x.get('detection_reliability', {}).get('detection_rate_percent', 0),
            'Processing Speed (FPS)': lambda x: x.get('processing_speed', {}).get('avg_fps', 0),
            'Position Stability': lambda x: x.get('position_stability', {}).get('stability_score', 0),
            'Tracking Smoothness': lambda x: x.get('tracking_smoothness', {}).get('smoothness_score', 0),
            'Kalman Effectiveness': lambda x: x.get('kalman_effectiveness', {}).get('effectiveness_score', 0)
        }
        
        for metric_name, key_func in rankings.items():
            print(f"\n{metric_name}:")
            sorted_methods = sorted(
                self.results.items(),
                key=lambda x: key_func(x[1]),
                reverse=True
            )
            for rank, (method, data) in enumerate(sorted_methods, 1):
                value = key_func(data)
                unit = " FPS" if "FPS" in metric_name else "%"
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                print(f"  {medal} {rank}. {method:<25} {value:>8.2f}{unit}")
        
        # Composite Score
        print("\n" + "="*90)
        print("🏆 COMPOSITE PERFORMANCE SCORE (Weighted Average)")
        print("="*90)
        
        composite_scores = {}
        for method, data in self.results.items():
            score = (
                data.get('detection_reliability', {}).get('detection_rate_percent', 0) * 0.25 +
                min(data.get('processing_speed', {}).get('avg_fps', 0) / 30 * 100, 100) * 0.20 +
                data.get('position_stability', {}).get('stability_score', 0) * 0.20 +
                data.get('tracking_smoothness', {}).get('smoothness_score', 0) * 0.20 +
                data.get('kalman_effectiveness', {}).get('effectiveness_score', 0) * 0.15
            )
            composite_scores[method] = score
        
        ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nWeights: Detection(25%), Speed(20%), Stability(20%), Smoothness(20%), Kalman(15%)\n")
        for rank, (method, score) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank}. {method:<25} {score:>8.2f}/100")
        
        # Recommendations
        print("\n" + "="*90)
        print("✨ RECOMMENDATIONS")
        print("="*90)
        
        best_overall = ranked[0][0]
        best_fps = max(self.results.items(), 
                      key=lambda x: x[1].get('processing_speed', {}).get('avg_fps', 0))[0]
        best_detection = max(self.results.items(),
                           key=lambda x: x[1].get('detection_reliability', {}).get('detection_rate_percent', 0))[0]
        best_stability = max(self.results.items(),
                           key=lambda x: x[1].get('position_stability', {}).get('stability_score', 0))[0]
        
        print(f"\n  ✓ Best Overall Performance: {best_overall}")
        print(f"  ✓ Fastest Processing: {best_fps}")
        print(f"  ✓ Most Reliable Detection: {best_detection}")
        print(f"  ✓ Most Stable Tracking: {best_stability}")
        
        print("\nUse Case Recommendations:")
        print("  • Real-time Applications: Choose the method with highest FPS")
        print("  • High Accuracy Needs: Choose the method with best detection rate")
        print("  • Precision Measurements: Choose the method with lowest position STD")
        print("  • Smooth Tracking: Choose the method with lowest jitter")
        
        print("\n" + "="*90 + "\n")
    
    def _save_comparison_report(self):
        """Save comparison report to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.benchmark_dir, f"comparison_report_{timestamp}.json")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'methods_compared': list(self.results.keys()),
            'detailed_results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Comparison report saved to: {filepath}")


def main():
    """Main entry point."""
    print("\n" + "="*90)
    print("LOADING RUNTIME BENCHMARK RESULTS")
    print("="*90 + "\n")
    
    comparator = BenchmarkComparator()
    
    if comparator.load_latest_results():
        comparator.generate_comparison_report()
    else:
        print("\n⚠️  No benchmark results found!")
        print("Please run the face tracking methods with performance monitoring enabled first.\n")


if __name__ == "__main__":
    main()
