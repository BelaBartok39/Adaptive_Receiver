#!/usr/bin/env python3
"""
Command-line RF anomaly detector for testing without GUI.
Shows that the detector works properly.
"""

import sys
import time
import threading
from pathlib import Path
from collections import deque
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.detection.anomaly_detector import AnomalyDetector
from network.receiver import RFDataReceiver
from core.utils.data_buffer import SlidingWindowBuffer
from config.config_loader import load_detector_config, load_network_config, load_signal_config


class CLIDetector:
    """Command-line detector for testing."""
    
    def __init__(self, port: int = None):
        """Initialize the detector."""
        # Load configs
        self.network_config = load_network_config()
        self.signal_config = load_signal_config()
        self.detector_config = load_detector_config()
        
        self.port = port or self.network_config.get('udp_port', 12345)
        self.window_size = self.signal_config.get('window_size', 1024)
        
        print(f"Initializing detector on port {self.port}")
        
        # Initialize detector
        self.detector = AnomalyDetector(
            window_size=self.window_size,
            config=self.detector_config
        )
        print(f"Using device: {self.detector.device}")
        
        # Initialize receiver
        self.receiver = RFDataReceiver(
            port=self.port,
            buffer_size=self.network_config.get('buffer_size', 65536),
            callback=self._on_data_received
        )
        
        # Window buffer
        self.window_buffer = SlidingWindowBuffer(
            window_size=self.window_size,
            stride=self.window_size // 2
        )
        
        # Statistics
        self.windows_processed = 0
        self.detections_count = 0
        self.recent_errors = deque(maxlen=100)
        self.last_print_time = time.time()
        self.running = False
    
    def _on_data_received(self, packet_data):
        """Handle incoming data."""
        samples = packet_data['samples']
        if len(samples) > 0:
            i_data = samples[:, 0]
            q_data = samples[:, 1]
            
            # Get windows
            windows = self.window_buffer.process_samples(i_data, q_data)
            
            # Process each window
            for i_window, q_window in windows:
                self._process_window(i_window, q_window)
    
    def _process_window(self, i_data: np.ndarray, q_data: np.ndarray):
        """Process a window of data."""
        try:
            # Detect anomalies
            is_anomaly, confidence, metrics = self.detector.detect(i_data, q_data)
            
            self.windows_processed += 1
            error = metrics['error']
            self.recent_errors.append(error)
            
            # Get threshold
            threshold = self.detector.threshold_manager.get_threshold()
            
            # Handle detection
            if is_anomaly and not self.detector.is_learning:
                self.detections_count += 1
                print(f"\n{'='*60}")
                print(f"🚨 JAMMING DETECTED! 🚨")
                print(f"Time: {time.strftime('%H:%M:%S')}")
                print(f"Confidence: {confidence:.2%}")
                print(f"Error: {error:.4f} (Threshold: {threshold:.4f})")
                print(f"Total detections: {self.detections_count}")
                print(f"{'='*60}\n")
            
            # Periodic status update
            current_time = time.time()
            if current_time - self.last_print_time > 1.0:
                self._print_status()
                self.last_print_time = current_time
                
        except Exception as e:
            print(f"Error processing window: {e}")
    
    def _print_status(self):
        """Print current status."""
        status = self.detector.get_status()
        mode = status.get('mode', 'unknown')
        
        if mode == 'learning':
            progress = status.get('progress', 0)
            remaining = status.get('remaining_time', 0)
            print(f"\r📚 Learning: {progress:.1f}% | "
                  f"Time left: {remaining:.1f}s | "
                  f"Windows: {self.windows_processed}", end='')
        else:
            threshold = self.detector.threshold_manager.get_threshold()
            if self.recent_errors:
                avg_error = np.mean(self.recent_errors)
                max_error = np.max(self.recent_errors)
            else:
                avg_error = 0
                max_error = 0
            
            print(f"\r🔍 Monitoring | Windows: {self.windows_processed} | "
                  f"Detections: {self.detections_count} | "
                  f"Threshold: {threshold:.4f} | "
                  f"Avg Error: {avg_error:.4f} | "
                  f"Max Error: {max_error:.4f}", end='')
    
    def start(self):
        """Start the detector."""
        self.running = True
        self.receiver.start()
        print("Detector started. Press Ctrl+C to stop.")
    
    def stop(self):
        """Stop the detector."""
        self.running = False
        self.receiver.stop()
        print("\nDetector stopped.")
    
    def run_learning(self, duration: int):
        """Run learning phase."""
        print(f"\n{'='*60}")
        print(f"Starting {duration}-second learning phase...")
        print("Make sure clean signal is being received!")
        print(f"{'='*60}\n")
        
        self.detector.start_learning(duration)
        
        # Wait for learning to complete
        time.sleep(duration)
        
        stats = self.detector.stop_learning()
        threshold_stats = self.detector.threshold_manager.get_statistics()
        
        print(f"\n{'='*60}")
        print("✅ Learning Complete!")
        print(f"Samples processed: {stats['samples_processed']:,}")
        print(f"Final threshold: {stats['final_threshold']:.4f}")
        print(f"Mean error: {threshold_stats.get('mean', 0):.4f}")
        print(f"Std deviation: {threshold_stats.get('std', 0):.4f}")
        print(f"{'='*60}\n")
        print("Now monitoring for jamming...")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLI RF Anomaly Detector")
    parser.add_argument('--port', type=int, help='UDP port')
    parser.add_argument('--learn', type=int, default=30, 
                       help='Learning duration in seconds (default: 30)')
    parser.add_argument('--no-learn', action='store_true',
                       help='Skip learning phase')
    
    args = parser.parse_args()
    
    # Create detector
    detector = CLIDetector(port=args.port)
    
    try:
        # Start detector
        detector.start()
        
        # Run learning phase unless skipped
        if not args.no_learn:
            detector.run_learning(args.learn)
        else:
            print("Skipping learning phase. Using default threshold.")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        detector.stop()
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"  Windows processed: {detector.windows_processed:,}")
        print(f"  Jamming detections: {detector.detections_count}")
        if detector.windows_processed > 0:
            rate = (detector.detections_count / detector.windows_processed) * 100
            print(f"  Detection rate: {rate:.2f}%")


if __name__ == "__main__":
    main()