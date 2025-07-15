"""
Simple example showing how to use the modular RF anomaly detection system.
"""

import sys
import numpy as np
import threading
import time
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.detection.anomaly_detector import AnomalyDetector
from core.preprocessing.signal_filters import SignalPreprocessor
from network.receiver import RFDataReceiver
from core.utils.data_buffer import SlidingWindowBuffer
from config.config_loader import load_detector_config, load_network_config, load_signal_config

# Import GUI class
from gui.main_window import AdaptiveReceiverGUI


class SimpleJammingDetector:
    """
    Simplified jamming detector using the modular components.
    """
    
    def __init__(self, port: int = None, window_size: int = None):
        """
        Initialize the detector.
        
        Args:
            port: UDP port to listen on (overrides config)
            window_size: Size of processing windows (overrides config)
        """
        # Load configurations
        self.network_config = load_network_config()
        self.signal_config = load_signal_config()
        self.detector_config = load_detector_config()
        
        # Use provided values or fall back to config
        self.port = port if port is not None else self.network_config.get('udp_port', 12345)
        self.window_size = window_size if window_size is not None else self.signal_config.get('window_size', 1024)
        self.running = False
        
        print(f"Loading configuration:")
        print(f"  - Port: {self.port}")
        print(f"  - Window size: {self.window_size}")
        print(f"  - Model latent dim: {self.detector_config['model']['latent_dim']}")
        print(f"  - Model beta (VAE): {self.detector_config['model'].get('beta', 1.0)}")
        print(f"  - Learning rate: {self.detector_config['training']['learning_rate']}")
        
        # Initialize detector with VAE
        self.detector = AnomalyDetector(
            window_size=self.window_size,
            config=self.detector_config
        )
        
        # Initialize receiver with callback
        self.receiver = RFDataReceiver(
            port=self.port,
            buffer_size=self.network_config.get('buffer_size', 65536),
            callback=self._on_data_received
        )
        
        # Initialize sliding window buffer
        self.window_buffer = SlidingWindowBuffer(
            window_size=self.window_size,
            stride=self.window_size // 2  # 50% overlap
        )
        
        # Processing thread
        self.process_thread = None
        self.process_queue = []
        self.process_lock = threading.Lock()
        
        # Statistics
        self.windows_processed = 0
        self.last_detection_time = 0
        
        print(f"Detector initialized on port {self.port}")
        print(f"Using device: {self.detector.device}")
        print(f"Buffer size: {self.network_config.get('buffer_size', 65536)}")
        print(f"Preprocessing: {self.detector_config['preprocessing']['normalization']} normalization")
    
    def _on_data_received(self, packet_data: Dict):
        """
        Callback for when data is received from network.
        
        Args:
            packet_data: Dictionary with timestamp, samples, num_samples
        """
        samples = packet_data['samples']
        if len(samples) > 0:
            i_data = samples[:, 0]
            q_data = samples[:, 1]
            
            # Add to processing queue
            with self.process_lock:
                self.process_queue.append((i_data, q_data, packet_data['timestamp']))
    
    def _process_loop(self):
        """Main processing loop running in separate thread."""
        while self.running:
            # Get data from queue
            data_to_process = []
            with self.process_lock:
                if self.process_queue:
                    data_to_process = self.process_queue.copy()
                    self.process_queue.clear()
            
            # Process each packet
            for i_data, q_data, timestamp in data_to_process:
                # Get windows from sliding window buffer
                windows = self.window_buffer.process_samples(i_data, q_data)
                
                # Process each window
                for i_window, q_window in windows:
                    self._process_window(i_window, q_window, timestamp)
            
            # Small sleep to prevent CPU spinning
            if not data_to_process:
                time.sleep(0.001)
    
    def _process_window(self, i_array: np.ndarray, q_array: np.ndarray, timestamp: float):
        """Process a window of samples using VAE detector."""
        try:
            # Detect anomalies
            is_anomaly, confidence, metrics = self.detector.detect(i_array, q_array)
            
            self.windows_processed += 1
            
            # Log detections
            if is_anomaly and not self.detector.is_learning:
                current_time = time.time()
                # Avoid duplicate detections within 1 second
                if current_time - self.last_detection_time > 1.0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] JAMMING DETECTED! "
                          f"Confidence: {confidence:.2f}, Anomaly Score: {metrics['error']:.4f}")
                    self.last_detection_time = current_time
            
            # Provide status for command-line interface
            if self.windows_processed % 20 == 0:
                status = self.detector.get_status()
                mode = status.get('mode', 'N/A')
                if mode == 'learning':
                    progress = status.get('progress', 0)
                    print(f"\rLearning... {progress:.1f}% complete", end="")
                else:
                    threshold = self.detector.threshold_manager.get_threshold()
                    print(f"\rMonitoring... Windows: {self.windows_processed}, "
                          f"Threshold: {threshold:.4f}", end="")
        
        except Exception as e:
            print(f"\nError in processing window: {e}")
            import traceback
            traceback.print_exc()

    def start(self):
        """Start the detector."""
        if not self.running:
            self.running = True
            self.receiver.start()
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.process_thread.start()
            print("Detector started.")

    def stop(self):
        """Stop the detector."""
        if self.running:
            self.running = False
            self.receiver.stop()
            if self.process_thread:
                self.process_thread.join(timeout=2)
            print("\nDetector stopped.")

    def _handle_jamming(self, metrics: Dict):
        """
        Placeholder for advanced jamming response.
        
        Args:
            metrics: Dictionary of metrics from the detection.
        """
        # TODO: Implement jammer classification based on metrics
        # TODO: Implement channel scanning/hopping logic
        pass


def main():
    """Main function to run the detector."""
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive RF Jamming Detector")
    parser.add_argument('--gui', action='store_true', help="Run with GUI")
    parser.add_argument('--port', type=int, help="Override UDP port")
    parser.add_argument('--learn', type=int, metavar='SECS', help="Run learning mode for SECS seconds, then exit")
    args = parser.parse_args()

    # Initialize the main detector
    detector = SimpleJammingDetector(port=args.port)

    if args.gui:
        # GUI mode
        print("Starting GUI mode...")
        # The GUI will take control of the detector
        gui = AdaptiveReceiverGUI(detector)
        # The GUI's run() method starts the Tkinter mainloop
        gui.run()
        # When the GUI is closed, the detector's stop() method is called by the GUI's on_closing handler
        
    else:
        # Command-line mode
        detector.start()
        
        if args.learn:
            print(f"Starting learning phase for {args.learn} seconds...")
            detector.detector.start_learning(args.learn)
            try:
                time.sleep(args.learn)
            except KeyboardInterrupt:
                print("\nLearning interrupted.")
            finally:
                stats = detector.detector.stop_learning()
                print("\nLearning complete.")
                print(f"  - Samples processed: {stats['samples_processed']}")
                print(f"  - Final threshold: {stats['final_threshold']:.4f}")
                detector.stop()
        else:
            print("Running in command-line mode. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
            finally:
                detector.stop()

if __name__ == "__main__":
    main()