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
    
    def _on_data_received(self, packet_data: dict):
        """
        Callback for when data is received from network.
        
        Args:
            packet_data: dictionary with timestamp, samples, num_samples
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
                          f"Confidence: {confidence:.2f}, Anomaly Score: {metrics['anomaly_score']:.4f}")
                    
                    # Log detection details
                    self._handle_jamming(confidence, metrics)
                    self.last_detection_time = current_time
            
            # Periodic status update
            if self.windows_processed % 100 == 0:
                status = self.detector.get_status()
                if status['mode'] == 'learning':
                    print(f"\rLearning: {status['remaining_time']:.1f}s remaining, "
                          f"Windows: {self.windows_processed}", end='')
                else:
                    print(f"\rDetection mode - Windows: {self.windows_processed}, "
                          f"Detections: {status['total_detections']}", end='')
            
        except Exception as e:
            print(f"\nProcessing error: {e}")
    
    def _handle_jamming(self, confidence: float, metrics: dict):
        """
        Handle jamming detection.
        
        In the full implementation, this would:
        1. Classify the jammer type using DRN
        2. Scan for clean channels
        3. Send channel change command to HackRF
        
        Args:
            confidence: Detection confidence
            metrics: Detection metrics
        """
        print(f"\nJamming characteristics:")
        print(f"  - Anomaly score: {metrics['anomaly_score']:.4f}")
        print(f"  - Spectral centroid: {metrics['spectral_centroid']:.2f}")
        print(f"  - Spectral bandwidth: {metrics['spectral_bandwidth']:.2f}")
        print(f"  - Frequency offset: {metrics['frequency_offset_normalized']:.4f}")
        print(f"  - Signal power: {metrics['signal_power_dbm']:.1f} dBm")
        
        # TODO: Implement DRN classification
        # jammer_type = self.drn_classifier.classify(metrics)
        # print(f"  - Jammer type: {jammer_type}")
        
        # TODO: Implement channel scanning
        # clean_channel = self.channel_scanner.find_clean_channel()
        # print(f"  - Recommended channel: {clean_channel}")
        
        # TODO: Send channel change command
        # self.send_channel_change(clean_channel)
    
    def start(self):
        """Start the detection system."""
        self.running = True
        
        # Start receiver
        self.receiver.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # Start learning phase
        print("Starting 60-second learning phase...")
        self.detector.start_learning(duration=60.0)
        
        # Main loop for status updates
        try:
            while self.running:
                # Get receiver statistics
                rx_stats = self.receiver.get_statistics()
                
                # Update status display
                status = self.detector.get_status()
                
                if status['mode'] == 'learning':
                    print(f"\rLearning: {status['remaining_time']:.1f}s remaining | "
                          f"Packets: {rx_stats['packets_received']} | "
                          f"Windows: {self.windows_processed}", end='')
                else:
                    print(f"\rDetection mode | "
                          f"Packets: {rx_stats['packets_received']} | "
                          f"Windows: {self.windows_processed} | "
                          f"Detections: {status['total_detections']}", end='')
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop()
    
    def stop(self):
        """Stop the detection system."""
        self.running = False
        
        # Stop receiver
        self.receiver.stop()
        
        # Wait for processing thread
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        # Save model
        model_path = self.detector.save_model()
        print(f"Model saved to: {model_path}")
        
        # Print final statistics
        print("\nFinal Statistics:")
        print(f"  - Windows processed: {self.windows_processed}")
        print(f"  - Total detections: {self.detector.total_detections}")
        rx_stats = self.receiver.get_statistics()
        print(f"  - Packets received: {rx_stats['packets_received']}")
        print(f"  - Bytes received: {rx_stats['bytes_received']}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple RF Jamming Detector using VAE')
    parser.add_argument('--port', type=int, default=None, 
                       help='UDP port to listen on (overrides config)')
    parser.add_argument('--window', type=int, default=None,
                       help='Window size for processing (overrides config)')
    parser.add_argument('--config-dir', type=str, default=None,
                       help='Configuration directory path')
    parser.add_argument('--gui', action='store_true',
                       help='Launch with graphical interface')
    parser.add_argument('--no-gui', action='store_true',
                       help='Force command-line interface only')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pre-trained model to load')
    args = parser.parse_args()
    
    try:
        # Initialize detector with config-based settings
        detector = SimpleJammingDetector(port=args.port, window_size=args.window)
        
        # Load pre-trained model if specified
        if args.model:
            print(f"Loading pre-trained model from {args.model}")
            detector.detector.load_model(args.model)
        
        print("\nStarting RF Jamming Detector with VAE...")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        # Choose interface mode
        if args.gui and not args.no_gui:
            # Launch GUI
            try:
                from gui.main_window import AdaptiveReceiverGUI
                
                print("Launching GUI interface...")
                
                # Create a wrapper that makes SimpleJammingDetector compatible with GUI
                class DetectorWrapper:
                    def __init__(self, simple_detector):
                        self.detector = simple_detector.detector
                        self.port = simple_detector.port
                        self.simple_detector = simple_detector
                        
                        # Make the detector compatible with GUI expectations
                        self.window_size = simple_detector.window_size
                        self.i_buffer = []
                        self.q_buffer = []
                        
                        # Hook into data processing
                        original_process = simple_detector._process_window
                        
                        def hooked_process(i_array, q_array, timestamp):
                            # Store for GUI
                            self.i_buffer = list(i_array)
                            self.q_buffer = list(q_array)
                            
                            # Call original
                            result = original_process(i_array, q_array, timestamp)
                            
                            # Store results for GUI
                            if hasattr(simple_detector.detector, 'last_is_anomaly'):
                                self.detector.last_is_anomaly = simple_detector.detector.last_is_anomaly
                            if hasattr(simple_detector.detector, 'last_error'):
                                self.detector.last_error = simple_detector.detector.last_error
                            
                            return result
                        
                        simple_detector._process_window = hooked_process
                        
                        # Start the detector
                        self.running = True
                        self.receive_thread = threading.Thread(
                            target=self._run_detector, 
                            daemon=True
                        )
                        self.receive_thread.start()
                    
                    def _run_detector(self):
                        """Run the detector in background."""
                        self.simple_detector.start()
                
                # Create wrapper and GUI
                wrapper = DetectorWrapper(detector)
                gui = AdaptiveReceiverGUI(
                    wrapper,
                    window_title="RF Jamming Detector with VAE"
                )
                
                # Start the GUI main loop
                gui.run()
                
            except ImportError as e:
                print(f"GUI not available: {e}")
                print("Falling back to command-line interface...")
                detector.start()
        else:
            # Command-line interface
            detector.start()
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Make sure you're running from the correct directory with config files.")
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'detector' in locals():
            detector.stop()


if __name__ == "__main__":
    main()