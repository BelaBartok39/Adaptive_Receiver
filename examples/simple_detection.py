"""
Simple example showing how to use the modular RF anomaly detection system.
"""

import sys
import numpy as np
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.detection.anomaly_detector import AnomalyDetector
from core.preprocessing.signal_filters import SignalPreprocessor
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
        print(f"  - Learning rate: {self.detector_config['training']['learning_rate']}")
        
        self.detector = AnomalyDetector(
            window_size=self.window_size,
            config=self.detector_config
        )
        
        # Setup UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.settimeout(self.network_config.get('timeout', 0.1))
        
        # Data buffers
        self.i_buffer = []
        self.q_buffer = []
        self.last_timestamp = 0
        
        print(f"Detector initialized on port {self.port}")
        print(f"Using device: {self.detector.device}")
        print(f"Buffer size: {self.network_config.get('buffer_size', 65536)}")
        print(f"Preprocessing: {self.detector_config['preprocessing']['normalization']} normalization")
    
    def start(self):
        """Start the detection system."""
        self.running = True
        
        # Start receiver thread
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        
        # Start learning phase
        print("Starting 60-second learning phase...")
        self.detector.start_learning(duration=60.0)
        
        # Main loop
        try:
            while self.running:
                status = self.detector.get_status()
                
                if status['mode'] == 'learning':
                    print(f"\rLearning: {status['remaining_time']:.1f}s remaining", end='')
                else:
                    print(f"\rDetection mode - Samples: {status['samples_processed']}, "
                          f"Detections: {status['total_detections']}", end='')
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop()
    
    def stop(self):
        """Stop the detection system."""
        self.running = False
        if hasattr(self, 'receive_thread'):
            self.receive_thread.join()
        self.socket.close()
        
        # Save model
        model_path = self.detector.save_model()
        print(f"Model saved to: {model_path}")
    
    def _receive_loop(self):
        """Receive and process UDP packets."""
        buffer_size = self.network_config.get('buffer_size', 65536)
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(buffer_size)
                
                if len(data) < 20:
                    continue
                
                # Parse packet header
                timestamp = struct.unpack('!d', data[0:8])[0]
                samples_in_packet = struct.unpack('!I', data[8:12])[0]
                
                # Check for new data block
                if timestamp != self.last_timestamp and self.last_timestamp != 0:
                    if len(self.i_buffer) >= self.window_size:
                        self._process_window()
                    self.i_buffer = []
                    self.q_buffer = []
                
                self.last_timestamp = timestamp
                
                # Extract I/Q samples
                offset = 20
                for _ in range(samples_in_packet):
                    if offset + 8 > len(data):
                        break
                    i_val = struct.unpack('!f', data[offset:offset+4])[0]
                    q_val = struct.unpack('!f', data[offset+4:offset+8])[0]
                    self.i_buffer.append(i_val)
                    self.q_buffer.append(q_val)
                    offset += 8
                
                # Process if we have enough samples
                if len(self.i_buffer) >= self.window_size:
                    self._process_window()
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"\nReceive error: {e}")
    
    def _process_window(self):
        """Process a window of samples."""
        try:
            # Get window
            i_array = np.array(self.i_buffer[:self.window_size])
            q_array = np.array(self.q_buffer[:self.window_size])
            
            # Detect anomalies
            is_anomaly, confidence, metrics = self.detector.detect(i_array, q_array)
            
            # Log detections
            if is_anomaly and not self.detector.is_learning:
                print(f"\n[{time.strftime('%H:%M:%S')}] JAMMING DETECTED! "
                      f"Confidence: {confidence:.2f}, Error: {metrics['error']:.4f}")
                
                # Here you would trigger the DRN classifier and channel change
                # For now, just log the detection
                self._handle_jamming(confidence, metrics)
            
            # Slide buffer
            self.i_buffer = self.i_buffer[self.window_size:]
            self.q_buffer = self.q_buffer[self.window_size:]
            
        except Exception as e:
            print(f"\nProcessing error: {e}")
    
    def _handle_jamming(self, confidence: float, metrics: Dict):
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
        # Placeholder for future implementation
        print(f"\nJamming characteristics:")
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


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple RF Jamming Detector')
    parser.add_argument('--port', type=int, default=None, 
                       help='UDP port to listen on (overrides config)')
    parser.add_argument('--window', type=int, default=None,
                       help='Window size for processing (overrides config)')
    parser.add_argument('--config-dir', type=str, default=None,
                       help='Configuration directory path')
    args = parser.parse_args()
    
    try:
        # Initialize detector with config-based settings
        detector = SimpleJammingDetector(port=args.port, window_size=args.window)
        
        print("\nStarting RF Jamming Detector...")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        detector.start()
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Make sure you're running from the correct directory with config files.")
    except Exception as e:
        print(f"Error: {e}")
        if 'detector' in locals():
            detector.stop()


if __name__ == "__main__":
    main()