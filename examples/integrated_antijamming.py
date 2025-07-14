"""
Example of integrated anomaly detection + DRL anti-jamming system.
Combines your autoencoder-based detection with the paper's channel selection.
"""

import numpy as np
import time
import socket
import struct
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.detection.anomaly_detector import AnomalyDetector
from core.detection.drl_antijamming_agent import (
    DRLAntiJammingAgent,
    IntegratedAntiJammingSystem
)


class SmartAntiJammingSystem:
    """
    Complete anti-jamming system combining:
    1. Your anomaly detection (autoencoder)
    2. Paper's DRL channel selection
    3. Future RF-PUF authentication
    """
    
    def __init__(self, 
                 port: int = 12345,
                 window_size: int = 1024,
                 num_channels: int = 8):
        """
        Initialize the smart anti-jamming system.
        
        Args:
            port: UDP port for receiving data
            window_size: Window size for processing
            num_channels: Number of available channels
        """
        print("Initializing Smart Anti-Jamming System...")
        
        # Your anomaly detector
        self.anomaly_detector = AnomalyDetector(
            window_size=window_size,
            config={
                'model': {'latent_dim': 32},
                'threshold': {'base_percentile': 99.0}
            }
        )
        
        # Paper's DRL agent
        self.drl_agent = DRLAntiJammingAgent(
            num_channels=num_channels,
            channel_switch_cost=0.05  # Optimal from paper
        )
        
        # Try to load pre-trained models
        self._load_models()
        
        # Channel configuration (5GHz UNII-1)
        self.channels = {
            0: 5180, 1: 5200, 2: 5220, 3: 5240,
            4: 5260, 5: 5280, 6: 5300, 7: 5320
        }
        self.current_channel_idx = 0
        
        # Setup UDP receiver
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', port))
        self.socket.settimeout(0.1)
        
        # Metrics tracking
        self.metrics = {
            'total_samples': 0,
            'jamming_detections': 0,
            'channel_switches': 0,
            'successful_transmissions': 0,
            'start_time': time.time()
        }
        
        print(f"System ready on port {port}")
        print(f"Current channel: {self.channels[self.current_channel_idx]} MHz")
    
    def _load_models(self):
        """Load pre-trained models if available."""
        try:
            # Load anomaly detector
            detector_path = "models/anomaly_detector.pth"
            if Path(detector_path).exists():
                self.anomaly_detector.load_model(detector_path)
                print("✓ Loaded anomaly detector model")
            
            # Load DRL agent
            drl_path = "models/antijamming_agent_gamma_0.05.pth"
            if Path(drl_path).exists():
                self.drl_agent.load(drl_path)
                print("✓ Loaded DRL anti-jamming agent")
                
        except Exception as e:
            print(f"Note: Starting with fresh models ({e})")
    
    def scan_all_channels(self) -> np.ndarray:
        """
        Scan all channels to get power levels.
        In real implementation, this would use HackRF.
        """
        # Simulate channel scanning
        powers = np.random.normal(-85, 5, len(self.channels))
        
        # Simulate jamming on some channel
        if np.random.random() < 0.3:  # 30% chance of jamming
            jammed_channel = np.random.randint(0, len(self.channels))
            powers[jammed_channel] = np.random.normal(-40, 3)
        
        return powers
    
    def process_window(self, i_data: np.ndarray, q_data: np.ndarray):
        """
        Process a window of I/Q data with integrated detection and mitigation.
        
        Args:
            i_data: In-phase data
            q_data: Quadrature data
        """
        self.metrics['total_samples'] += 1
        
        # Step 1: Anomaly Detection (Your System)
        is_anomaly, confidence, detection_metrics = self.anomaly_detector.detect(
            i_data, q_data
        )
        
        # Step 2: Multi-Channel Assessment (Paper's Approach)
        channel_powers = self.scan_all_channels()
        
        # Step 3: Decision Making
        if is_anomaly and confidence > 0.8:
            self.metrics['jamming_detections'] += 1
            
            print(f"\n🚨 JAMMING DETECTED!")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Current channel: {self.channels[self.current_channel_idx]} MHz")
            print(f"   Error metrics: {detection_metrics['error']:.4f}")
            
            # Step 4: DRL Channel Selection
            recommended_channel = self.drl_agent.select_action(
                channel_powers, 
                training=False
            )
            
            # Step 5: Execute Channel Switch if Needed
            if recommended_channel != self.current_channel_idx:
                old_channel = self.channels[self.current_channel_idx]
                new_channel = self.channels[recommended_channel]
                
                print(f"   → Switching to channel {new_channel} MHz")
                
                # Send channel switch command (future: to HackRF)
                self.execute_channel_switch(recommended_channel)
                self.metrics['channel_switches'] += 1
                
                # Log the event
                self.log_jamming_event({
                    'timestamp': time.time(),
                    'old_channel': old_channel,
                    'new_channel': new_channel,
                    'confidence': confidence,
                    'channel_powers': dict(zip(self.channels.values(), channel_powers))
                })
            else:
                print(f"   → Staying on current channel (best option)")
        
        else:
            # No jamming detected
            self.metrics['successful_transmissions'] += 1
            
            # Optional: Update DRL agent with successful transmission
            reward = self.drl_agent.calculate_reward(
                channel=self.current_channel_idx,
                jammed=False,
                throughput=1.0
            )
        
        # Step 6: Display periodic statistics
        if self.metrics['total_samples'] % 100 == 0:
            self.display_statistics()
    
    def execute_channel_switch(self, new_channel_idx: int):
        """
        Execute channel switch.
        Future: Send command to HackRF.
        """
        # Simulate CSA (Channel Switch Announcement)
        time.sleep(0.01)  # Small delay for switch
        
        # Update current channel
        self.current_channel_idx = new_channel_idx
        
        # In real implementation:
        # self.hackrf.set_frequency(self.channels[new_channel_idx])
    
    def log_jamming_event(self, event: dict):
        """Log jamming detection and mitigation event."""
        with open('jamming_events.log', 'a') as f:
            f.write(f"{event}\n")
    
    def display_statistics(self):
        """Display system statistics."""
        runtime = time.time() - self.metrics['start_time']
        
        print(f"\n📊 System Statistics (Runtime: {runtime/60:.1f} min)")
        print(f"   Total samples: {self.metrics['total_samples']}")
        print(f"   Jamming detections: {self.metrics['jamming_detections']}")
        print(f"   Channel switches: {self.metrics['channel_switches']}")
        print(f"   Success rate: {self.metrics['successful_transmissions']/self.metrics['total_samples']:.1%}")
        print(f"   Switch rate: {self.metrics['channel_switches']/self.metrics['total_samples']:.1%}")
    
    def run(self):
        """Main processing loop."""
        print("\n🚀 Starting Smart Anti-Jamming System")
        print("=" * 50)
        
        # Start learning phase for anomaly detector
        print("📚 Learning phase (60 seconds)...")
        self.anomaly_detector.start_learning(duration=60)
        
        # Buffers for I/Q data
        i_buffer = []
        q_buffer = []
        last_timestamp = 0
        
        try:
            while True:
                try:
                    # Receive UDP packet
                    data, addr = self.socket.recvfrom(65536)
                    
                    if len(data) < 20:
                        continue
                    
                    # Parse packet
                    timestamp = struct.unpack('!d', data[0:8])[0]
                    samples_in_packet = struct.unpack('!I', data[8:12])[0]
                    
                    # New data block?
                    if timestamp != last_timestamp and last_timestamp != 0:
                        if len(i_buffer) >= self.anomaly_detector.window_size:
                            # Process window
                            i_array = np.array(i_buffer[:self.anomaly_detector.window_size])
                            q_array = np.array(q_buffer[:self.anomaly_detector.window_size])
                            
                            self.process_window(i_array, q_array)
                            
                            # Slide buffer
                            i_buffer = i_buffer[self.anomaly_detector.window_size:]
                            q_buffer = q_buffer[self.anomaly_detector.window_size:]
                    
                    last_timestamp = timestamp
                    
                    # Extract I/Q samples
                    offset = 20
                    for _ in range(samples_in_packet):
                        if offset + 8 > len(data):
                            break
                        i_val = struct.unpack('!f', data[offset:offset+4])[0]
                        q_val = struct.unpack('!f', data[offset+4:offset+8])[0]
                        i_buffer.append(i_val)
                        q_buffer.append(q_val)
                        offset += 8
                    
                except socket.timeout:
                    # Check if learning phase is complete
                    status = self.anomaly_detector.get_status()
                    if status['mode'] == 'learning':
                        print(f"\rLearning: {status['remaining_time']:.1f}s remaining", end='')
                    elif status['mode'] == 'detection' and self.metrics['total_samples'] == 0:
                        print("\n✓ Learning complete! Starting detection mode...")
                        
                except Exception as e:
                    print(f"\nError in receive loop: {e}")
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown."""
        # Save models
        print("💾 Saving models...")
        self.anomaly_detector.save_model("models/anomaly_detector.pth")
        self.drl_agent.save("models/antijamming_agent_latest.pth")
        
        # Final statistics
        self.display_statistics()
        
        # Close socket
        self.socket.close()
        
        print("\n✅ Shutdown complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Smart Anti-Jamming System with Anomaly Detection + DRL'
    )
    parser.add_argument('--port', type=int, default=12345,
                       help='UDP port to listen on')
    parser.add_argument('--channels', type=int, default=8,
                       help='Number of RF channels')
    parser.add_argument('--window', type=int, default=1024,
                       help='Processing window size')
    args = parser.parse_args()
    
    # Create and run system
    system = SmartAntiJammingSystem(
        port=args.port,
        window_size=args.window,
        num_channels=args.channels
    )
    
    try:
        system.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        system.shutdown()


if __name__ == "__main__":
    main()