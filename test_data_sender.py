#!/usr/bin/env python3
"""
Simple test data sender to verify GUI data reception.
"""

import socket
import struct
import time
import numpy as np
import argparse


def send_test_data(host='localhost', port=12345, duration=30):
    """Send test I/Q data to the receiver."""
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"Sending test data to {host}:{port} for {duration} seconds...")
    
    start_time = time.time()
    packet_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Generate test I/Q data (1024 samples per packet)
            num_samples = 1024
            t = np.linspace(0, 1, num_samples)
            
            # Create a complex signal with multiple frequency components
            freq1 = 10.0  # 10 Hz
            freq2 = 25.0  # 25 Hz
            noise_level = 0.1
            
            # Generate complex signal
            signal1 = np.exp(1j * 2 * np.pi * freq1 * t)
            signal2 = 0.3 * np.exp(1j * 2 * np.pi * freq2 * t) 
            noise = noise_level * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
            
            complex_signal = signal1 + signal2 + noise
            
            # Extract I and Q components
            i_data = np.real(complex_signal).astype(np.float32)
            q_data = np.imag(complex_signal).astype(np.float32)
            
            # Build packet
            timestamp = time.time()
            packet = struct.pack('!d', timestamp)  # 8 bytes timestamp
            packet += struct.pack('!I', num_samples)  # 4 bytes num_samples
            packet += b'\x00' * 8  # 8 bytes reserved
            
            # Add I/Q samples
            for i, q in zip(i_data, q_data):
                packet += struct.pack('!ff', i, q)
            
            # Send packet
            sock.sendto(packet, (host, port))
            packet_count += 1
            
            # Send at approximately 10 packets per second
            time.sleep(0.1)
            
            if packet_count % 50 == 0:
                print(f"Sent {packet_count} packets...")
    
    except KeyboardInterrupt:
        print("\nStopping data transmission...")
    
    finally:
        sock.close()
        print(f"Sent {packet_count} packets total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send test RF data")
    parser.add_argument('--host', default='localhost', help='Target host')
    parser.add_argument('--port', type=int, default=12345, help='Target port')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    
    args = parser.parse_args()
    send_test_data(args.host, args.port, args.duration)
