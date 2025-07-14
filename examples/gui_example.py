#!/usr/bin/env python3
"""
GUI example for the Adaptive RF Receiver.
Demonstrates the complete graphical interface with constellation plots.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.detection.anomaly_detector import AnomalyDetector
from config.config_loader import load_detector_config, load_network_config, load_signal_config
from gui.main_window import AdaptiveReceiverGUI


def main():
    """Main entry point for GUI application."""
    parser = argparse.ArgumentParser(description='Adaptive RF Receiver - GUI Interface')
    parser.add_argument('--port', type=int, default=None, 
                       help='UDP port to listen on (overrides config)')
    parser.add_argument('--window', type=int, default=None,
                       help='Window size for processing (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, overrides config)')
    parser.add_argument('--config-dir', type=str, default=None,
                       help='Configuration directory path')
    args = parser.parse_args()
    
    try:
        # Load configurations
        print("Loading configuration...")
        detector_config = load_detector_config()
        network_config = load_network_config()
        signal_config = load_signal_config()
        
        # Use provided values or fall back to config
        port = args.port if args.port is not None else network_config.get('udp_port', 12345)
        window_size = args.window if args.window is not None else signal_config.get('window_size', 1024)
        device = args.device if args.device is not None else 'auto'
        
        print(f"Configuration loaded:")
        print(f"  - Port: {port}")
        print(f"  - Window size: {window_size}")
        print(f"  - Device: {device}")
        print(f"  - Model latent dim: {detector_config['model']['latent_dim']}")
        
        # Initialize detector
        print("Initializing anomaly detector...")
        detector = AnomalyDetector(
            window_size=window_size,
            device=device,
            config=detector_config
        )
        
        print(f"Detector initialized using device: {detector.device}")
        
        # Create and run GUI
        print("Starting GUI...")
        gui = AdaptiveReceiverGUI(
            detector=detector,
            port=port,
            window_title="Adaptive RF Receiver - Full GUI"
        )
        
        print("\nGUI Instructions:")
        print("1. Click 'Start Detection' to begin monitoring")
        print("2. Click 'Learn Environment' for initial calibration")
        print("3. Watch the constellation plot for signal quality")
        print("4. Monitor error plots for jamming detection")
        print("5. Use Save/Load Model for persistence")
        print("-" * 60)
        
        gui.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
