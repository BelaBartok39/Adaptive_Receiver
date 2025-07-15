#!/usr/bin/env python3
"""
Direct launcher for the Adaptive RF Receiver GUI.
No wrapper - just clean, direct integration.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from gui.main_window import AdaptiveReceiverGUI


def main():
    """Launch the GUI directly."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Adaptive RF Receiver - Direct GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_gui.py
  
  # Run on different port
  python run_gui.py --port 5000
  
  # Run with larger window size
  python run_gui.py --window-size 2048
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=12345,
        help='UDP port to listen on (default: 12345)'
    )
    
    parser.add_argument(
        '--window-size', 
        type=int, 
        default=1024,
        help='Processing window size (default: 1024)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Adaptive RF Receiver - Jamming Detection System")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Window Size: {args.window_size}")
    print("=" * 60)
    
    # Create and run GUI
    gui = AdaptiveReceiverGUI(port=args.port, window_size=args.window_size)
    gui.run()


if __name__ == "__main__":
    main()