#!/usr/bin/env python3
"""
Test the GUI integration with SimpleJammingDetector.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_gui_integration():
    """Test that GUI can work with SimpleJammingDetector."""
    try:
        # Test imports
        from examples.simple_detection import SimpleJammingDetector
        from gui.main_window import AdaptiveReceiverGUI
        from config.config_loader import load_detector_config
        
        print("✓ All imports successful")
        
        # Test detector creation
        detector = SimpleJammingDetector(port=12346, window_size=512)  # Use different port
        print("✓ SimpleJammingDetector created")
        
        # Test GUI creation with SimpleJammingDetector
        gui = AdaptiveReceiverGUI(detector, window_title="Test GUI")
        print("✓ GUI created with SimpleJammingDetector")
        
        # Test that GUI detected it's using external networking
        if gui.use_external_socket:
            print("✓ GUI correctly detected SimpleJammingDetector networking")
        else:
            print("✗ GUI should use external networking with SimpleJammingDetector")
            return False
        
        # Test statistics method
        stats = gui.get_statistics()
        print(f"✓ Statistics method works: {stats.get('status', 'Unknown')}")
        
        # Cleanup
        try:
            detector.socket.close()
        except:
            pass
        
        gui.root.destroy()
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration test."""
    print("Testing GUI integration with SimpleJammingDetector...")
    print("-" * 60)
    
    if test_gui_integration():
        print("-" * 60)
        print("✓ GUI integration test passed!")
        print("\nTo run with GUI:")
        print("python examples/simple_detection.py --gui")
        return 0
    else:
        print("-" * 60)
        print("✗ GUI integration test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
