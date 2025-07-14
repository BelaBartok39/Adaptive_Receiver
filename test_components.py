#!/usr/bin/env python3
"""
Test script to verify GUI components are working.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_gui_imports():
    """Test that all GUI components can be imported."""
    try:
        from gui import AdaptiveReceiverGUI, PlotManager, ConstellationPlot
        print("✓ Main GUI components imported successfully")
        
        from gui.widgets import StatusPanel, ControlPanel, StatisticsPanel
        print("✓ GUI widgets imported successfully")
        
        from gui.plots import PlotManager
        print("✓ Plot manager imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_config_loader():
    """Test configuration loading."""
    try:
        from config.config_loader import load_detector_config, load_network_config
        
        config = load_detector_config()
        print(f"✓ Detector config loaded: {config.get('model', {}).get('latent_dim', 'N/A')} latent dim")
        
        network_config = load_network_config()
        print(f"✓ Network config loaded: port {network_config.get('udp_port', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading error: {e}")
        return False

def test_detector_import():
    """Test that detector can be imported."""
    try:
        from core.detection.anomaly_detector import AnomalyDetector
        print("✓ AnomalyDetector imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Detector import error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Adaptive RF Receiver components...")
    print("-" * 50)
    
    all_passed = True
    
    # Test imports
    all_passed &= test_config_loader()
    all_passed &= test_detector_import()
    all_passed &= test_gui_imports()
    
    print("-" * 50)
    if all_passed:
        print("✓ All tests passed! The system is ready to use.")
        print("\nTo run the GUI:")
        print("python examples/gui_example.py --gui")
        print("\nTo run command-line version:")
        print("python examples/simple_detection.py")
    else:
        print("✗ Some tests failed. Please check dependencies and file structure.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
