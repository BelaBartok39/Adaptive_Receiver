#!/usr/bin/env python3
"""
Simple GUI test without torch dependencies.
Tests the GUI structure and socket handling.
"""

import sys
import tkinter as tk
from pathlib import Path
import numpy as np
import time
from collections import deque

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

class MockDetector:
    """Mock detector for testing GUI without torch."""
    
    def __init__(self, window_size=1024, device='cpu'):
        self.window_size = window_size
        self.device = device
        self.sample_count = 0
        self.total_detections = 0
        self.is_learning = False
        self.model_dir = "/tmp"
        self.threshold_manager = MockThresholdManager()
    
    def detect(self, i_data, q_data):
        """Mock detection that returns fake results."""
        self.sample_count += len(i_data)
        # Generate fake error based on signal power
        error = np.mean(i_data**2 + q_data**2) + np.random.normal(0, 0.1)
        is_jammed = error > 1.0  # Simple threshold
        if is_jammed:
            self.total_detections += 1
        return is_jammed, float(error), {}
    
    def get_status(self):
        """Get detector status."""
        if self.is_learning:
            return {
                'mode': 'learning',
                'remaining_time': 30.0
            }
        else:
            return {
                'mode': 'detection'
            }
    
    def start_learning_phase(self, duration):
        """Start learning phase."""
        self.is_learning = True
        print(f"Started learning phase for {duration} seconds")
    
    def stop_learning_phase(self):
        """Stop learning phase."""
        self.is_learning = False
        return "Learning phase completed"
    
    def save_model(self):
        """Mock save model."""
        return "/tmp/test_model.pth"
    
    def load_model(self, path):
        """Mock load model."""
        return f"Model loaded from {path}"

class MockThresholdManager:
    """Mock threshold manager."""
    
    def get_threshold(self):
        return 1.0
    
    def get_statistics(self):
        return {'current_threshold': 1.0}

def main():
    """Test the GUI with mock detector."""
    try:
        print("Creating mock detector...")
        detector = MockDetector(window_size=1024, device='cpu')
        
        print("Importing GUI...")
        from gui.main_window import AdaptiveReceiverGUI
        
        print("Creating GUI...")
        gui = AdaptiveReceiverGUI(
            detector_or_wrapper=detector,
            port=12345,
            window_title="Adaptive RF Receiver - Test GUI"
        )
        
        print("Starting GUI...")
        print("This tests the GUI structure without torch dependencies.")
        print("The GUI should open and you can test the interface.")
        print("-" * 60)
        
        gui.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
