#!/usr/bin/env python3
"""
Simple test to verify the plotting system works.
"""

import sys
import time
import numpy as np
from pathlib import Path
from collections import deque

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import tkinter as tk
from gui.plots import PlotManager


def test_plots():
    """Test the plotting system with dummy data."""
    # Create root window
    root = tk.Tk()
    root.title("Plot Test")
    root.geometry("1200x800")
    
    # Create plot data
    plot_data = {
        'time': deque(maxlen=500),
        'error': deque(maxlen=500),
        'threshold': deque(maxlen=500),
        'detections': deque(maxlen=500),
        'i_constellation': deque(maxlen=200),
        'q_constellation': deque(maxlen=200),
        'spec_freqs': None,
        'spec_psd': None
    }
    
    # Create plot manager
    plot_manager = PlotManager(root, plot_data)
    plot_manager.pack(fill=tk.BOTH, expand=True)
    
    # Function to add test data
    def add_test_data():
        # Add some test data
        current_time = len(plot_data['time'])
        
        # Add time series data
        plot_data['time'].append(current_time)
        error = 0.01 + 0.005 * np.sin(current_time * 0.1) + np.random.normal(0, 0.001)
        plot_data['error'].append(error)
        plot_data['threshold'].append(0.02)  # Fixed threshold
        plot_data['detections'].append(error > 0.02)
        
        # Add constellation data
        if current_time % 5 == 0:
            # Generate some I/Q data
            n_points = 20
            i_data = np.random.normal(0, 1, n_points)
            q_data = np.random.normal(0, 1, n_points)
            plot_data['i_constellation'].extend(i_data)
            plot_data['q_constellation'].extend(q_data)
        
        # Add spectral data
        if current_time % 10 == 0:
            freqs = np.linspace(0, 1, 128)
            psd = -30 - 20 * np.log10(freqs + 0.1) + np.random.normal(0, 2, 128)
            plot_data['spec_freqs'] = freqs
            plot_data['spec_psd'] = 10**(psd/10)  # Convert from dB
        
        # Print status
        if current_time % 20 == 0:
            print(f"Added data point {current_time}")
            print(f"  - Error: {error:.4f}")
            print(f"  - Plot data sizes: time={len(plot_data['time'])}, "
                  f"error={len(plot_data['error'])}, "
                  f"constellation={len(plot_data['i_constellation'])}")
        
        # Schedule next update
        if current_time < 200:  # Add 200 data points
            root.after(50, add_test_data)
        else:
            print("\nTest complete! You should see:")
            print("1. Error plot with sine wave + noise")
            print("2. Detection events when error > threshold")
            print("3. I/Q constellation scatter plot")
            print("4. Power spectral density plot")
    
    # Start animation
    plot_manager.start_animation()
    
    # Start adding data
    root.after(100, add_test_data)
    
    # Run main loop
    root.mainloop()


if __name__ == "__main__":
    print("Testing plot system...")
    print("This will add 200 data points over 10 seconds")
    test_plots()