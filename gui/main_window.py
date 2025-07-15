"""
Main GUI application for the Adaptive RF Receiver.
Provides real-time visualization and control interface.
"""

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import tkinter.filedialog
import socket
import struct
import threading
import time
import datetime
from scipy import signal
import numpy as np
from collections import deque
from typing import Optional, Union, Dict

from .plots import PlotManager
from .widgets import StatusPanel, ControlPanel, StatisticsPanel


class AdaptiveReceiverGUI:
    """Main GUI application for the Adaptive RF Receiver."""
    
    def __init__(self, detector_instance, window_title: str = "Adaptive RF Receiver"):
        """
        Initialize the GUI.
        
        Args:
            detector_instance: An instance of SimpleJammingDetector.
            window_title: Title for the main window.
        """
        # The GUI now expects a fully-initialized SimpleJammingDetector
        self.simple_detector = detector_instance
        self.detector = self.simple_detector.detector
        self.port = self.simple_detector.port
        
        self.running = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Networking is now handled entirely by SimpleJammingDetector
        
        # Data buffers for plots
        self.plot_data = {
            'time': deque(maxlen=500),
            'error': deque(maxlen=500),
            'threshold': deque(maxlen=500),
            'detections': deque(maxlen=500),
            # Reduce constellation points for performance
            'i_constellation': deque(maxlen=200),
            'q_constellation': deque(maxlen=200),
            # Spectral data
            'spec_freqs': None,
            'spec_psd': None
        }
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Setup GUI components
        self.setup_gui()
        
        # Hook into SimpleJammingDetector's processing
        self._setup_simple_detector_hooks()
        
        print(f"GUI initialized, attached to detector on port {self.port}")
        print(f"Detector device: {self.detector.device}")
    
    def _setup_simple_detector_hooks(self):
        """Hook into SimpleJammingDetector's processing pipeline."""
        # Store reference to original method
        original_process = self.simple_detector._process_window
        
        # Store reference to GUI for use in hook
        gui_ref = self
        
        def hooked_process(i_array: np.ndarray, q_array: np.ndarray, timestamp: float):
            """Hooked version that updates GUI plots after processing."""
            try:
                # Call original processing
                original_process(i_array, q_array, timestamp)
                
                # After processing, update GUI data
                # The detector's state is updated by the original_process call
                detector = gui_ref.detector
                
                # Get detection results from last detection
                is_anomaly = False
                error = 0.0
                
                if hasattr(detector, 'detection_history') and len(detector.detection_history) > 0:
                    last_detection = detector.detection_history[-1]
                    is_anomaly = last_detection.get('is_anomaly', False)
                    error = last_detection.get('error', 0.0)
                
                # Update plot data
                current_time = detector.sample_count
                gui_ref.plot_data['time'].append(current_time)
                gui_ref.plot_data['error'].append(error)
                
                # Get threshold
                threshold = detector.threshold_manager.get_threshold()
                if threshold == float('inf'):
                    threshold = error * 2  # For display during learning
                gui_ref.plot_data['threshold'].append(threshold)
                gui_ref.plot_data['detections'].append(is_anomaly)
                
                # Update constellation plot with actual I/Q data
                if len(i_array) > 100:
                    step = len(i_array) // 100
                    gui_ref.plot_data['i_constellation'].extend(i_array[::step])
                    gui_ref.plot_data['q_constellation'].extend(q_array[::step])
                else:
                    gui_ref.plot_data['i_constellation'].extend(i_array[:100])  # Limit points
                    gui_ref.plot_data['q_constellation'].extend(q_array[:100])
                
                # Compute spectral data
                try:
                    freqs, psd = signal.periodogram(i_array + 1j * q_array, scaling='density')
                    gui_ref.plot_data['spec_freqs'] = freqs
                    gui_ref.plot_data['spec_psd'] = psd
                except Exception as e:
                    print(f"Spectral calculation error: {e}")
                
                # Update performance counter
                gui_ref.fps_counter.append(time.time())
                
            except Exception as e:
                print(f"GUI hook error: {e}")
                import traceback
                traceback.print_exc()
        
        # Replace the method
        self.simple_detector._process_window = hooked_process
    
    def setup_gui(self):
        """Create and layout GUI components."""
        # Create main panels
        self.control_panel = ControlPanel(self.root, self)
        self.control_panel.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_panel = StatusPanel(self.root, self)
        self.status_panel.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_panel = StatisticsPanel(self.root, self)
        self.stats_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # Create plot manager
        self.plot_manager = PlotManager(self.root, self.plot_data)
        self.plot_manager.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def start_detection(self):
        """Start the detection system."""
        if not self.running:
            self.running = True
            
            # SimpleJammingDetector handles its own threading
            self.simple_detector.start()
            
            # Start the plot updates
            self.plot_manager.start_animation()
            self.control_panel.on_detection_started()
            print("GUI started detection via SimpleJammingDetector")
    
    def stop_detection(self):
        """Stop the detection system."""
        if self.running:
            self.running = False
            
            # Stop the detector
            self.simple_detector.stop()
            
            # Stop plot updates
            self.plot_manager.stop_animation()
            
            # Update control panel
            self.control_panel.on_detection_stopped()
            
            print("GUI stopped detection")
    
    def start_learning(self, duration: int = 60):
        """Start learning phase."""
        # Delegate directly to the detector
        self.detector.start_learning(duration)
            
        self.control_panel.on_learning_started(duration)
        
        # Schedule learning end
        def end_learning():
            stats = self.detector.stop_learning()
            message = f"Learning complete. Processed {stats['samples_processed']} samples"
            self.control_panel.on_learning_stopped()
            tk.messagebox.showinfo("Learning Complete", message)
            
        self.root.after(duration * 1000, end_learning)
        
        print(f"Learning phase started for {duration} seconds")
    
    def save_model(self):
        """Save the current model."""
        try:
            path = self.detector.save_model()
            tk.messagebox.showinfo("Success", f"Model saved to:\n{path}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")
    
    def load_model(self):
        """Load a saved model."""
        try:
            filename = tk.filedialog.askopenfilename(
                initialdir=self.detector.model_dir,
                title="Select model file",
                filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
            )
            if filename:
                self.detector.load_model(filename)
                tk.messagebox.showinfo("Success", "Model loaded successfully")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def get_statistics(self) -> dict:
        """Get current system statistics."""
        # Calculate FPS
        fps = 0
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                fps = len(self.fps_counter) / time_diff
        
        # Get threshold info
        threshold_stats = self.detector.threshold_manager.get_statistics()
        threshold = threshold_stats.get('current_threshold', float('inf'))
        
        # Calculate detection rate
        detection_rate = 0
        if len(self.plot_data['detections']) > 10:
            recent_detections = list(self.plot_data['detections'])[-100:]
            detection_rate = sum(recent_detections) / len(recent_detections) * 100
        
        # GPU memory if available
        gpu_memory = 0
        if hasattr(self.detector, 'device') and self.detector.device.type == 'cuda':
            try:
                import torch
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            except:
                pass
        
        # Get status
        status_info = self.detector.get_status()
            
        if status_info.get('mode') == 'learning':
            status = f"Learning: {status_info.get('remaining_time', 0):.1f}s"
            status_color = 'orange'
        else:
            status = "Detection Mode"
            status_color = 'green'
        
        return {
            'samples': self.detector.sample_count,
            'detections': self.detector.total_detections,
            'threshold': "Learning..." if threshold == float('inf') else f"{threshold:.4f}",
            'fps': f"{fps:.1f}",
            'gpu_memory': f"{gpu_memory:.1f} MB",
            'detection_rate': f"{detection_rate:.1f}%",
            'status': status,
            'status_color': status_color
        }
    
    def on_closing(self):
        """Handle window closing."""
        # Stop detection if running
        if self.running:
            self.stop_detection()
        
        # The SimpleJammingDetector is stopped by stop_detection()
        
        # Stop plot animation
        try:
            self.plot_manager.stop_animation()
        except:
            pass
        
        # Destroy window
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop."""
        print(f"Adaptive RF Receiver GUI ready on port {self.port}")
        print(f"Device: {self.detector.device}")
        print("Click 'Start Detection' to begin monitoring")
        
        # Enter main loop
        self.root.mainloop()