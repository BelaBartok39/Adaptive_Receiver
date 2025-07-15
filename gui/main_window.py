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
from typing import Optional, Union

from .plots import PlotManager
from .widgets import StatusPanel, ControlPanel, StatisticsPanel


class AdaptiveReceiverGUI:
    """Main GUI application for the Adaptive RF Receiver."""
    
    def __init__(self, detector_or_wrapper, port: Optional[int] = None, window_title: str = "Adaptive RF Receiver"):
        """
        Initialize the GUI.
        
        Args:
            detector_or_wrapper: Either AnomalyDetector or SimpleJammingDetector instance
            port: UDP port for receiving data (optional if using SimpleJammingDetector)
            window_title: Title for the main window
        """
        # Handle both SimpleJammingDetector and raw AnomalyDetector
        if hasattr(detector_or_wrapper, 'detector'):
            # This is a SimpleJammingDetector wrapper
            self.simple_detector = detector_or_wrapper
            self.detector = detector_or_wrapper.detector
            self.port = port or detector_or_wrapper.port
            self.use_external_socket = True  # SimpleJammingDetector handles networking
        else:
            # This is a raw AnomalyDetector
            self.simple_detector = None
            self.detector = detector_or_wrapper
            self.port = port or 12345
            self.use_external_socket = False  # We handle networking
        
        self.running = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup networking only if not using SimpleJammingDetector
        if not self.use_external_socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)
        
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
        
        # IQ signal buffers (only used if we handle networking)
        if not self.use_external_socket:
            self.i_buffer = []
            self.q_buffer = []
            self.last_timestamp = 0
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Setup GUI components
        self.setup_gui()
        
        # If using SimpleJammingDetector, hook into its processing
        if self.use_external_socket:
            self._setup_simple_detector_hooks()
        
        print(f"GUI initialized on port {self.port}")
        print(f"Detector device: {self.detector.device}")
        print(f"Using external networking: {self.use_external_socket}")
    
    def _setup_simple_detector_hooks(self):
        """Hook into SimpleJammingDetector's processing pipeline."""
        # Store original process_window method (robustly detect name)
        orig = getattr(self.simple_detector, '_process_window', None)
        if orig is None:
            orig = getattr(self.simple_detector, 'process_window', None)
        if orig is None:
            print("No process_window method found on detector wrapper; GUI hooks disabled.")
            return
        original_process_window = orig

        def hooked_process_window():
            """Hooked version that updates GUI plots after processing."""
            try:
                # Snapshot I/Q buffers before processing clears them
                buf = list(self.simple_detector.i_buffer)
                qb = list(self.simple_detector.q_buffer)
                ws = self.simple_detector.window_size
                if len(buf) < ws or len(qb) < ws:
                    # Not enough data yet: still process but ignore errors
                    try:
                        original_process_window()
                    except Exception:
                        pass
                    return
                # Take window data
                i_array = np.array(buf[:ws])
                q_array = np.array(qb[:ws])
                # Call original processing (which calls detect and clears buffers)
                try:
                    original_process_window()
                except Exception:
                    # Ignore FP16 unscale errors and others
                    pass
                # Retrieve last detection results stored by detector
                is_jammed = getattr(self.detector, 'last_is_anomaly', None)
                error = getattr(self.detector, 'last_error', None)
                # Only update if valid
                if is_jammed is None or error is None:
                    return
                # Compute spectral data for spectral plot
                try:
                    freqs, psd = signal.periodogram(i_array + 1j * q_array, scaling='density')
                    self.plot_data['spec_freqs'] = freqs
                    self.plot_data['spec_psd'] = psd
                except Exception as _:
                    pass
                # Update plot data
                current_time = self.detector.sample_count
                self.plot_data['time'].append(current_time)
                self.plot_data['error'].append(error)
                # Use actual adaptive threshold
                # Freeze threshold after learning using stable_threshold to allow crossings
                manager = self.detector.threshold_manager
                if not self.detector.is_learning and manager.stable_threshold is not None:
                    threshold = manager.stable_threshold
                else:
                    threshold = manager.get_threshold()
                self.plot_data['threshold'].append(threshold)
                self.plot_data['detections'].append(is_jammed)
                # Update constellation using snapshot
                if len(i_array) > 100:
                    step = len(i_array) // 100
                    self.plot_data['i_constellation'].extend(i_array[::step])
                    self.plot_data['q_constellation'].extend(q_array[::step])
                else:
                    self.plot_data['i_constellation'].extend(i_array)
                    self.plot_data['q_constellation'].extend(q_array)
                # Update performance counter
                self.fps_counter.append(time.time())
            except Exception as e:
                print(f"GUI hook error: {e}")

        # Replace the method on the simple_detector
        self.simple_detector._process_window = hooked_process_window
    
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
            
            if self.use_external_socket:
                # Let SimpleJammingDetector handle everything
                # Just start the plot updates
                self.plot_manager.start_animation()
                self.control_panel.on_detection_started()
                print("GUI connected to SimpleJammingDetector")
            else:
                # Start our own data reception thread
                self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self.receive_thread.start()
                
                # Start plot updates
                self.plot_manager.start_animation()
                
                # Update control panel
                self.control_panel.on_detection_started()
                
                print("Detection started")
    
    def stop_detection(self):
        """Stop the detection system."""
        if self.running:
            self.running = False
            
            # Stop plot updates
            self.plot_manager.stop_animation()
            
            # Update control panel
            self.control_panel.on_detection_stopped()
            
            print("Detection stopped")
    
    def start_learning(self, duration: int = 60):
        """Start learning phase."""
        if self.use_external_socket:
            # Use SimpleJammingDetector's detector
            self.simple_detector.detector.start_learning(duration)
        else:
            # Use direct detector
            self.detector.start_learning(duration)
            
        self.control_panel.on_learning_started(duration)
        
        # Schedule learning end
        def end_learning():
            if self.use_external_socket:
                message = self.simple_detector.detector.stop_learning()
            else:
                message = self.detector.stop_learning()
            self.control_panel.on_learning_stopped()
            tkinter.messagebox.showinfo("Learning Complete", message)
            
        self.root.after(duration * 1000, end_learning)
        
        print(f"Learning phase started for {duration} seconds")
    
    def save_model(self):
        """Save the current model."""
        try:
            path = self.detector.save_model()
            tkinter.messagebox.showinfo("Success", f"Model saved to:\n{path}")
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")
    
    def load_model(self):
        """Load a saved model."""
        try:
            filename = tkinter.filedialog.askopenfilename(
                initialdir=self.detector.model_dir,
                title="Select model file",
                filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
            )
            if filename:
                message = self.detector.load_model(filename)
                tkinter.messagebox.showinfo("Success", message)
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def _receive_loop(self):
        """Main data reception loop (only used when not using SimpleJammingDetector)."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                
                if len(data) < 20:
                    continue
                
                # Parse packet header
                timestamp = struct.unpack('!d', data[0:8])[0]
                samples_in_packet = struct.unpack('!I', data[8:12])[0]
                
                # Check for new data block
                if timestamp != self.last_timestamp and self.last_timestamp != 0:
                    if len(self.i_buffer) >= self.detector.window_size:
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
                if len(self.i_buffer) >= self.detector.window_size:
                    self._process_window()
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receive error: {e}")
    
    def _process_window(self):
        """Process a window of I/Q samples (only used when not using SimpleJammingDetector)."""
        # Ensure metrics variable is always defined
        metrics = {}
        try:
            # Get processing window
            i_array = np.array(self.i_buffer[:self.detector.window_size])
            q_array = np.array(self.q_buffer[:self.detector.window_size])
            
            # Run detection
            is_jammed, error, metrics = self.detector.detect(i_array, q_array)
            
            # Compute spectral data for spectral plot
            try:
                freqs, psd = signal.periodogram(i_array + 1j * q_array, scaling='density')
                self.plot_data['spec_freqs'] = freqs
                self.plot_data['spec_psd'] = psd
            except Exception:
                pass
            # Update plot data
            current_time = self.detector.sample_count
            self.plot_data['time'].append(current_time)
            self.plot_data['error'].append(error)
            
            # Freeze threshold after learning for detection crossings
            manager = self.detector.threshold_manager
            if not self.detector.is_learning and manager.stable_threshold is not None:
                threshold = manager.stable_threshold
            else:
                threshold = manager.get_threshold()
            if threshold == float('inf'):
                threshold = error * 2  # For display purposes during learning
            self.plot_data['threshold'].append(threshold)
            self.plot_data['detections'].append(is_jammed)
            
            # Update constellation data (subsample for performance)
            if len(i_array) > 100:
                step = len(i_array) // 100
                self.plot_data['i_constellation'].extend(i_array[::step])
                self.plot_data['q_constellation'].extend(q_array[::step])
            else:
                self.plot_data['i_constellation'].extend(i_array)
                self.plot_data['q_constellation'].extend(q_array)
            
            # Update performance counters
            self.fps_counter.append(time.time())
            
            # Log significant detections
            if is_jammed and not self.detector.is_learning:
                timestamp_str = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp_str}] JAMMING DETECTED! Error: {error:.4f}, Threshold: {threshold:.4f}")
            
            # Slide buffer
            self.i_buffer = self.i_buffer[self.detector.window_size:]
            self.q_buffer = self.q_buffer[self.detector.window_size:]
            
        except Exception as e:
            print(f"Processing error: {e}")
    
    def get_statistics(self) -> dict:
        """Get current system statistics."""
        # Calculate FPS
        fps = 0
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
        
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
                # Import torch only when needed
                import torch
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
            except Exception:
                pass
        
        # Get status from appropriate detector
        if self.use_external_socket and hasattr(self.simple_detector, 'detector'):
            status_info = self.simple_detector.detector.get_status()
        else:
            status_info = self.detector.get_status()
            
        # status_info is a dictionary
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
        
        # If using SimpleJammingDetector wrapper, stop its receiving thread and socket
        if self.use_external_socket and self.simple_detector:
            try:
                self.simple_detector.running = False
                if hasattr(self.simple_detector, 'receive_thread'):
                    self.simple_detector.receive_thread.join(timeout=1)
            except Exception:
                pass
            try:
                self.simple_detector.socket.close()
            except Exception:
                pass
        # Close our own socket if created
        if not self.use_external_socket:
            try:
                self.socket.close()
            except Exception:
                pass
        # Stop plot animation
        try:
            self.plot_manager.stop_animation()
        except Exception:
            pass
        # Quit mainloop and exit
        try:
            self.root.quit()
        except Exception:
            pass
        import sys; sys.exit(0)
        
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop."""
        print(f"Adaptive RF Receiver GUI ready on port {self.port}")
        print(f"Device: {self.detector.device}")
        if self.use_external_socket:
            print("Using SimpleJammingDetector for data processing")
            print("Auto-starting detection...")
            self.start_detection()
        else:
            print("Click 'Start Detection' to begin monitoring")
        # Enter main loop
        self.root.mainloop()
