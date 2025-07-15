"""
Main GUI application for the Adaptive RF Receiver.
Direct integration with AnomalyDetector - no wrapper needed.
"""

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import tkinter.filedialog
import numpy as np
import torch
import threading
import queue
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque

# Core imports
from core.detection.anomaly_detector import AnomalyDetector
from network.receiver import RFDataReceiver
from core.utils.data_buffer import SlidingWindowBuffer
from config.config_loader import load_detector_config, load_network_config, load_signal_config

# GUI imports
from .plots import PlotManager
from .widgets import StatusPanel, ControlPanel, StatisticsPanel


class AdaptiveReceiverGUI:
    """Direct GUI for the Adaptive RF Receiver with optimized performance."""
    
    def __init__(self, port: int = None, window_size: int = None, auto_start: bool = True):
        """
        Initialize the GUI with direct detector integration.
        
        Args:
            port: UDP port to listen on (overrides config)
            window_size: Size of processing windows (overrides config)
            auto_start: If True, detection starts automatically when GUI launches.
        """
        # Store auto-start flag
        self.auto_start = auto_start
        # Load configurations
        self.network_config = load_network_config()
        self.signal_config = load_signal_config()
        self.detector_config = load_detector_config()
        
        # Set parameters
        self.port = port if port is not None else self.network_config.get('udp_port', 12345)
        self.window_size = window_size if window_size is not None else self.signal_config.get('window_size', 1024)
        
        # Initialize detector directly
        print(f"Initializing detector on port {self.port}")
        self.detector = AnomalyDetector(
            window_size=self.window_size,
            config=self.detector_config
        )
        print(f"Using device: {self.detector.device}")
        
        # Initialize receiver
        self.receiver = RFDataReceiver(
            port=self.port,
            buffer_size=self.network_config.get('buffer_size', 65536),
            callback=self._on_data_received
        )
        
        # Initialize window buffer
        self.window_buffer = SlidingWindowBuffer(
            window_size=self.window_size,
            stride=self.window_size // 2
        )
        
        # Threading components for GPU processing
        self.process_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        self.gpu_thread = None
        self.running = False
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.detection_count = 0
        self.last_detection_time = 0
        
        # Data buffers for plots (reduced size for performance)
        self.plot_data = {
            'time': deque(maxlen=500),
            'error': deque(maxlen=500),
            'threshold': deque(maxlen=500),
            'detections': deque(maxlen=500),
            'i_constellation': deque(maxlen=200),
            'q_constellation': deque(maxlen=200),
            'spec_freqs': None,
            'spec_psd': None
        }
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Adaptive RF Receiver")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup GUI components
        self.setup_gui()
        # Schedule auto-start of detection if configured
        if self.auto_start:
            self.root.after(100, self.start_detection)
        
        # Start periodic GUI update
        self.update_counter = 0
        self.schedule_gui_update()
        
        print("GUI initialized and ready")
        # Start plot animation so canvas is active
        self.plot_manager.start_animation()
    
    def _on_data_received(self, packet_data: Dict):
        """Handle incoming data from network."""
        if not self.running:
            return
        # Debug: data reception callback invoked
        print(f"[DEBUG] GUI _on_data_received: received {len(packet_data['samples'])} samples")
        
        samples = packet_data['samples']
        if len(samples) > 0:
            i_data = samples[:, 0]
            q_data = samples[:, 1]
            
            # Get windows from buffer
            windows = self.window_buffer.process_samples(i_data, q_data)
            
            # Queue windows for GPU processing
            for i_window, q_window in windows:
                try:
                    self.process_queue.put_nowait((i_window, q_window, packet_data['timestamp']))
                except queue.Full:
                    # Drop oldest if queue is full
                    try:
                        self.process_queue.get_nowait()
                        self.process_queue.put_nowait((i_window, q_window, packet_data['timestamp']))
                    except:
                        pass
    
    def _gpu_processing_thread(self):
        """Dedicated thread for GPU processing."""
        import torch
        
        # Ensure CUDA context is created in this thread
        if self.detector.device.type == 'cuda':
            device = self.detector.device
            if isinstance(device, str):
                device = torch.device(device)
            torch.cuda.set_device(device.index if device.index is not None else 0)
        
        while self.running:
            try:
                # Get data from queue with timeout
                i_window, q_window, timestamp = self.process_queue.get(timeout=0.1)
                
                # Process on GPU
                is_anomaly, confidence, metrics = self.detector.detect(i_window, q_window)
                
                # Queue results for GUI update
                result = {
                    'timestamp': timestamp,
                    'is_anomaly': is_anomaly,
                    'confidence': confidence,
                    'error': metrics['error'],
                    'i_data': i_window[::10],  # Downsample for constellation
                    'q_data': q_window[::10]
                }
                
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Drop oldest result
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU processing error: {e}")
    
    def schedule_gui_update(self):
        """Schedule periodic GUI updates from main thread."""
        print(f"[DEBUG] schedule_gui_update: running, update_counter={self.update_counter}")
        # Update GUI data and plots
        self.update_gui_from_results()
        # Update plots directly in case animation callback fails
        try:
            self.plot_manager.update_plots()
            self.plot_manager.canvas.draw_idle()
        except Exception:
            pass
        # Schedule next update (30 FPS)
        if self.running or self.detector.is_learning:
            self.root.after(33, self.schedule_gui_update)
        else:
            self.root.after(100, self.schedule_gui_update)
    
    def update_gui_from_results(self):
        """Update GUI with processing results (main thread only)."""
        # Debug: start processing GUI results
        print("[DEBUG] update_gui_from_results: starting cycle")
        # Process all available results
        results_processed = 0
        max_results = 10  # Limit per update cycle
        
        while results_processed < max_results:
            try:
                result = self.result_queue.get_nowait()
                results_processed += 1
                
                # Update plot data
                current_time = self.detector.sample_count
                self.plot_data['time'].append(current_time)
                self.plot_data['error'].append(result['error'])
                
                # Get current threshold
                threshold = self.detector.threshold_manager.get_threshold()
                # Handle learning mode display
                if threshold == float('inf'):
                    # During learning, show a reasonable threshold for visualization
                    if len(self.plot_data['error']) > 10:
                        recent_errors = list(self.plot_data['error'])[-50:]
                        threshold = np.mean(recent_errors) + 2 * np.std(recent_errors)
                    else:
                        threshold = result['error'] * 1.5
                
                self.plot_data['threshold'].append(threshold)
                self.plot_data['detections'].append(result['is_anomaly'])
                
                # Update constellation data (less frequently)
                if self.update_counter % 5 == 0:
                    self.plot_data['i_constellation'].extend(result['i_data'])
                    self.plot_data['q_constellation'].extend(result['q_data'])
                
                # Handle detection logging
                if result['is_anomaly'] and not self.detector.is_learning:
                    current_time = time.time()
                    if current_time - self.last_detection_time > 1.0:
                        self.detection_count += 1
                        print(f"\n[{time.strftime('%H:%M:%S')}] JAMMING DETECTED! "
                              f"Confidence: {result['confidence']:.2f}, "
                              f"Error: {result['error']:.4f}, "
                              f"Threshold: {threshold:.4f}")
                        self.last_detection_time = current_time
                
                # Update FPS counter
                self.fps_counter.append(time.time())
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"GUI update error: {e}")
                break
        
        # Debug: finished processing GUI results
        print(f"[DEBUG] update_gui_from_results: processed {results_processed} results")
        self.update_counter += 1
    
    def setup_gui(self):
        """Create and layout GUI components."""
        self.control_panel = ControlPanel(self.root, self)
        self.control_panel.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_panel = StatusPanel(self.root, self)
        self.status_panel.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_panel = StatisticsPanel(self.root, self)
        self.stats_panel.pack(fill=tk.X, padx=10, pady=5)
        
        self.plot_manager = PlotManager(self.root, self.plot_data)
        self.plot_manager.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def start_detection(self):
        """Start the detection system."""
        print("[DEBUG] start_detection called")
        if not self.running:
            self.running = True
            
            # Clear queues
            while not self.process_queue.empty():
                self.process_queue.get_nowait()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
            
            # Start receiver
            self.receiver.start()
            
            # Start GPU processing thread
            self.gpu_thread = threading.Thread(target=self._gpu_processing_thread, daemon=True)
            self.gpu_thread.start()
            
            # Start plot animation
            self.plot_manager.start_animation()
            self.control_panel.on_detection_started()
            
            print("Detection started")
    
    def stop_detection(self):
        """Stop the detection system."""
        if self.running:
            self.running = False
            
            # Stop receiver
            self.receiver.stop()
            
            # Wait for GPU thread
            if self.gpu_thread:
                self.gpu_thread.join(timeout=2)
            
            # Stop plots
            self.plot_manager.stop_animation()
            self.control_panel.on_detection_stopped()
            
            print("Detection stopped")
    
    def start_learning(self, duration: int = 60):
        """Start learning phase."""
        # Clear previous detection data
        self.plot_data['detections'].clear()
        self.detection_count = 0
        
        # Start learning on detector
        self.detector.start_learning(duration)
        self.control_panel.on_learning_started(duration)
        
        # Schedule learning end
        def end_learning():
            stats = self.detector.stop_learning()
            
            # Get final threshold info
            threshold_stats = self.detector.threshold_manager.get_statistics()
            final_threshold = threshold_stats.get('current_threshold', 'Unknown')
            
            if isinstance(final_threshold, float):
                threshold_str = f"{final_threshold:.4f}"
            else:
                threshold_str = str(final_threshold)
            message = (f"Learning complete!\n"
                      f"Samples processed: {stats['samples_processed']:,}\n"
                      f"Final threshold: {threshold_str}\n"
                      f"Mean error: {threshold_stats.get('mean', 0):.4f}\n"
                      f"Std deviation: {threshold_stats.get('std', 0):.4f}")
            
            self.control_panel.on_learning_stopped()
            tk.messagebox.showinfo("Learning Complete", message)
            
            print(f"\nLearning complete. Threshold set to: {final_threshold:.4f}")
        
        self.root.after(duration * 1000, end_learning)
        print(f"Learning phase started for {duration} seconds...")
    
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
        
        # GPU memory
        gpu_memory = 0
        if self.detector.device.type == 'cuda':
            try:
                import torch
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            except:
                pass
        
        # Status
        status_info = self.detector.get_status()
        if status_info.get('mode') == 'learning':
            status = f"Learning: {status_info.get('remaining_time', 0):.1f}s"
            status_color = 'orange'
        else:
            status = "Detection Mode"
            status_color = 'green'
        
        return {
            'samples': self.detector.sample_count,
            'detections': self.detection_count,
            'threshold': "Learning..." if threshold == float('inf') else f"{threshold:.4f}",
            'fps': f"{fps:.1f}",
            'gpu_memory': f"{gpu_memory:.1f} MB",
            'detection_rate': f"{detection_rate:.1f}%"
        }

    def on_closing(self):
        """Handle window closing."""
        print("Closing GUI...")
        if self.running:
            self.stop_detection()
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop."""
        print(f"Adaptive RF Receiver GUI ready on port {self.port}")
        print(f"Device: {self.detector.device}")
        # Optionally auto-start detection (configurable)
        if self.auto_start:
            self.root.after(100, self.start_detection)
        
        # Run main loop
        self.root.mainloop()
        
        # Cleanup
        self.root.destroy()
        print("GUI closed.")

def main():
    """
    Direct GUI launcher.

    Note: This function auto-starts the GUI and detection upon launch.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive RF Receiver GUI")
    parser.add_argument('--port', type=int, help="UDP port")
    parser.add_argument('--window-size', type=int, help="Window size")
    parser.add_argument('--no-auto-start', action='store_true', help="Do not auto-start detection")
    args = parser.parse_args()
    
    gui = AdaptiveReceiverGUI(
        port=args.port,
        window_size=args.window_size,
        auto_start=not args.no_auto_start
    )
    gui.run()

if __name__ == "__main__":
    main()