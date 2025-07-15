"""
Optimized matplotlib plotting for the Adaptive RF Receiver GUI.
Fixed animation and data display issues.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import numpy as np
from collections import deque
from typing import Dict, List, Optional


class PlotManager:
    """Optimized plot manager with proper animation handling."""
    
    def __init__(self, parent, plot_data: Dict):
        """Initialize the plot manager."""
        self.parent = parent
        self.plot_data = plot_data
        self.animation = None
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        # Use constrained layout for better performance
        self.fig.set_constrained_layout(True)
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
        
        self.ax_error = self.fig.add_subplot(gs[0, 0])
        self.ax_detections = self.fig.add_subplot(gs[1, 0])
        self.ax_spectral = self.fig.add_subplot(gs[2, 0])
        self.ax_constellation = self.fig.add_subplot(gs[:, 1])
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.setup_plots()
        
        # Optimization flags
        self.update_counter = 0
        self._update_scheduled = False
        self.min_draw_interval_ms = 100  # 10 FPS for plots
        
        # Draw initial empty plots
        self.canvas.draw()
    
    def setup_plots(self):
        """Initialize all plots with optimized settings."""
        # Error and threshold plot
        self.ax_error.set_title("Reconstruction Error & Detection Threshold", fontsize=12, weight='bold')
        self.ax_error.set_xlabel("Sample Number")
        self.ax_error.set_ylabel("Error Magnitude")
        self.ax_error.grid(True, alpha=0.3)
        
        self.error_line, = self.ax_error.plot([], [], 'b-', linewidth=1.5, label='Error')
        self.threshold_line, = self.ax_error.plot([], [], 'r--', linewidth=2, label='Threshold')
        self.ax_error.legend(loc='upper right')
        self.ax_error.set_xlim(0, 100)
        self.ax_error.set_ylim(0, 0.1)
        
        # Detection events plot
        self.ax_detections.set_title("Jamming Detection Events", fontsize=12, weight='bold')
        self.ax_detections.set_xlabel("Sample Number")
        self.ax_detections.set_ylabel("Detection")
        self.ax_detections.set_ylim(-0.1, 1.1)
        self.ax_detections.grid(True, alpha=0.3)
        self.ax_detections.set_yticks([0, 1])
        self.ax_detections.set_yticklabels(['Clean', 'Jammed'])
        self.ax_detections.set_xlim(0, 100)
        
        # Detection scatter for efficiency
        self.detection_scatter = self.ax_detections.scatter([], [], c='red', s=50, alpha=0.8)
        
        # Constellation plot
        self.ax_constellation.set_title("I/Q Constellation", fontsize=12, weight='bold')
        self.ax_constellation.set_xlabel("I (In-phase)")
        self.ax_constellation.set_ylabel("Q (Quadrature)")
        self.ax_constellation.grid(True, alpha=0.3)
        self.ax_constellation.set_aspect('equal')
        self.ax_constellation.set_xlim(-4, 4)
        self.ax_constellation.set_ylim(-4, 4)
        
        # Efficient constellation scatter
        self.constellation_scatter = self.ax_constellation.scatter(
            [], [], s=1, alpha=0.6, c='blue'
        )
        
        # Reference circles
        for radius in [1, 2, 3]:
            circle = patches.Circle((0, 0), radius, fill=False, 
                                  color='gray', alpha=0.3, linestyle='--')
            self.ax_constellation.add_patch(circle)
        
        # Spectral plot
        self.ax_spectral.set_title("Power Spectral Density", fontsize=12, weight='bold')
        self.ax_spectral.set_xlabel("Frequency Bin")
        self.ax_spectral.set_ylabel("Power (dB)")
        self.ax_spectral.grid(True, alpha=0.3)
        self.spectral_line, = self.ax_spectral.plot([], [], 'm-', linewidth=1.5)
        self.ax_spectral.set_xlim(0, 1)
        self.ax_spectral.set_ylim(-60, 0)
    
    def update_plots(self, frame=None):
        """Update all plots with current data."""
        if self._update_scheduled:
            return
        self._update_scheduled = True
        self.parent.after(self.min_draw_interval_ms, self._reset_update_scheduled)

        self.update_counter += 1
        
        try:
            # Update error plot
            if len(self.plot_data['time']) > 0 and len(self.plot_data['error']) > 0:
                self.update_error_plot()
            
            # Update detection plot
            if len(self.plot_data['time']) > 0 and len(self.plot_data['detections']) > 0:
                self.update_detection_plot()
            
            # Update constellation less frequently
            if self.update_counter % 5 == 0 and len(self.plot_data['i_constellation']) > 0:
                self.update_constellation_plot()
            
            # Update spectral least frequently
            if self.update_counter % 10 == 0 and self.plot_data.get('spec_freqs') is not None:
                self.update_spectral_plot()
            
            # Single canvas update
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_update_scheduled(self):
        self._update_scheduled = False

    def update_error_plot(self):
        """Update the error and threshold plot."""
        # Get data
        time_data = list(self.plot_data['time'])
        error_data = list(self.plot_data['error'])
        threshold_data = list(self.plot_data['threshold'])
        
        # Ensure same length
        min_len = min(len(time_data), len(error_data), len(threshold_data))
        if min_len == 0:
            return
            
        time_data = time_data[-min_len:]
        error_data = error_data[-min_len:]
        threshold_data = threshold_data[-min_len:]
        
        # Update lines
        self.error_line.set_data(time_data, error_data)
        self.threshold_line.set_data(time_data, threshold_data)
        
        # Update limits
        if len(time_data) > 1:
            self.ax_error.set_xlim(time_data[0], time_data[-1])
            
            # Filter out infinities for y-axis scaling
            valid_errors = [e for e in error_data if np.isfinite(e)]
            valid_thresholds = [t for t in threshold_data if np.isfinite(t)]
            
            if valid_errors:
                all_values = valid_errors + valid_thresholds
                y_min = min(0, min(all_values) * 0.9)
                y_max = max(all_values) * 1.1
                self.ax_error.set_ylim(y_min, y_max)
    
    def update_detection_plot(self):
        """Update the detection events plot."""
        time_data = list(self.plot_data['time'])
        detection_data = list(self.plot_data['detections'])
        
        # Ensure same length
        min_len = min(len(time_data), len(detection_data))
        if min_len == 0:
            return
            
        time_data = time_data[-min_len:]
        detection_data = detection_data[-min_len:]
        
        # Find detection points
        detection_times = []
        for i, det in enumerate(detection_data):
            if det:
                detection_times.append(time_data[i])
        
        # Update scatter plot
        if detection_times:
            detection_y = [0.5] * len(detection_times)
            self.detection_scatter.set_offsets(np.column_stack([detection_times, detection_y]))
        else:
            self.detection_scatter.set_offsets(np.empty((0, 2)))
        
        # Update x-limits to match error plot
        if len(time_data) > 1:
            self.ax_detections.set_xlim(time_data[0], time_data[-1])
    
    def update_constellation_plot(self):
        """Update the I/Q constellation plot."""
        # Get data
        i_data = list(self.plot_data['i_constellation'])
        q_data = list(self.plot_data['q_constellation'])
        
        if not i_data or not q_data:
            return
        
        # Limit points for performance
        max_points = 500
        if len(i_data) > max_points:
            # Take evenly spaced points
            step = max(1, len(i_data) // max_points)
            i_data = i_data[::step]
            q_data = q_data[::step]
        
        # Update scatter plot
        points = np.column_stack([i_data, q_data])
        self.constellation_scatter.set_offsets(points)
        
        # Note: Setting an array here does not affect point colors unless a colormap is set for the scatter plot.
        # If you want to color by recency, create the scatter with a colormap and pass 'c=colors' and 'cmap' arguments.
        # colors = np.linspace(0.3, 1.0, len(i_data))
        # self.constellation_scatter.set_array(colors)
        
        # Auto-scale
        if i_data and q_data:
            i_range = max(abs(min(i_data)), abs(max(i_data)), 0.1)
            q_range = max(abs(min(q_data)), abs(max(q_data)), 0.1)
            plot_range = max(i_range, q_range) * 1.2
            self.ax_constellation.set_xlim(-plot_range, plot_range)
            self.ax_constellation.set_ylim(-plot_range, plot_range)
    
    def update_spectral_plot(self):
        """Update the spectral plot."""
        freqs = self.plot_data.get('spec_freqs')
        psd = self.plot_data.get('spec_psd')
        
        if freqs is not None and psd is not None and len(freqs) > 0:
            psd_db = 10 * np.log10(np.clip(psd, 0, None) + 1e-10)
            psd_db = 10 * np.log10(psd + 1e-10)
            
            self.spectral_line.set_data(freqs, psd_db)
            self.ax_spectral.set_xlim(0, max(freqs))
            
            if len(psd_db) > 0:
                y_min = np.min(psd_db) - 10
                y_max = np.max(psd_db) + 10
                self.ax_spectral.set_ylim(y_min, y_max)
    
    def start_animation(self):
        """Start the plot animation."""
        # Stop any existing animation
        self.stop_animation()
        
        # Create new animation
        self.animation = FuncAnimation(
            self.fig, 
            self.update_plots,
            interval=100,  # Update every 100ms (10 FPS)
            blit=False,
            cache_frame_data=False
        )
        
        print("Plot animation started")
    
    def stop_animation(self):
        """Stop the plot animation."""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
            print("Plot animation stopped")
    
    def pack(self, **kwargs):
        """Pack the frame widget."""
        self.frame.pack(**kwargs)
    
    def clear_plots(self):
        """Clear all plot data and reset plots."""
        # Stop animation first
        was_running = self.animation is not None
        if was_running:
            self.stop_animation()
        
        # Clear data
        for key in self.plot_data:
            if hasattr(self.plot_data[key], 'clear'):
                self.plot_data[key].clear()
        
        # Reset plot elements
        self.error_line.set_data([], [])
        self.threshold_line.set_data([], [])
        self.detection_scatter.set_offsets(np.empty((0, 2)))
        self.constellation_scatter.set_offsets(np.empty((0, 2)))
        self.spectral_line.set_data([], [])
        
        # Reset axes limits
        self.ax_error.set_xlim(0, 100)
        self.ax_error.set_ylim(0, 0.1)
        self.ax_detections.set_xlim(0, 100)
        self.ax_constellation.set_xlim(-4, 4)
        self.ax_constellation.set_ylim(-4, 4)
        
        # Redraw
        self.canvas.draw()
        
        # Restart animation if it was running
        if was_running:
            self.start_animation()

class ConstellationPlot:
    """Standalone constellation plot widget."""
    
    def __init__(self, parent, title: str = "I/Q Constellation"):
        """Initialize constellation plot."""
        self.parent = parent
        self.title = title
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text=title, padding="5")
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.fig.patch.set_facecolor('white')
        
        # Setup plot
        self.ax.set_xlabel("I (In-phase)")
        self.ax.set_ylabel("Q (Quadrature)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plot
        self.scatter = self.ax.scatter([], [], s=1, alpha=0.6)
        
        # Add reference circles
        for radius in [1, 2, 3]:
            circle = patches.Circle((0, 0), radius, fill=False, 
                                  color='gray', alpha=0.3, linestyle='--')
            self.ax.add_patch(circle)
        
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
    
    def update(self, i_data: List[float], q_data: List[float]):
        """Update constellation with new I/Q data."""
        if not i_data or not q_data:
            return
        
        # Limit points for performance
        max_points = 500
        if len(i_data) > max_points:
            step = max(1, len(i_data) // max_points)
            i_data = i_data[::step]
            q_data = q_data[::step]
        
        # Update scatter plot
        self.scatter.set_offsets(np.column_stack([i_data, q_data]))
        
        # Auto-scale if needed
        if i_data and q_data:
            i_range = max(abs(min(i_data)), abs(max(i_data)))
            q_range = max(abs(min(q_data)), abs(max(q_data)))
            plot_range = max(i_range, q_range, 1) * 1.2
            self.ax.set_xlim(-plot_range, plot_range)
            self.ax.set_ylim(-plot_range, plot_range)
        
        self.canvas.draw_idle()
    
    def pack(self, **kwargs):
        """Pack the frame widget."""
        self.frame.pack(**kwargs)
