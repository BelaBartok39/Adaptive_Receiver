"""
Matplotlib plotting functions for the Adaptive RF Receiver GUI.
Includes error plots, threshold visualization, and I/Q constellation plots.
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
    """Manages all plots for the RF receiver GUI."""
    
    def __init__(self, parent, plot_data: Dict):
        """
        Initialize the plot manager.
        
        Args:
            parent: Parent tkinter widget
            plot_data: Dictionary containing plot data buffers
        """
        self.parent = parent
        self.plot_data = plot_data
        self.animation = None
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        # Create subplot layout: 3 rows x 2 columns (spectral added)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
        
        # Error and threshold plot (top left)
        self.ax_error = self.fig.add_subplot(gs[0, 0])
        
        # Detection events (middle left)
        self.ax_detections = self.fig.add_subplot(gs[1, 0])
        
        # Spectral power plot (bottom left)
        self.ax_spectral = self.fig.add_subplot(gs[2, 0])
        
        # I/Q constellation plot (right side, spanning all rows)
        self.ax_constellation = self.fig.add_subplot(gs[:, 1])
        
        self.fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.setup_plots()
        
        # Plot update control
        self.update_counter = 0
        self.update_interval = 3  # Update every 3 calls for better performance
        
        # Performance optimization flags
        self.last_constellation_update = 0
        self.last_spectral_update = 0
    
    def setup_plots(self):
        """Initialize all plots with proper formatting."""
        # Reset tracking variables
        self._last_detection_count = 0
        if hasattr(self, 'constellation_scatter_data'):
            delattr(self, 'constellation_scatter_data')
        
        # Clear all axes first to remove old plot elements and legends
        self.ax_error.clear()
        self.ax_detections.clear()
        self.ax_constellation.clear()
        self.ax_spectral.clear()
        
        # Error and threshold plot
        self.ax_error.set_title("Reconstruction Error & Detection Threshold", fontsize=12, weight='bold')
        self.ax_error.set_xlabel("Sample Number")
        self.ax_error.set_ylabel("Error Magnitude")
        self.ax_error.grid(True, alpha=0.3)
        
        self.error_line, = self.ax_error.plot([], [], 'b-', linewidth=1.5, label='Reconstruction Error')
        self.threshold_line, = self.ax_error.plot([], [], 'r--', linewidth=2, label='Detection Threshold')
        self.ax_error.legend(loc='upper right')
        
        # Detection events plot
        self.ax_detections.set_title("Jamming Detection Events", fontsize=12, weight='bold')
        self.ax_detections.set_xlabel("Sample Number")
        self.ax_detections.set_ylabel("Detection")
        self.ax_detections.set_ylim(-0.1, 1.1)
        self.ax_detections.grid(True, alpha=0.3)
        self.ax_detections.set_yticks([0, 1])
        self.ax_detections.set_yticklabels(['Clean', 'Jammed'])
        
        # I/Q constellation plot
        self.ax_constellation.set_title("I/Q Constellation", fontsize=12, weight='bold')
        self.ax_constellation.set_xlabel("I (In-phase)")
        self.ax_constellation.set_ylabel("Q (Quadrature)")
        self.ax_constellation.grid(True, alpha=0.3)
        self.ax_constellation.set_aspect('equal')
        
        # Constellation scatter plot - create empty plot
        self.constellation_scatter = self.ax_constellation.scatter(
            [], [], s=1, alpha=0.6, c='blue', label='Signal Points'
        )
        
        # Add constellation reference circles
        for radius in [1, 2, 3]:
            circle = patches.Circle((0, 0), radius, fill=False, 
                                  color='gray', alpha=0.3, linestyle='--')
            self.ax_constellation.add_patch(circle)
        
        # Set initial limits
        self.ax_constellation.set_xlim(-4, 4)
        self.ax_constellation.set_ylim(-4, 4)
        
        # Spectral power plot
        self.ax_spectral.set_title("Power Spectral Density", fontsize=12, weight='bold')
        self.ax_spectral.set_xlabel("Frequency Bin")
        self.ax_spectral.set_ylabel("Power")
        self.ax_spectral.grid(True, alpha=0.3)
        # Initial empty line
        self.spectral_line, = self.ax_spectral.plot([], [], 'm-', linewidth=1.5)
        
        # Ensure layout
        self.fig.tight_layout(pad=3.0)
    
    def update_spectral_plot(self):
        """Update the spectral power plot."""
        freqs = self.plot_data.get('spec_freqs')
        psd = self.plot_data.get('spec_psd')
        if freqs is None or psd is None:
            return
        try:
            self.spectral_line.set_data(freqs, psd)
            self.ax_spectral.relim()
            self.ax_spectral.autoscale_view()
        except Exception as e:
            print(f"Spectral plot error: {e}")
    
    def update_plots(self, frame=None):
        """Update all plots with current data."""
        self.update_counter += 1
        
        # Skip updates for performance (update every N calls)
        if self.update_counter % self.update_interval != 0:
            return []
        
        try:
            # Track if any plot actually updated
            plots_updated = False
            
            # Always update error and detection plots (lightweight)
            if self.plot_data['time'] and self.plot_data['error']:
                self.update_error_plot()
                plots_updated = True
                
            if self.plot_data['time'] and self.plot_data['detections']:
                self.update_detection_plot()
                plots_updated = True
            
            # Update expensive plots less frequently
            if self.update_counter % 15 == 0:  # Every 5th cycle
                if self.plot_data['i_constellation'] and self.plot_data['q_constellation']:
                    self.update_constellation_plot()
                    self.last_constellation_update = self.update_counter
                    plots_updated = True
                
            if self.update_counter % 30 == 0:  # Every 10th cycle  
                if self.plot_data.get('spec_freqs') is not None and self.plot_data.get('spec_psd') is not None:
                    self.update_spectral_plot()
                    self.last_spectral_update = self.update_counter
                    plots_updated = True
            
            # Single canvas update at the end, only if something changed
            if plots_updated:
                self.canvas.draw_idle()
            return []
        except Exception as e:
            print(f"Plot update error: {e}")
            return []
    
    def update_error_plot(self):
        """Update the error and threshold plot."""
        # Ensure we have data to plot
        if not self.plot_data['time'] or not self.plot_data['error'] or not self.plot_data['threshold']:
            return
        # Align data lengths and clamp infinities
        time_data = list(self.plot_data['time'])
        error_data = list(self.plot_data['error'])
        threshold_data = [t if t != float('inf') else float('nan') for t in self.plot_data['threshold']]
        n = min(len(time_data), len(error_data), len(threshold_data))
        if n == 0:
            return
        time_data = time_data[-n:]
        error_data = error_data[-n:]
        threshold_data = threshold_data[-n:]
        
        # Update error and threshold lines with aligned data
        self.error_line.set_data(time_data, error_data)
        self.threshold_line.set_data(time_data, threshold_data)
        
        # Auto-scale axes
        self.ax_error.relim()
        self.ax_error.autoscale_view()
        
        # Set reasonable y-limits
        if error_data:
            max_error = max(error_data)
            max_threshold = max([t for t in threshold_data if t != float('inf')] or [max_error])
            y_max = max(max_error, max_threshold) * 1.1
            self.ax_error.set_ylim(0, y_max)
    
    def update_detection_plot(self):
        """Update the detection events plot."""
        if not self.plot_data['time'] or not self.plot_data['detections']:
            return
        # Align data lengths for detection plot
        time_data = list(self.plot_data['time'])
        detection_data = list(self.plot_data['detections'])
        n = min(len(time_data), len(detection_data))
        if n <= 0:
            return
        time_data = time_data[-n:]
        detection_data = detection_data[-n:]
        
        # Only clear and redraw if we don't have existing detection lines or if data structure changed significantly
        if not hasattr(self, '_last_detection_count') or abs(len(time_data) - self._last_detection_count) > 50:
            # Clear previous detection markers
            self.ax_detections.clear()
            
            # Redraw basic setup
            self.ax_detections.set_title("Jamming Detection Events", fontsize=12, weight='bold')
            self.ax_detections.set_xlabel("Sample Number")
            self.ax_detections.set_ylabel("Detection")
            self.ax_detections.set_ylim(-0.1, 1.1)
            self.ax_detections.grid(True, alpha=0.3)
            self.ax_detections.set_yticks([0, 1])
            self.ax_detections.set_yticklabels(['Clean', 'Jammed'])
            
            # Plot detection events as vertical lines
            detection_times = [t for t, d in zip(time_data, detection_data) if d]
            if detection_times:
                self.ax_detections.vlines(detection_times, 0, 1, 
                                        colors='red', alpha=0.8, linewidth=2, label='Jamming Detected')
                # Only add legend if we have detections
                self.ax_detections.legend(loc='upper right')
            
            # Set x-axis to match error plot
            if time_data:
                self.ax_detections.set_xlim(min(time_data), max(time_data))
            
            self._last_detection_count = len(time_data)
    
    def update_constellation_plot(self):
        """Update the I/Q constellation plot."""
        if not self.plot_data['i_constellation'] or not self.plot_data['q_constellation']:
            return
        
        i_data = list(self.plot_data['i_constellation'])
        q_data = list(self.plot_data['q_constellation'])
        
        # Limit data points for performance (show last N points)
        max_points = 500  # Reduced from 1000
        if len(i_data) > max_points:
            i_data = i_data[-max_points:]
            q_data = q_data[-max_points:]
        
        # Update existing scatter plot instead of clearing
        if hasattr(self, 'constellation_scatter') and hasattr(self, 'constellation_scatter_data'):
            # Update existing plot data
            if i_data and q_data:
                self.constellation_scatter.set_offsets(np.column_stack([i_data, q_data]))
                # Update colors for recency effect
                colors = np.linspace(0.3, 1.0, len(i_data))
                self.constellation_scatter.set_array(colors)
        else:
            # First time setup - but don't clear, just update the existing scatter plot
            if i_data and q_data:
                self.constellation_scatter.set_offsets(np.column_stack([i_data, q_data]))
                colors = np.linspace(0.3, 1.0, len(i_data))
                self.constellation_scatter.set_array(colors)
                self.constellation_scatter_data = True
        
        # Set appropriate limits based on data
        if i_data and q_data:
            i_range = max(abs(min(i_data)), abs(max(i_data)))
            q_range = max(abs(min(q_data)), abs(max(q_data)))
            plot_range = max(i_range, q_range, 1) * 1.2
            self.ax_constellation.set_xlim(-plot_range, plot_range)
            self.ax_constellation.set_ylim(-plot_range, plot_range)
        else:
            self.ax_constellation.set_xlim(-4, 4)
            self.ax_constellation.set_ylim(-4, 4)
        
        # Add statistics overlay (less frequently) - but remove old text first
        if i_data and q_data and self.update_counter % 60 == 0:  # Update stats less often
            i_std = np.std(i_data)
            q_std = np.std(q_data)
            power = np.mean(np.array(i_data)**2 + np.array(q_data)**2)
            
            stats_text = f"I σ: {i_std:.3f}\nQ σ: {q_std:.3f}\nPower: {power:.3f}"
            # Remove old text if exists
            for text in self.ax_constellation.texts[:]:  # Use slice to avoid modifying list during iteration
                if 'σ:' in text.get_text():
                    text.remove()
            
            self.ax_constellation.text(0.02, 0.98, stats_text, 
                                     transform=self.ax_constellation.transAxes,
                                     verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def start_animation(self):
        """Start the plot animation."""
        # Stop any existing animation first
        if self.animation is not None:
            self.stop_animation()
            
        # Force initial plot setup
        self.setup_plots()
        self.canvas.draw()
        
        # Start animation without blitting to avoid returning Artist objects
        self.animation = FuncAnimation(
            self.fig, self.update_plots,
            interval=200, blit=False, cache_frame_data=False
        )
        
        # Force immediate update
        self.update_plots(0)
    
    def stop_animation(self):
        """Stop the plot animation."""
        if self.animation is not None:
            try:
                self.animation.event_source.stop()
                # Properly dispose of the animation to prevent warnings
                self.animation._stop()
            except AttributeError:
                # Fallback for different matplotlib versions
                pass
            finally:
                self.animation = None
    
    def pack(self, **kwargs):
        """Pack the frame widget."""
        self.frame.pack(**kwargs)
    
    def clear_plots(self):
        """Clear all plot data."""
        # Stop animation first to prevent interference
        was_running = self.animation is not None
        if was_running:
            self.stop_animation()
        
        # Clear data buffers
        for key in self.plot_data:
            if hasattr(self.plot_data[key], 'clear'):
                self.plot_data[key].clear()
        
        # Reset constellation tracking
        if hasattr(self, 'constellation_scatter_data'):
            delattr(self, 'constellation_scatter_data')
        
        # Reset plots
        self.setup_plots()
        self.canvas.draw()
        
        # Restart animation if it was running
        if was_running:
            self.start_animation()
    
    def save_plots(self, filename: str):
        """Save the current plots to file."""
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            return f"Plots saved to {filename}"
        except Exception as e:
            return f"Error saving plots: {e}"


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
            step = len(i_data) // max_points
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
