"""
Optimized matplotlib plotting for the Adaptive RF Receiver GUI.
Uses blitting and efficient updates for better performance.
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
import time


class PlotManager:
    """Optimized plot manager with GPU-friendly updates."""
    
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
        self.last_draw_time = 0
        self.min_draw_interval = 0.05  # Max 20 FPS for plots
        
        # For efficient updates
        self.backgrounds = {}
        self.artists = {}
    
    def setup_plots(self):
        """Initialize all plots with optimized settings."""
        # Error and threshold plot
        self.ax_error.set_title("Reconstruction Error & Detection Threshold", fontsize=12, weight='bold')
        self.ax_error.set_xlabel("Sample Number")
        self.ax_error.set_ylabel("Error Magnitude")
        self.ax_error.grid(True, alpha=0.3)
        
        self.error_line, = self.ax_error.plot([], [], 'b-', linewidth=1.5, label='Error', animated=True)
        self.threshold_line, = self.ax_error.plot([], [], 'r--', linewidth=2, label='Threshold', animated=True)
        self.ax_error.legend(loc='upper right')
        self.ax_error.set_xlim(0, 500)
        self.ax_error.set_ylim(0, 1)
        
        # Detection events plot
        self.ax_detections.set_title("Jamming Detection Events", fontsize=12, weight='bold')
        self.ax_detections.set_xlabel("Sample Number")
        self.ax_detections.set_ylabel("Detection")
        self.ax_detections.set_ylim(-0.1, 1.1)
        self.ax_detections.grid(True, alpha=0.3)
        self.ax_detections.set_yticks([0, 1])
        self.ax_detections.set_yticklabels(['Clean', 'Jammed'])
        
        # Detection scatter for efficiency
        self.detection_scatter = self.ax_detections.scatter([], [], c='red', s=50, alpha=0.8, animated=True)
        
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
            [], [], s=1, alpha=0.6, c='blue', animated=True
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
        self.spectral_line, = self.ax_spectral.plot([], [], 'm-', linewidth=1.5, animated=True)
        self.ax_spectral.set_xlim(0, 1)
        self.ax_spectral.set_ylim(-60, 0)
        
        # Draw once to get backgrounds
        self.fig.canvas.draw()
        
        # Store backgrounds for blitting
        self.backgrounds = {
            'error': self.fig.canvas.copy_from_bbox(self.ax_error.bbox),
            'detections': self.fig.canvas.copy_from_bbox(self.ax_detections.bbox),
            'constellation': self.fig.canvas.copy_from_bbox(self.ax_constellation.bbox),
            'spectral': self.fig.canvas.copy_from_bbox(self.ax_spectral.bbox)
        }
        
        # Store artists for efficient updates
        self.artists = {
            'error': [self.error_line, self.threshold_line],
            'detections': [self.detection_scatter],
            'constellation': [self.constellation_scatter],
            'spectral': [self.spectral_line]
        }
    
    def update_plots(self, frame=None):
        """Optimized plot update using blitting."""
        current_time = time.time()
        
        # Rate limit updates
        if current_time - self.last_draw_time < self.min_draw_interval:
            return
        
        try:
            # Update error plot
            if self.plot_data['time'] and self.plot_data['error']:
                self.update_error_plot_fast()
            
            # Update detection plot
            if self.plot_data['time'] and self.plot_data['detections']:
                self.update_detection_plot_fast()
            
            # Update constellation (less frequently)
            if len(self.plot_data['i_constellation']) > 50:
                self.update_constellation_plot_fast()
            
            # Update spectral (least frequently)
            if self.plot_data.get('spec_freqs') is not None:
                self.update_spectral_plot_fast()
            
            self.last_draw_time = current_time
            
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def update_error_plot_fast(self):
        """Fast error plot update using blitting."""
        time_data = np.array(self.plot_data['time'])
        error_data = np.array(self.plot_data['error'])
        threshold_data = np.array(self.plot_data['threshold'])
        
        if len(time_data) > 0:
            # Update data
            self.error_line.set_data(time_data, error_data)
            self.threshold_line.set_data(time_data, threshold_data)
            
            # Update limits if needed
            if len(time_data) > 1:
                self.ax_error.set_xlim(time_data[0], time_data[-1])
                
                # Dynamic y-limits
                valid_errors = error_data[error_data < float('inf')]
                valid_thresholds = threshold_data[threshold_data < float('inf')]
                
                if len(valid_errors) > 0:
                    y_max = max(np.max(valid_errors), 
                               np.max(valid_thresholds) if len(valid_thresholds) > 0 else 1) * 1.1
                    self.ax_error.set_ylim(0, y_max)
            
            # Blit update
            self.blit_update('error')
    
    def update_detection_plot_fast(self):
        """Fast detection plot update."""
        time_data = np.array(self.plot_data['time'])
        detection_data = np.array(self.plot_data['detections'])
        
        # Get detection times
        detection_indices = np.where(detection_data)[0]
        if len(detection_indices) > 0:
            detection_times = time_data[detection_indices]
            detection_y = np.ones_like(detection_times) * 0.5
            
            # Update scatter data
            self.detection_scatter.set_offsets(np.column_stack([detection_times, detection_y]))
        else:
            self.detection_scatter.set_offsets(np.empty((0, 2)))
        
        # Update x-limits to match error plot
        if len(time_data) > 1:
            self.ax_detections.set_xlim(time_data[0], time_data[-1])
        
        self.blit_update('detections')
    
    def update_constellation_plot_fast(self):
        """Fast constellation update."""
        # Use last N points for performance
        max_points = 500
        i_data = list(self.plot_data['i_constellation'])[-max_points:]
        q_data = list(self.plot_data['q_constellation'])[-max_points:]
        
        if i_data and q_data:
            # Update scatter data
            points = np.column_stack([i_data, q_data])
            self.constellation_scatter.set_offsets(points)
            
            # Color by recency
            colors = np.linspace(0.3, 1.0, len(i_data))
            self.constellation_scatter.set_array(colors)
            
            # Dynamic limits
            i_range = max(abs(min(i_data)), abs(max(i_data)), 1) * 1.2
            q_range = max(abs(min(q_data)), abs(max(q_data)), 1) * 1.2
            plot_range = max(i_range, q_range)
            
            self.ax_constellation.set_xlim(-plot_range, plot_range)
            self.ax_constellation.set_ylim(-plot_range, plot_range)
        
        self.blit_update('constellation')
    
    def update_spectral_plot_fast(self):
        """Fast spectral update."""
        freqs = self.plot_data.get('spec_freqs')
        psd = self.plot_data.get('spec_psd')
        
        if freqs is not None and psd is not None:
            # Convert to dB
            psd_db = 10 * np.log10(psd + 1e-10)
            
            self.spectral_line.set_data(freqs, psd_db)
            self.ax_spectral.set_xlim(0, max(freqs))
            self.ax_spectral.set_ylim(np.min(psd_db) - 10, np.max(psd_db) + 10)
        
        self.blit_update('spectral')
    
    def blit_update(self, plot_name: str):
        """Efficient blit update for a specific plot."""
        if plot_name in self.backgrounds and plot_name in self.artists:
            # Restore background
            self.canvas.restore_region(self.backgrounds[plot_name])
            
            # Redraw artists
            ax = getattr(self, f'ax_{plot_name}')
            for artist in self.artists[plot_name]:
                ax.draw_artist(artist)
            
            # Blit the updated region
            self.canvas.blit(ax.bbox)
    
    def start_animation(self):
        """Start optimized animation."""
        if self.animation is not None:
            self.stop_animation()
        
        # Use faster interval since we're rate limiting internally
        self.animation = FuncAnimation(
            self.fig, self.update_plots,
            interval=50,  # 20 FPS max
            blit=False,  # We handle blitting manually
            cache_frame_data=False
        )
    
    def stop_animation(self):
        """Stop animation."""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation._stop()
            self.animation = None
    
    def pack(self, **kwargs):
        """Pack the frame widget."""
        self.frame.pack(**kwargs)
    
    def clear_plots(self):
        """Clear all plots efficiently."""
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
        
        # Redraw
        self.canvas.draw_idle()


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
