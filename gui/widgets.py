"""
Custom GUI widgets for the Adaptive RF Receiver interface.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Dict, Any


class StatusPanel(ttk.LabelFrame):
    """Status display panel showing current system state."""
    
    def __init__(self, parent, gui_instance):
        """Initialize the status panel."""
        super().__init__(parent, text="System Status", padding="10")
        
        self.gui = gui_instance
        
        # Status indicator
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(fill=tk.X)
        
        ttk.Label(self.status_frame, text="Status:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.status_label = ttk.Label(self.status_frame, text="Ready", 
                                     font=('Arial', 10), foreground='blue')
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Device info
        device_text = f"Device: {self.gui.detector.device}"
        ttk.Label(self.status_frame, text=device_text).pack(side=tk.RIGHT)
        
        # Update status periodically
        self.update_status()
    
    def update_status(self):
        """Update the status display."""
        try:
            stats = self.gui.get_statistics()
            status = stats.get('status', 'Unknown')
            color = stats.get('status_color', 'black')
            
            self.status_label.config(text=status, foreground=color)
            
        except Exception as e:
            self.status_label.config(text="Error", foreground='red')
        
        # Schedule next update
        self.after(1000, self.update_status)


class ControlPanel(ttk.LabelFrame):
    """Main control panel with start/stop and learning controls."""
    
    def __init__(self, parent, gui_instance):
        """Initialize the control panel."""
        super().__init__(parent, text="Controls", padding="10")
        
        self.gui = gui_instance
        
        # Main controls frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.X)
        
        # Detection control
        self.start_btn = ttk.Button(main_frame, text="Start Detection", 
                                   command=self.toggle_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Learning control
        self.learn_btn = ttk.Button(main_frame, text="Learn Environment (60s)", 
                                   command=self.start_learning)
        self.learn_btn.pack(side=tk.LEFT, padx=5)
        
        # Model controls frame
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(model_frame, text="Save Model", 
                  command=self.gui.save_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_frame, text="Load Model", 
                  command=self.gui.load_model).pack(side=tk.LEFT, padx=2)
        
        # Clear plots button
        ttk.Button(model_frame, text="Clear Plots", 
                  command=self.clear_plots).pack(side=tk.LEFT, padx=2)
    
    def toggle_detection(self):
        """Toggle detection on/off."""
        if not self.gui.running:
            self.gui.start_detection()
        else:
            self.gui.stop_detection()
    
    def start_learning(self):
        """Start learning phase."""
        duration = 60  # Could be made configurable
        if hasattr(self.gui, 'simple_detector') and self.gui.simple_detector:
            # Using SimpleJammingDetector - start its learning
            print("Starting 60-second learning phase...")
            self.gui.simple_detector.detector.start_learning_phase(duration)
            self.on_learning_started(duration)
            
            # Schedule learning end
            def end_learning():
                message = self.gui.simple_detector.detector.stop_learning_phase()
                self.on_learning_stopped()
                import tkinter.messagebox
                tkinter.messagebox.showinfo("Learning Complete", message)
                
            self.gui.root.after(duration * 1000, end_learning)
        else:
            # Direct detector access
            self.gui.start_learning(duration)
    
    def clear_plots(self):
        """Clear all plot data."""
        self.gui.plot_manager.clear_plots()
    
    def on_detection_started(self):
        """Called when detection starts."""
        self.start_btn.config(text="Stop Detection")
    
    def on_detection_stopped(self):
        """Called when detection stops."""
        self.start_btn.config(text="Start Detection")
    
    def on_learning_started(self, duration: int):
        """Called when learning starts."""
        self.learn_btn.config(state='disabled', text=f"Learning... ({duration}s)")
    
    def on_learning_stopped(self):
        """Called when learning stops."""
        self.learn_btn.config(state='normal', text="Learn Environment (60s)")


class StatisticsPanel(ttk.LabelFrame):
    """Panel displaying performance and detection statistics."""
    
    def __init__(self, parent, gui_instance):
        """Initialize the statistics panel."""
        super().__init__(parent, text="Performance Metrics", padding="5")
        
        self.gui = gui_instance
        
        # Create statistics labels
        self.stat_labels = {}
        
        # Define statistics to display
        stats_info = [
            ("Samples", "0", "Number of processed samples"),
            ("Detections", "0", "Total jamming detections"),
            ("Threshold", "Learning...", "Current detection threshold"),
            ("FPS", "0", "Processing frames per second"),
            ("GPU Memory", "0 MB", "GPU memory usage"),
            ("Detection Rate", "0%", "Percentage of samples detected as jammed")
        ]
        
        # Create grid layout
        for i, (name, default, tooltip) in enumerate(stats_info):
            row = i // 3
            col = (i % 3) * 2
            
            # Label
            label = ttk.Label(self, text=f"{name}:")
            label.grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            
            # Value
            value = ttk.Label(self, text=default, font=('Arial', 9, 'bold'))
            value.grid(row=row, column=col+1, padx=5, pady=2, sticky=tk.W)
            
            # Store reference
            self.stat_labels[name] = value
            
            # Add tooltip (simple implementation)
            self._create_tooltip(label, tooltip)
            self._create_tooltip(value, tooltip)
        
        # Start updating statistics
        self.update_statistics()
    
    def update_statistics(self):
        """Update all statistics displays."""
        try:
            stats = self.gui.get_statistics()
            
            for stat_name, label in self.stat_labels.items():
                value = stats.get(stat_name.lower().replace(' ', '_'), "N/A")
                label.config(text=str(value))
                
        except Exception as e:
            print(f"Statistics update error: {e}")
        
        # Schedule next update
        self.after(1000, self.update_statistics)
    
    def _create_tooltip(self, widget, text):
        """Create a simple tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="lightyellow", 
                            relief="solid", borderwidth=1, font=('Arial', 8))
            label.pack()
            
            # Store tooltip reference
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)


class ConfigPanel(ttk.LabelFrame):
    """Panel for runtime configuration adjustments."""
    
    def __init__(self, parent, gui_instance):
        """Initialize the configuration panel."""
        super().__init__(parent, text="Configuration", padding="10")
        
        self.gui = gui_instance
        
        # Learning duration
        learn_frame = ttk.Frame(self)
        learn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(learn_frame, text="Learning Duration (s):").pack(side=tk.LEFT)
        self.learn_duration = tk.StringVar(value="60")
        ttk.Entry(learn_frame, textvariable=self.learn_duration, width=10).pack(side=tk.LEFT, padx=5)
        
        # Update interval
        update_frame = ttk.Frame(self)
        update_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(update_frame, text="Plot Update Interval:").pack(side=tk.LEFT)
        self.update_interval = tk.StringVar(value="5")
        update_entry = ttk.Entry(update_frame, textvariable=self.update_interval, width=10)
        update_entry.pack(side=tk.LEFT, padx=5)
        
        # Bind update interval changes
        def on_interval_change(*args):
            try:
                interval = int(self.update_interval.get())
                self.gui.plot_manager.update_interval = max(1, interval)
            except ValueError:
                pass
        
        self.update_interval.trace('w', on_interval_change)
    
    def get_learning_duration(self) -> int:
        """Get the configured learning duration."""
        try:
            return int(self.learn_duration.get())
        except ValueError:
            return 60


class AdvancedControlPanel(ttk.LabelFrame):
    """Advanced controls for expert users."""
    
    def __init__(self, parent, gui_instance):
        """Initialize the advanced control panel."""
        super().__init__(parent, text="Advanced Controls", padding="10")
        
        self.gui = gui_instance
        
        # Threshold adjustment
        threshold_frame = ttk.Frame(self)
        threshold_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(threshold_frame, text="Threshold Multiplier:").pack(side=tk.LEFT)
        self.threshold_scale = tk.Scale(threshold_frame, from_=0.5, to=3.0, 
                                       resolution=0.1, orient=tk.HORIZONTAL)
        self.threshold_scale.set(1.0)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Model training controls
        training_frame = ttk.Frame(self)
        training_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(training_frame, text="Force Model Update", 
                  command=self.force_update).pack(side=tk.LEFT, padx=2)
        ttk.Button(training_frame, text="Reset Threshold", 
                  command=self.reset_threshold).pack(side=tk.LEFT, padx=2)
    
    def force_update(self):
        """Force a model update."""
        try:
            if hasattr(self.gui.detector, '_update_model_batch'):
                self.gui.detector._update_model_batch()
                print("Forced model update")
        except Exception as e:
            print(f"Error forcing model update: {e}")
    
    def reset_threshold(self):
        """Reset the detection threshold."""
        try:
            self.gui.detector.threshold_manager.errors.clear()
            self.gui.detector.threshold_manager.stable_threshold = None
            self.gui.detector.threshold_manager.ema_threshold = None
            print("Threshold reset")
        except Exception as e:
            print(f"Error resetting threshold: {e}")


class DebugPanel(ttk.LabelFrame):
    """Debug information panel."""
    
    def __init__(self, parent, gui_instance):
        """Initialize the debug panel."""
        super().__init__(parent, text="Debug Info", padding="10")
        
        self.gui = gui_instance
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(self)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.debug_text = tk.Text(text_frame, height=8, width=50, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=scrollbar.set)
        
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=5)
        
        # Auto-scroll checkbox
        self.auto_scroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_frame, text="Auto Scroll", 
                       variable=self.auto_scroll).pack(side=tk.RIGHT)
    
    def log_message(self, message: str):
        """Add a message to the debug log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.debug_text.insert(tk.END, full_message)
        
        if self.auto_scroll.get():
            self.debug_text.see(tk.END)
        
        # Limit log size
        lines = int(self.debug_text.index('end-1c').split('.')[0])
        if lines > 500:
            self.debug_text.delete(1.0, "50.0")
    
    def clear_log(self):
        """Clear the debug log."""
        self.debug_text.delete(1.0, tk.END)
    
    def save_log(self):
        """Save the debug log to file."""
        import tkinter.filedialog
        import datetime
        
        filename = tkinter.filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialname=f"debug_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.debug_text.get(1.0, tk.END))
                print(f"Debug log saved to {filename}")
            except Exception as e:
                print(f"Error saving log: {e}")
