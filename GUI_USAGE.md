# GUI Usage Guide

This guide explains how to use the graphical interface for the Adaptive RF Receiver.

## Quick Start

### Command Line with GUI Option
```bash
# Run simple detector with GUI
python examples/simple_detection.py --gui

# Run dedicated GUI application
python examples/gui_example.py
```

### Configuration
The GUI uses the same configuration files as the command-line version:
- `config/system_config.yaml` - System settings
- `config/model_config.yaml` - ML model parameters

## GUI Features

### Main Window Components

1. **Control Panel**
   - Start/Stop Detection button
   - Learn Environment button (60-second calibration)
   - Save/Load Model buttons
   - Clear Plots button

2. **Status Panel**
   - Shows current system status (Ready/Learning/Detection Mode)
   - Displays device information (CUDA/CPU)

3. **Statistics Panel**
   - Real-time performance metrics:
     - Samples processed
     - Total detections
     - Current threshold
     - Processing FPS
     - GPU memory usage
     - Detection rate percentage

4. **Plot Area** (2x2 layout)
   - **Error Plot**: Reconstruction error vs. detection threshold
   - **Detection Events**: Vertical bars showing jamming detections
   - **I/Q Constellation**: Real-time constellation diagram

### Key Features

#### I/Q Constellation Plot
- **Purpose**: Visual representation of signal quality and jamming
- **Clean Signal**: Points clustered around origin or constellation points
- **Jammed Signal**: Scattered, distorted, or unusual patterns
- **Reference Circles**: Gray dashed circles at 1, 2, 3 unit radius
- **Statistics Overlay**: Shows I/Q standard deviation and power

#### Real-time Updates
- Plots update automatically during detection
- Performance-optimized updates (configurable interval)
- Auto-scaling axes based on signal levels

#### Model Management
- Save trained models with timestamp
- Load previously trained models
- Automatic model persistence during learning

## Usage Workflow

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_components.py
```

### 2. Start the GUI
```bash
python examples/gui_example.py --port 12345
```

### 3. Learning Phase
1. Ensure clean RF environment (no jamming)
2. Click "Learn Environment (60s)"
3. Wait for learning to complete
4. Threshold will be automatically set

### 4. Detection Phase
1. Click "Start Detection"
2. Monitor the constellation plot for signal changes
3. Watch for detection events on the plots
4. Red bars indicate jamming detected

### 5. Signal Analysis
- **Normal Signal**: Constellation points form clear patterns
- **Weak Jamming**: Slight constellation spreading
- **Strong Jamming**: Severe constellation distortion
- **Wideband Jamming**: Uniform constellation spreading
- **Narrowband Jamming**: Specific frequency effects

## Troubleshooting

### Common Issues

1. **No Data Received**
   - Check UDP port (default: 12345)
   - Verify HackRF/SDR is transmitting
   - Check network connectivity

2. **Poor Detection Performance**
   - Run learning phase in clean environment
   - Adjust threshold manually if needed
   - Check signal preprocessing settings

3. **GUI Performance Issues**
   - Increase plot update interval
   - Reduce constellation points displayed
   - Use CPU instead of GPU if memory limited

4. **Import Errors**
   - Install missing dependencies: `pip install matplotlib tkinter`
   - Ensure Python version >= 3.8
   - Check virtual environment activation

### Configuration Tips

```yaml
# For better GUI performance
performance:
  update_interval: 10  # Slower updates
  max_constellation_points: 500
  
# For better detection
threshold:
  safety_margin: 2.0  # More sensitive
  base_percentile: 98.0  # Lower threshold
```

## Advanced Features

### Debug Panel (Optional)
- Real-time log messages
- Error tracking
- Performance monitoring

### Advanced Controls (Optional)
- Manual threshold adjustment
- Force model updates
- Threshold reset

### Configuration Panel (Optional)
- Runtime parameter adjustment
- Learning duration control
- Update interval modification

## Keyboard Shortcuts

- **Ctrl+S**: Save model
- **Ctrl+O**: Load model
- **Ctrl+Q**: Quit application
- **Space**: Toggle detection
- **L**: Start learning phase

## Data Export

### Plot Export
```python
# Save current plots
gui.plot_manager.save_plots("detection_plots.png")
```

### Model Export
Models are automatically saved with timestamps in the `models/` directory.

## Integration with Original Code

The GUI is designed to be compatible with your original `original_adaptive.py` file:

```python
# Convert original detector to use new GUI
from gui.main_window import AdaptiveReceiverGUI

# Use existing detector instance
gui = AdaptiveReceiverGUI(your_existing_detector, port=12345)
gui.run()
```

## Performance Optimization

- Use GPU when available for model inference
- Adjust plot update intervals based on system performance
- Limit constellation points for real-time display
- Use efficient data structures (deques) for buffering
