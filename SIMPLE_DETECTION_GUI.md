# GUI Integration Guide

Your GUI is now properly connected to work with `simple_detection.py`! Here's how to use it:

## Quick Start

### Run with GUI
```bash
python examples/simple_detection.py --gui
```

### Run without GUI (command line only)
```bash
python examples/simple_detection.py --no-gui
```

## How It Works

The GUI integration works by:

1. **Automatic Detection**: The GUI automatically detects that it's being passed a `SimpleJammingDetector` instance
2. **Hook Integration**: It hooks into the detector's `_process_window()` method to capture data for plotting
3. **Shared Processing**: The `SimpleJammingDetector` handles all networking and data processing
4. **Real-time Visualization**: The GUI displays the processed data in real-time plots

## GUI Features Available

### Control Panel
- **Start Detection**: Enables GUI monitoring (your SimpleJammingDetector continues running)
- **Learn Environment**: Triggers the 60-second learning phase
- **Save/Load Model**: Uses your detector's model persistence
- **Clear Plots**: Clears the visualization data

### Real-time Plots
- **Error Plot**: Shows reconstruction error vs. detection threshold
- **Detection Events**: Red bars indicate jamming detected
- **I/Q Constellation**: Live constellation plot showing signal quality

### Statistics Panel
- Samples processed
- Total detections
- Current threshold
- Processing FPS
- Detection rate

## Key Benefits

1. **Non-Intrusive**: The GUI doesn't interfere with your existing `SimpleJammingDetector` logic
2. **Real-time Monitoring**: See jamming detection in action with live plots
3. **Signal Quality Assessment**: The constellation plot helps you visually confirm jamming
4. **Performance Monitoring**: Track system performance and detection rates

## Example Usage

```bash
# Start with GUI and custom parameters
python examples/simple_detection.py --gui --port 54321 --window 2048

# The GUI will show:
# - Your detector's configuration on startup
# - Real-time I/Q constellation (great for seeing jamming!)
# - Error plots with threshold visualization
# - Performance statistics
```

## Constellation Plot Usage

The I/Q constellation plot is particularly useful for jamming detection:

- **Clean Signal**: Points form clear patterns (clustered or organized)
- **Jammed Signal**: Points become scattered, distorted, or show unusual patterns
- **Real-time Updates**: You can see jamming happen in real-time
- **Reference Circles**: Gray circles help assess signal strength

## Integration Details

The GUI creates a wrapper around your `SimpleJammingDetector` that:
- Uses your existing UDP socket and data processing
- Hooks into your `_process_window()` method to capture plot data
- Shares the same `AnomalyDetector` instance for model operations
- Preserves all your original detector functionality

Your `simple_detection.py` continues to work exactly as before - the GUI just adds visualization on top!
