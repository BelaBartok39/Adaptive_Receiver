# Configuration Guide

This document explains how to configure the Adaptive RF Receiver system.

## Configuration Files

The system uses two main configuration files:

### 1. `config/system_config.yaml`
Contains system-level settings:
- Network parameters (UDP port, buffer sizes)
- Device settings (CUDA/CPU preference)
- File paths and logging configuration
- Performance settings

### 2. `config/model_config.yaml`
Contains machine learning and signal processing parameters:
- Autoencoder model architecture
- Training parameters (learning rate, batch size, etc.)
- Threshold management settings
- Signal preprocessing options
- DRN classifier settings

## Usage

### Basic Usage
```python
from config.config_loader import load_detector_config

# Load configuration for the anomaly detector
config = load_detector_config()

# Initialize detector with config
detector = AnomalyDetector(window_size=1024, config=config)
```

### Running the Simple Detection Example
```bash
# Use default configuration
python examples/simple_detection.py

# Override specific parameters
python examples/simple_detection.py --port 54321 --window 2048
```

## Configuration Parameters

### Key Model Parameters
- `model.latent_dim`: Size of the autoencoder's latent space (default: 32)
- `training.learning_rate`: Learning rate for online training (default: 0.001)
- `training.weight_decay`: L2 regularization parameter (default: 0.0001)
- `threshold.base_percentile`: Percentile for anomaly threshold (default: 99.0)
- `preprocessing.normalization`: Normalization method ('robust', 'standard', 'minmax')

### Key System Parameters
- `network.udp_port`: UDP port for receiving I/Q data (default: 12345)
- `device.preferred`: Device preference ('cuda', 'cpu', 'auto')
- `signal.window_size`: Processing window size (default: 1024)

## Customization

You can create custom configuration files or override specific parameters:

```python
from config.config_loader import ConfigLoader

# Load custom config directory
loader = ConfigLoader("/path/to/custom/config")
config = loader.get_detector_config()

# Or modify loaded config
config = load_detector_config()
config['model']['latent_dim'] = 64  # Increase model capacity
config['training']['learning_rate'] = 0.0001  # Slower learning
```

## Environment Variables

You can also override config values using environment variables:
- `RF_RECEIVER_PORT`: Override UDP port
- `RF_RECEIVER_DEVICE`: Override device preference
- `RF_RECEIVER_MODEL_DIR`: Override model directory

## Troubleshooting

1. **Missing config files**: The system will use default values if config files are missing
2. **YAML parsing errors**: Check file syntax with a YAML validator
3. **Permission errors**: Ensure the config directory is readable
4. **Device errors**: Set `device.preferred: "cpu"` if CUDA is unavailable
