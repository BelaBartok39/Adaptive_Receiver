# RF Security System - Modular Architecture

This project implements an adaptive RF jamming detection system with a modular architecture designed for edge deployment on NVIDIA Jetson devices.

## Key Features

- **Modular Design**: Clean separation of concerns with reusable components
- **Edge-Optimized**: Mixed precision training and inference for Jetson GPUs
- **Adaptive Learning**: Dynamic threshold management that adapts to changing RF environments
- **RF-PUF Ready**: Architecture supports future integration of RF fingerprinting for device authentication
- **Real-time Processing**: Efficient streaming processing of RF I/Q data

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Signal    │────▶│   Feature   │────▶│  Anomaly    │
│   Buffer    │     │ Extraction  │     │  Detection  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    │ If Anomaly Detected
                    ▼
         ┌─────────────────┐     ┌─────────────────┐
         │ Jammer Type     │────▶│ Channel Change  │
         │ Classification  │     │ Recommendation  │
         └─────────────────┘     └─────────────────┘
```

## Core Modules

### 1. Detection (`core/detection/`)
- **anomaly_detector.py**: Main detection orchestrator
- **threshold_manager.py**: Dynamic threshold adaptation
- **jammer_classifier.py**: DRN-based jammer type classification (TODO)

### 2. Models (`core/models/`)
- **autoencoder.py**: Improved autoencoder with attention mechanism
- **drn_classifier.py**: Deep Residual Network for classification (TODO)
- **rf_puf.py**: RF-PUF authentication model (Future)

### 3. Preprocessing (`core/preprocessing/`)
- **signal_filters.py**: Signal conditioning and filtering
- **feature_extraction.py**: Spectral and temporal feature extraction

### 4. Network (`network/`)
- **receiver.py**: UDP packet reception and parsing
- **transmitter.py**: Channel change messaging (TODO)
- **protocols.py**: Communication protocols (TODO)

## Quick Start

```python
from core.detection.anomaly_detector import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(window_size=1024)

# Start learning phase
detector.start_learning(duration=60)

# Process I/Q data
is_anomaly, confidence, metrics = detector.detect(i_data, q_data)

# Save trained model
model_path = detector.save_model("my_model")
```

## RF-PUF Integration

The architecture is designed to support RF-PUF concepts from the research paper:

1. **Feature Extraction**: Extract device-specific RF signatures
   - I-Q imbalance
   - Frequency offset
   - Power amplifier characteristics

2. **Dual-Mode Operation**:
   - **Authentication Mode**: Identify legitimate devices
   - **Anomaly Mode**: Detect jamming/interference

3. **Benefits**:
   - No additional hardware at transmitter
   - Leverages existing RF imperfections
   - Strong PUF properties for security

## Next Steps

### Phase 1: Complete Core Implementation
- [x] Modular architecture design
- [x] Core detection modules
- [x] Signal preprocessing
- [x] Adaptive threshold management
- [ ] Complete test suite
- [ ] Performance benchmarking

### Phase 2: DRN Classifier
- [ ] Implement ResNet-based classifier
- [ ] Generate synthetic jammer dataset
- [ ] Train classifier for jammer types:
  - Narrowband
  - Wideband
  - Sweep
  - Pulse
  - Smart/Adaptive

### Phase 3: Channel Management
- [ ] Implement channel scanner
- [ ] Energy detection across channels
- [ ] Channel quality prediction
- [ ] HackRF control interface
- [ ] Hysteresis for stability

### Phase 4: RF-PUF Enhancement
- [ ] Device signature extraction
- [ ] Authentication database
- [ ] Multi-factor security
- [ ] Integration with anomaly detection

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# config/system_config.yaml
system:
  device: cuda
  window_size: 1024

detection:
  learning_duration: 60
  threshold_percentile: 99.0

model:
  latent_dim: 32
  use_attention: true

training:
  batch_size: 16
  learning_rate: 0.001
```

## Performance Optimizations

### Jetson-Specific
- Mixed precision (FP16) training and inference
- CUDA stream optimization
- TensorRT integration (planned)
- Efficient memory management

### Algorithm Optimizations
- Robust normalization using MAD
- Multi-scale error metrics
- Adaptive threshold with EMA
- Efficient batch processing

## Contributing

When adding new modules:
1. Follow the existing directory structure
2. Include comprehensive docstrings
3. Add unit tests in `tests/`
4. Update configuration schemas
5. Document any new dependencies

## License

[Your license here]

## References

- RF-PUF: "Enhancing IoT Security through Authentication of Wireless Nodes using In-situ Machine Learning"
- Adaptive threshold management techniques
- Deep residual networks for RF signal classification