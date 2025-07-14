Adaptive_Receiver/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── system_config.yaml
│   └── model_config.yaml
├── core/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── signal_filters.py      # Highpass, signal conditioning
│   │   └── feature_extraction.py   # I/Q processing, spectral features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py         # Improved RF Autoencoder
│   │   ├── drn_classifier.py      # Deep Residual Network for jammer types
│   │   └── rf_puf.py              # RF-PUF authentication (future)
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py    # Main detection logic
│   │   ├── threshold_manager.py   # Dynamic threshold management
│   │   └── jammer_classifier.py   # DRN-based classification
│   └── utils/
│       ├── __init__.py
│       ├── data_buffer.py         # Efficient data buffering
│       └── gpu_utils.py           # Jetson optimization utilities
├── network/
│   ├── __init__.py
│   ├── receiver.py                # UDP receiver, packet parsing
│   ├── transmitter.py             # Channel change messaging
│   └── protocols.py               # Message formats, protocols
├── gui/
│   ├── __init__.py
│   ├── main_window.py            # Main GUI application
│   ├── plots.py                  # Matplotlib plotting functions
│   └── widgets.py                # Custom GUI widgets
├── training/
│   ├── __init__.py
│   ├── train_autoencoder.py     # Training scripts
│   ├── train_drn.py              # DRN training
│   └── datasets/                 # Training data management
├── tests/
│   ├── __init__.py
│   ├── test_detection.py
│   ├── test_models.py
│   └── test_network.py
└── examples/
    ├── simple_detection.py       # Basic usage example
    └── full_system.py           # Complete system example


┌─────────────────┐     ┌──────────────────┐      ┌─────────────────┐
│   HackRF RX     │────▶│  Signal Buffer   │────▶│ Preprocessing   │
│  (UDP Stream)   │     │   & Windowing    │      │  & Features     │
└─────────────────┘     └──────────────────┘      └────────┬────────┘
                                                           │
                                ┌──────────────────────────▼────────┐
                                │      Anomaly Detection Layer      │
                                │  ┌──────────────┐ ┌─────────────┐ │
                                │  │ Autoencoder  │ │  Threshold  │ │
                                │  │   (Normal?)  │ │  Manager    │ │
                                │  └──────┬───────┘ └─────────────┘ │
                                └─────────┼─────────────────────────┘
                                         │ If Anomaly Detected
                                ┌────────▼─────────┐
                                │  DRN Classifier  │
                                │  (Jammer Type)   │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐     ┌──────────────┐
                                │ Channel Scanner  │────▶│ HackRF TX   │
                                │ (Find Clean Ch.) │     │ (Change Ch.) │
                                └──────────────────┘     └──────────────┘