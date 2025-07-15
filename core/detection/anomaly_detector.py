"""
Main anomaly detection module that orchestrates the detection pipeline.
Integrates preprocessing, autoencoder, and threshold management.
"""

import torch
import torch.cuda.amp as amp
import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional, List
import time
import os
import json
import datetime

from ..models.autoencoder import ImprovedRFAutoencoder
from ..preprocessing.signal_filters import SignalPreprocessor
from .threshold_manager import DynamicThresholdManager


class AnomalyDetector:
    """
    Complete anomaly detection system for RF signals using VAE.
    Optimized for edge deployment on Jetson devices.
    """
    
    def __init__(self, 
                 window_size: int = 1024,
                 device: Optional[str] = None,
                 config: Optional[dict] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            window_size: Size of signal windows to process
            device: Device to run on ('cuda', 'cpu', or None for auto)
            config: Optional configuration dictionary
        """
        self.window_size = window_size
        self.config = config or self._default_config()
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.preprocessor = SignalPreprocessor(self.config.get('preprocessing', {}))
        
        # Initialize VAE model
        self.model = ImprovedRFAutoencoder(
            window_size, 
            latent_dim=self.config['model']['latent_dim'],
            beta=self.config['model'].get('beta', 1.0)
        ).to(self.device)
        
        # Use half precision on CUDA devices
        if self.device.type == 'cuda':
            self.model.half()
        
        # Optimizer and scaler for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Threshold manager
        self.threshold_manager = DynamicThresholdManager(
            window_size=self.config['threshold']['window_size'],
            config=self.config['threshold']
        )
        
        # State tracking
        self.is_learning = False
        self.sample_count = 0
        self.total_detections = 0
        self.session_start = time.time()
        
        # Batch buffer for online learning
        self.batch_buffer = deque(maxlen=self.config['training']['batch_buffer_size'])
        
        # History tracking
        self.detection_history = deque(maxlen=1000)
        self.feature_history = deque(maxlen=1000)
        
        # Model persistence
        self.model_dir = self.config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _default_config(self) -> dict:
        """Get default configuration."""
        return {
            'model': {
                'latent_dim': 32,
                'beta': 1.0  # Beta for beta-VAE
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'batch_buffer_size': 32,
                'update_interval': 100,
                'batch_size': 16
            },
            'threshold': {
                'window_size': 1000,
                'base_percentile': 99.0,
                'adaptation_rate': 0.01,
                'min_samples': 100,
                'ema_alpha': 0.05,
                'safety_margin': 1.5,
                'min_threshold_ratio': 0.8
            },
            'preprocessing': {
                'normalization': 'robust',
                'clip_sigma': 5.0
            },
            'model_dir': 'models'
        }
    
    @torch.amp.autocast('cuda')
    def detect(self, i_data: np.ndarray, q_data: np.ndarray) -> Tuple[bool, float, dict]:
        """
        Detect anomalies in I/Q data using VAE.
        
        Args:
            i_data: In-phase component
            q_data: Quadrature component
            
        Returns:
            Tuple of (is_anomaly, confidence, metrics)
        """
        # Preprocess data
        iq_tensor = self.preprocessor.preprocess_iq(i_data, q_data)
        
        # Extract additional features
        spectral_features = self.preprocessor.extract_spectral_features(i_data, q_data)
        temporal_features = self.preprocessor.extract_temporal_features(i_data, q_data)
        rf_puf_features = self.preprocessor.extract_rf_puf_features(i_data, q_data)
        
        # Forward pass through VAE
        self.model.eval()
        with torch.no_grad():
            # Get anomaly score from VAE
            anomaly_score = self.model.get_anomaly_score(iq_tensor)
            
            # Get reconstruction for additional metrics
            reconstruction, mu, logvar = self.model(iq_tensor)
            
            # Calculate various error metrics
            mse_error = torch.mean((iq_tensor - reconstruction) ** 2)
            mae_error = torch.mean(torch.abs(iq_tensor - reconstruction))
            
            # Use anomaly score as primary detection metric
            error = anomaly_score.item()
        
        # Update threshold manager
        self.threshold_manager.update(error, self.is_learning)
        
        # Get detection result
        is_anomaly = self.threshold_manager.should_trigger_alert(error)
        confidence = self.threshold_manager.get_confidence(error)
        
        # Update statistics
        self.sample_count += 1
        if is_anomaly and not self.is_learning:
            self.total_detections += 1
        
        # Store history
        self.detection_history.append({
            'timestamp': time.time(),
            'error': error,
            'is_anomaly': is_anomaly,
            'confidence': confidence
        })
        
        self.feature_history.append({
            **spectral_features,
            **temporal_features,
            **rf_puf_features
        })
        
        # Conditional batch update during learning
        if self.is_learning:
            self.batch_buffer.append(iq_tensor)
            if (len(self.batch_buffer) >= self.config['training']['batch_size'] and 
                self.sample_count % self.config['training']['update_interval'] == 0):
                self._update_model_batch()
        
        # Compile metrics
        metrics = {
            'error': error,
            'anomaly_score': error,
            'mse_error': mse_error.item(),
            'mae_error': mae_error.item(),
            'threshold': self.threshold_manager.get_threshold(),
            'latent_mean': torch.mean(mu).item(),
            'latent_std': torch.std(mu).item(),
            **spectral_features,
            **temporal_features,
            **rf_puf_features
        }
        
        return is_anomaly, confidence, metrics
    
    def _update_model_batch(self) -> None:
        """Perform batch update of the VAE model during learning."""
        if len(self.batch_buffer) < self.config['training']['batch_size']:
            return
        
        self.model.train()
        
        # Create batch
        batch_size = self.config['training']['batch_size']
        batch = torch.cat(list(self.batch_buffer)[:batch_size], dim=0)
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            # Forward pass
            reconstruction, mu, logvar = self.model(batch)
            
            # Calculate VAE loss
            losses = self.model.loss_function(batch, reconstruction, mu, logvar)
            loss = losses['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Clear some of the buffer
        for _ in range(batch_size // 2):
            if self.batch_buffer:
                self.batch_buffer.popleft()
    
    def start_learning(self, duration: float = 60.0) -> None:
        """
        Start learning phase.
        
        Args:
            duration: Learning duration in seconds
        """
        self.is_learning = True
        self.learning_start = time.time()
        self.learning_duration = duration
        self.threshold_manager.set_learning_mode(True)
        print(f"Learning phase started for {duration} seconds")
    
    def stop_learning(self) -> dict:
        """
        Stop learning phase and return statistics.
        
        Returns:
            Learning statistics
        """
        self.is_learning = False
        self.threshold_manager.set_learning_mode(False)
        
        stats = self.threshold_manager.get_statistics()
        learning_time = time.time() - self.learning_start
        
        print(f"Learning complete. Processed {self.sample_count} samples")
        
        return {
            'learning_duration': learning_time,
            'samples_processed': self.sample_count,
            'threshold_stats': stats
        }
    
    def get_status(self) -> dict:
        """
        Get current detector status.
        
        Returns:
            Status dictionary
        """
        if self.is_learning:
            elapsed = time.time() - self.learning_start
            remaining = max(0, self.learning_duration - elapsed)
            if remaining == 0:
                self.stop_learning()
            
            return {
                'mode': 'learning',
                'remaining_time': remaining,
                'samples_processed': self.sample_count
            }
        else:
            return {
                'mode': 'detection',
                'total_detections': self.total_detections,
                'samples_processed': self.sample_count,
                'session_duration': time.time() - self.session_start
            }
    
    def save_model(self, name: Optional[str] = None) -> str:
        """
        Save the current model and state.
        
        Args:
            name: Optional model name
            
        Returns:
            Path to saved model
        """
        if name is None:
            name = f"vae_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        path = os.path.join(self.model_dir, f"{name}.pth")
        
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'threshold_stats': self.threshold_manager.get_statistics(),
            'config': self.config,
            'sample_count': self.sample_count,
            'total_detections': self.total_detections,
            'feature_history': list(self.feature_history)[-100:]  # Save recent features
        }, path)
        
        print(f"Model saved to {path}")
        return path
    
    def load_model(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        # Restore other state
        if 'sample_count' in checkpoint:
            self.sample_count = checkpoint['sample_count']
        if 'total_detections' in checkpoint:
            self.total_detections = checkpoint['total_detections']
        
        # Restore threshold manager state if available
        if 'threshold_stats' in checkpoint:
            stats = checkpoint['threshold_stats']
            if 'current_threshold' in stats:
                self.threshold_manager.stable_threshold = stats['current_threshold']
                
        print(f"Model loaded from {path}")