"""
Core anomaly detection module using VAE.
Fixed to properly detect anomalies after learning phase.
"""

import torch
import torch.optim as optim
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import time
import json

from ..models.autoencoder import ImprovedRFAutoencoder
from ..preprocessing.signal_filters import SignalPreprocessor
from ..detection.threshold_manager import DynamicThresholdManager


class AnomalyDetector:
    """
    Main anomaly detection class using VAE with proper threshold management.
    """
    
    def __init__(self, window_size: int = 1024, config: Optional[Dict] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            window_size: Size of input windows
            config: Optional configuration dictionary
        """
        self.window_size = window_size
        self.config = config or self._default_config()
        
        # Device selection
        self.device = self._select_device()
        print(f"Anomaly detector using device: {self.device}")
        
        # Initialize components
        self.model = self._build_model()
        self.preprocessor = SignalPreprocessor(self.config.get('preprocessing', {}))
        self.threshold_manager = DynamicThresholdManager(
            window_size=1000,
            config=self.config.get('threshold', {})
        )
        
        # Optimizer and scaler for mixed precision
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Use new GradScaler API to avoid deprecation warning
        if self.device.type == 'cuda':
            try:
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
            except ImportError:
                # Fallback for older PyTorch versions
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # State tracking
        self.is_learning = False
        self.learning_start_time = None
        self.learning_duration = None
        self.sample_count = 0
        self.total_detections = 0
        
        # Detection history
        self.detection_history = []
        self.max_history = 100
        
        # Model persistence
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Batch processing for efficiency
        self.batch_buffer = []
        self.batch_size = self.config['training'].get('batch_size', 32)
        
        print(f"Detector initialized with window size: {window_size}")
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'model': {
                'latent_dim': 32,
                'beta': 1.0
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'update_frequency': 10
            },
            'preprocessing': {
                'normalization': 'robust',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'threshold': {
                'percentile': 99.0,
                'margin_multiplier': 1.5,
                'min_samples': 100
            }
        }
    
    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if torch.cuda.is_available():
            # Get GPU properties
            props = torch.cuda.get_device_properties(0)
            print(f"GPU detected: {props.name} ({props.total_memory // 1024**2} MB)")
            return torch.device('cuda')
        else:
            print("No GPU available, using CPU")
            return torch.device('cpu')
    
    def _build_model(self) -> ImprovedRFAutoencoder:
        """Build and initialize the model."""
        model = ImprovedRFAutoencoder(
            input_size=self.window_size,
            latent_dim=self.config['model']['latent_dim'],
            beta=self.config['model'].get('beta', 1.0)
        )
        model.to(self.device)
        
        # Enable mixed precision if available
        if self.device.type == 'cuda':
            model.half()
        
        return model
    
    def detect(self, i_data: np.ndarray, q_data: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect anomalies in I/Q data.
        
        Args:
            i_data: In-phase component
            q_data: Quadrature component
            
        Returns:
            Tuple of (is_anomaly, confidence, metrics)
        """
        self.sample_count += 1
        
        # Preprocess data
        iq_tensor = self.preprocessor.preprocess_iq(i_data, q_data)
        
        # Get model output
        with torch.no_grad():
            if self.device.type == 'cuda':
                try:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        anomaly_score = self.model.get_anomaly_score(iq_tensor)
                except ImportError:
                    # Fallback for older PyTorch versions
                    from torch.cuda.amp import autocast
                    with autocast():
                        anomaly_score = self.model.get_anomaly_score(iq_tensor)
            else:
                anomaly_score = self.model.get_anomaly_score(iq_tensor)
        
        # Convert to scalar
        error = float(anomaly_score.cpu().numpy()[0])
        
        # Update threshold manager
        self.threshold_manager.update(error, is_learning=self.is_learning)
        
        # During learning, always update model
        if self.is_learning:
            self.batch_buffer.append(iq_tensor)
            if len(self.batch_buffer) >= self.batch_size:
                self._update_model_batch()
        
        # Get detection threshold
        threshold = self.threshold_manager.get_threshold()
        
        # Determine if anomaly
        is_anomaly = False
        confidence = 0.0
        
        if not self.is_learning and threshold != float('inf'):
            # Check if error exceeds threshold
            is_anomaly = error > threshold
            
            # Use threshold manager's alert logic for better accuracy
            should_alert = self.threshold_manager.should_trigger_alert(error)
            
            if is_anomaly:
                confidence = self.threshold_manager.get_confidence(error)
                self.total_detections += 1
                
                # Only set is_anomaly if alert should be triggered
                is_anomaly = should_alert
        
        # Build metrics
        metrics = {
            'error': error,
            'threshold': threshold if threshold != float('inf') else None,
            'is_learning': self.is_learning,
            'sample_count': self.sample_count
        }
        
        # Add to history
        detection_entry = {
            'timestamp': time.time(),
            'error': error,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'confidence': confidence
        }
        
        self.detection_history.append(detection_entry)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return is_anomaly, confidence, metrics
    
    def _update_model_batch(self):
        """Update model with accumulated batch."""
        if not self.batch_buffer:
            return
        
        # Stack tensors into batch
        batch = torch.cat(self.batch_buffer, dim=0)
        self.batch_buffer.clear()
        
        # Forward pass
        self.model.train()
        
        if self.device.type == 'cuda' and self.scaler:
            try:
                from torch.amp import autocast
                with autocast('cuda'):
                    reconstruction, mu, logvar = self.model(batch)
                    loss_dict = self.model.loss_function(batch, reconstruction, mu, logvar)
                    loss = loss_dict['loss']
            except ImportError:
                # Fallback for older PyTorch versions
                from torch.cuda.amp import autocast
                with autocast():
                    reconstruction, mu, logvar = self.model(batch)
                    loss_dict = self.model.loss_function(batch, reconstruction, mu, logvar)
                    loss = loss_dict['loss']
            
            # Backward pass with mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            reconstruction, mu, logvar = self.model(batch)
            loss_dict = self.model.loss_function(batch, reconstruction, mu, logvar)
            loss = loss_dict['loss']
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.model.eval()
    
    def start_learning(self, duration: int):
        """
        Start learning phase.
        
        Args:
            duration: Learning duration in seconds
        """
        print(f"Starting learning phase for {duration} seconds...")
        self.is_learning = True
        self.learning_start_time = time.time()
        self.learning_duration = duration
        
        # Reset threshold manager for fresh learning
        self.threshold_manager.set_learning_mode(True)
        
        # Clear detection history
        self.detection_history.clear()
        self.total_detections = 0
    
    def stop_learning(self) -> Dict:
        """
        Stop learning phase and calculate threshold.
        
        Returns:
            Learning statistics
        """
        if not self.is_learning:
            return {}
        
        # Process any remaining batch
        if self.batch_buffer:
            self._update_model_batch()
        
        # Stop learning mode
        self.is_learning = False
        self.threshold_manager.set_learning_mode(False)
        
        # Get final statistics
        stats = self.threshold_manager.get_statistics()
        
        learning_stats = {
            'duration': time.time() - self.learning_start_time,
            'samples_processed': self.sample_count,
            'final_threshold': stats.get('current_threshold', float('inf')),
            'mean_error': stats.get('learning_mean', 0),
            'std_error': stats.get('learning_std', 0)
        }
        
        print(f"Learning complete. Threshold: {learning_stats['final_threshold']:.4f}")
        
        return learning_stats
    
    def get_status(self) -> Dict:
        """Get current detector status."""
        status = {
            'mode': 'learning' if self.is_learning else 'detection',
            'sample_count': self.sample_count,
            'total_detections': self.total_detections,
            'device': str(self.device)
        }
        
        if self.is_learning and self.learning_start_time:
            elapsed = time.time() - self.learning_start_time
            remaining = max(0, self.learning_duration - elapsed)
            progress = min(100, (elapsed / self.learning_duration) * 100)
            
            status.update({
                'learning_elapsed': elapsed,
                'remaining_time': remaining,
                'progress': progress
            })
        
        # Add threshold info
        threshold_stats = self.threshold_manager.get_statistics()
        status['threshold_info'] = {
            'current': threshold_stats.get('current_threshold', float('inf')),
            'is_calculated': threshold_stats.get('threshold_calculated', False)
        }
        
        return status
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save model and configuration.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to saved model
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rf_anomaly_model_{timestamp}.pth"
        
        filepath = self.model_dir / filename
        
        # Save model state and configuration
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'window_size': self.window_size,
            'sample_count': self.sample_count,
            'threshold_stats': self.threshold_manager.get_statistics()
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update configuration if available
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # Restore threshold if available
        if 'threshold_stats' in checkpoint:
            stats = checkpoint['threshold_stats']
            if 'current_threshold' in stats and stats['current_threshold'] != float('inf'):
                self.threshold_manager.stable_threshold = stats['current_threshold']
                self.threshold_manager.threshold_calculated = True
                self.threshold_manager.is_learning = False
                print(f"Loaded threshold: {stats['current_threshold']:.4f}")
        
        print(f"Model loaded from {filepath}")