"""
Dynamic threshold management for anomaly detection.
Implements sophisticated threshold adaptation strategies.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple


class DynamicThresholdManager:
    """
    Manages adaptive thresholds for anomaly detection with multiple strategies.
    
    This manager maintains separate statistics for learning and detection phases,
    implementing smooth transitions and safety bounds to prevent false positives.
    """
    
    def __init__(self, window_size: int = 1000, config: Optional[Dict] = None):
        """
        Initialize threshold manager.
        
        Args:
            window_size: Size of the sliding window for statistics
            config: Optional configuration dictionary
        """
        self.window_size = window_size
        self.config = config or self._default_config()
        
        # Error tracking
        self.errors = deque(maxlen=window_size)
        self.learning_errors = deque(maxlen=window_size)
        self.detection_errors = deque(maxlen=window_size)
        
        # State tracking
        self.is_learning = True
        self.stable_threshold = None
        self.ema_threshold = None
        
        # Statistics cache
        self._stats_cache = {}
        self._cache_valid = False
        
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            # Use lower percentile and margin for more sensitivity
            'base_percentile': 95.0,
            'adaptation_rate': 0.01,
            # Require fewer samples before threshold becomes active
            'min_samples': 50,
            # Increase EMA alpha for faster adaptation
            'ema_alpha': 0.1,
            # Remove extra margin to follow waveform closely
            'safety_margin': 1.0,
            # Allow threshold to adapt down to half of stable value
            'min_threshold_ratio': 0.5
        }
    
    def update(self, error: float, is_learning: bool = False) -> None:
        """
        Update threshold with new error value.
        
        Args:
            error: Current reconstruction error
            is_learning: Whether system is in learning phase
        """
        self.errors.append(error)
        self._cache_valid = False
        
        if is_learning:
            self.learning_errors.append(error)
            self.is_learning = True
        else:
            self.detection_errors.append(error)
            self.is_learning = False
    
    def get_threshold(self) -> float:
        """
        Get current threshold using adaptive strategy.
        
        Returns:
            Current threshold value
        """
        if len(self.errors) < self.config['min_samples']:
            return float('inf')
        
        # During learning, use high percentile with margin
        if self.is_learning:
            if len(self.learning_errors) > 50:
                threshold = np.percentile(
                    self.learning_errors, 
                    self.config['base_percentile']
                )
                self.stable_threshold = threshold * self.config['safety_margin']
                return self.stable_threshold
            return float('inf')
        
        # During detection, use adaptive threshold
        if self.stable_threshold is None:
            return float('inf')
        
        # Calculate EMA threshold for smooth adaptation
        recent_high = np.percentile(
            list(self.errors)[-100:], 
            95
        )
        
        if self.ema_threshold is None:
            self.ema_threshold = self.stable_threshold
        else:
            # Exponential moving average
            alpha = self.config['ema_alpha']
            self.ema_threshold = (
                (1 - alpha) * self.ema_threshold + 
                alpha * recent_high * 1.3
            )
        
        # Ensure threshold doesn't drop too low
        min_threshold = self.stable_threshold * self.config['min_threshold_ratio']
        return max(self.ema_threshold, min_threshold)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive threshold statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self._cache_valid and len(self.errors) >= 10:
            errors_array = np.array(list(self.errors))
            self._stats_cache = {
                'mean': np.mean(errors_array),
                'std': np.std(errors_array),
                'median': np.median(errors_array),
                'p25': np.percentile(errors_array, 25),
                'p50': np.percentile(errors_array, 50),
                'p75': np.percentile(errors_array, 75),
                'p90': np.percentile(errors_array, 90),
                'p95': np.percentile(errors_array, 95),
                'p99': np.percentile(errors_array, 99),
                'current_threshold': self.get_threshold(),
                'n_samples': len(self.errors),
                'is_learning': self.is_learning
            }
            self._cache_valid = True
        
        return self._stats_cache
    
    def reset(self) -> None:
        """Reset all statistics and thresholds."""
        self.errors.clear()
        self.learning_errors.clear()
        self.detection_errors.clear()
        self.stable_threshold = None
        self.ema_threshold = None
        self.is_learning = True
        self._cache_valid = False
    
    def set_learning_mode(self, is_learning: bool) -> None:
        """
        Explicitly set learning mode.
        
        Args:
            is_learning: Whether to enable learning mode
        """
        self.is_learning = is_learning
        if not is_learning and self.stable_threshold is None:
            # Force threshold calculation if transitioning from learning
            _ = self.get_threshold()
    
    def get_confidence(self, error: float) -> float:
        """
        Get confidence score for an error value.
        
        Args:
            error: Error value to evaluate
            
        Returns:
            Confidence score (0-1, where 1 is high confidence in anomaly)
        """
        threshold = self.get_threshold()
        if threshold == float('inf'):
            return 0.0
        
        if error <= threshold:
            return 0.0
        
        # Sigmoid-like confidence scaling
        ratio = error / threshold
        confidence = 1.0 - np.exp(-2 * (ratio - 1))
        return np.clip(confidence, 0.0, 1.0)
    
    def should_trigger_alert(self, error: float, history_size: int = 5) -> bool:
        """
        Determine if an alert should be triggered based on error history.
        
        Args:
            error: Current error value
            history_size: Number of recent samples to consider
            
        Returns:
            Whether to trigger an alert
        """
        if self.is_learning:
            return False
        
        threshold = self.get_threshold()
        if threshold == float('inf'):
            return False
        
        # Check if current error exceeds threshold
        if error <= threshold:
            return False
        
        # Check if recent errors also indicate anomaly (reduces false positives)
        if len(self.errors) >= history_size:
            recent_errors = list(self.errors)[-history_size:]
            anomalous_count = sum(1 for e in recent_errors if e > threshold)
            return anomalous_count >= history_size // 2
        
        return True