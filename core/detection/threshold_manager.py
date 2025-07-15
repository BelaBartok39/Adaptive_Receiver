"""
Dynamic threshold management for anomaly detection.
Fixed to properly retain and use learned thresholds.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple


class DynamicThresholdManager:
    """
    Manages adaptive thresholds for anomaly detection with proper learning retention.
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
        self.learning_errors = deque(maxlen=window_size * 10)  # Keep more learning samples
        
        # State tracking
        self.is_learning = True
        self.stable_threshold = None
        self.threshold_calculated = False
        
        # Statistics
        self.learning_mean = None
        self.learning_std = None
        
        # Statistics cache
        self._stats_cache = {}
        self._cache_valid = False
        
        print(f"Threshold manager initialized with config: {self.config}")
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'percentile': 99.0,  # Use 99th percentile for threshold
            'margin_multiplier': 1.5,  # Multiplier for margin above mean
            'min_samples': 100,  # Minimum samples before calculating threshold
            'outlier_factor': 3.0  # Factor for outlier detection
        }
    
    def update(self, error: float, is_learning: bool = None) -> None:
        """
        Update threshold with new error value.
        
        Args:
            error: Current reconstruction error
            is_learning: Override learning state (optional)
        """
        # Use provided learning state or current state
        learning = is_learning if is_learning is not None else self.is_learning
        
        # Always add to general errors
        self.errors.append(error)
        self._cache_valid = False
        
        # Add to learning errors if in learning mode
        if learning:
            self.learning_errors.append(error)
    
    def get_threshold(self) -> float:
        """
        Get current threshold value.
        
        Returns:
            Current threshold or inf if not yet calculated
        """
        # During learning, return infinity
        if self.is_learning:
            return float('inf')
        
        # After learning, return the stable threshold
        if self.stable_threshold is not None:
            return self.stable_threshold
        
        # If no threshold set yet, return infinity
        return float('inf')
    
    def calculate_threshold_from_learning(self) -> float:
        """
        Calculate threshold from learning data.
        
        Returns:
            Calculated threshold value
        """
        if len(self.learning_errors) < self.config['min_samples']:
            print(f"Warning: Only {len(self.learning_errors)} samples for threshold calculation")
            return float('inf')
        
        errors_array = np.array(self.learning_errors)
        
        # Remove outliers using IQR method
        q1 = np.percentile(errors_array, 25)
        q3 = np.percentile(errors_array, 75)
        iqr = q3 - q1
        outlier_factor = self.config['outlier_factor']
        lower_bound = q1 - outlier_factor * iqr
        upper_bound = q3 + outlier_factor * iqr
        
        # Filter outliers
        filtered_errors = errors_array[(errors_array >= lower_bound) & (errors_array <= upper_bound)]
        
        if len(filtered_errors) < 10:
            filtered_errors = errors_array  # Use all if too few remain
        
        # Calculate statistics on filtered data
        self.learning_mean = np.mean(filtered_errors)
        self.learning_std = np.std(filtered_errors)
        
        # Calculate threshold using percentile method
        percentile_value = np.percentile(filtered_errors, self.config['percentile'])
        
        # Alternative: mean + k*std method
        margin = self.config['margin_multiplier']
        std_threshold = self.learning_mean + margin * self.learning_std
        
        # Use the maximum of both methods for safety
        threshold = max(percentile_value, std_threshold)
        
        print(f"Threshold calculation:")
        print(f"  - Samples: {len(self.learning_errors)} total, {len(filtered_errors)} after filtering")
        print(f"  - Mean: {self.learning_mean:.4f}, Std: {self.learning_std:.4f}")
        print(f"  - {self.config['percentile']}th percentile: {percentile_value:.4f}")
        print(f"  - Mean + {margin}*std: {std_threshold:.4f}")
        print(f"  - Final threshold: {threshold:.4f}")
        
        return threshold
    
    def set_learning_mode(self, is_learning: bool) -> None:
        """
        Set learning mode and calculate threshold when exiting learning.
        
        Args:
            is_learning: Whether to enable learning mode
        """
        # If transitioning from learning to detection
        if self.is_learning and not is_learning:
            # Calculate and set the threshold
            self.stable_threshold = self.calculate_threshold_from_learning()
            self.threshold_calculated = True
            print(f"Learning mode ended. Threshold set to: {self.stable_threshold:.4f}")
        
        # If starting learning mode
        elif not self.is_learning and is_learning:
            # Clear learning data for fresh start
            self.learning_errors.clear()
            self.stable_threshold = None
            self.threshold_calculated = False
            self.learning_mean = None
            self.learning_std = None
            print("Learning mode started. Clearing previous data.")
        
        self.is_learning = is_learning
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive threshold statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self._cache_valid and len(self.errors) >= 10:
            errors_array = np.array(list(self.errors))
            
            # Basic statistics
            stats = {
                'mean': np.mean(errors_array),
                'std': np.std(errors_array),
                'median': np.median(errors_array),
                'min': np.min(errors_array),
                'max': np.max(errors_array),
                'p25': np.percentile(errors_array, 25),
                'p50': np.percentile(errors_array, 50),
                'p75': np.percentile(errors_array, 75),
                'p90': np.percentile(errors_array, 90),
                'p95': np.percentile(errors_array, 95),
                'p99': np.percentile(errors_array, 99),
                'current_threshold': self.get_threshold(),
                'n_samples': len(self.errors),
                'n_learning_samples': len(self.learning_errors),
                'is_learning': self.is_learning,
                'threshold_calculated': self.threshold_calculated
            }
            
            # Add learning statistics if available
            if self.learning_mean is not None:
                stats['learning_mean'] = self.learning_mean
                stats['learning_std'] = self.learning_std
            
            self._stats_cache = stats
            self._cache_valid = True
        
        return self._stats_cache
    
    def get_confidence(self, error: float) -> float:
        """
        Get confidence score for an error value.
        
        Args:
            error: Error value to evaluate
            
        Returns:
            Confidence score (0-1, where 1 is high confidence in anomaly)
        """
        threshold = self.get_threshold()
        if threshold == float('inf') or error <= threshold:
            return 0.0
        
        # Calculate how far above threshold
        if self.learning_std is not None and self.learning_std > 0:
            # Normalize by learning standard deviation
            z_score = (error - threshold) / self.learning_std
            # Sigmoid-like scaling
            confidence = 1.0 / (1.0 + np.exp(-2 * z_score))
        else:
            # Simple ratio-based confidence
            ratio = error / threshold
            confidence = min(1.0, (ratio - 1.0) * 2)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def should_trigger_alert(self, error: float, history_size: int = 5) -> bool:
        """
        Determine if an alert should be triggered.
        
        Args:
            error: Current error value
            history_size: Number of recent samples to consider
            
        Returns:
            Whether to trigger an alert
        """
        # Don't alert during learning
        if self.is_learning:
            return False
        
        threshold = self.get_threshold()
        if threshold == float('inf'):
            return False
        
        # Check if current error exceeds threshold significantly
        if error <= threshold:
            return False
        
        # For strong anomalies, alert immediately
        if error > threshold * 1.5:
            return True
        
        # For weaker anomalies, check persistence
        if len(self.errors) >= history_size:
            recent_errors = list(self.errors)[-history_size:]
            anomalous_count = sum(1 for e in recent_errors if e > threshold)
            return anomalous_count >= history_size // 2
        
        return error > threshold
    
    def reset(self) -> None:
        """Reset all statistics and thresholds."""
        self.errors.clear()
        self.learning_errors.clear()
        self.stable_threshold = None
        self.threshold_calculated = False
        self.is_learning = True
        self.learning_mean = None
        self.learning_std = None
        self._cache_valid = False
        print("Threshold manager reset")