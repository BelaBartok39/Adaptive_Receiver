"""
Channel scanner for finding clean frequencies.
This is a skeleton implementation for Phase 3.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import time


class ChannelScanner:
    """
    Scans RF channels to find clean frequencies for communication.
    
    This scanner can work in parallel with the main detection system
    to maintain an up-to-date map of channel conditions.
    """
    
    # Standard 2.4 GHz WiFi channels (MHz)
    WIFI_24_CHANNELS = {
        1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432,
        6: 2437, 7: 2442, 8: 2447, 9: 2452, 10: 2457,
        11: 2462, 12: 2467, 13: 2472, 14: 2484
    }
    
    def __init__(self, 
                 current_channel: int = 2412,
                 scan_bandwidth: float = 20.0,
                 config: Optional[Dict] = None):
        """
        Initialize channel scanner.
        
        Args:
            current_channel: Current operating frequency (MHz)
            scan_bandwidth: Bandwidth to scan around each channel (MHz)
            config: Optional configuration
        """
        self.current_channel = current_channel
        self.scan_bandwidth = scan_bandwidth
        self.config = config or self._default_config()
        
        # Channel quality history
        self.channel_history = {
            freq: deque(maxlen=self.config['history_size']) 
            for freq in self.WIFI_24_CHANNELS.values()
        }
        
        # Last scan results
        self.last_scan = {}
        self.last_scan_time = 0
        
        # Channel switching history (for hysteresis)
        self.switch_history = deque(maxlen=10)
        self.last_switch_time = 0
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'history_size': 100,
            'scan_interval': 5.0,  # seconds
            'energy_threshold': -80,  # dBm
            'quality_threshold': 0.8,
            'min_switch_interval': 10.0,  # seconds
            'hysteresis_factor': 1.2
        }
    
    def scan_channel(self, frequency: int, duration: float = 0.1) -> Dict[str, float]:
        """
        Scan a single channel for interference.
        
        Args:
            frequency: Channel frequency in MHz
            duration: Scan duration in seconds
            
        Returns:
            Dictionary of channel metrics
        """
        # TODO: Implement actual channel scanning
        # This would interface with HackRF to measure:
        # - Energy level
        # - Spectral occupancy
        # - Interference patterns
        
        # Placeholder implementation
        metrics = {
            'frequency': frequency,
            'energy_dbm': np.random.normal(-85, 5),
            'occupancy': np.random.uniform(0, 0.3),
            'interference_score': np.random.uniform(0, 0.5),
            'timestamp': time.time()
        }
        
        return metrics
    
    def scan_all_channels(self) -> Dict[int, Dict[str, float]]:
        """
        Scan all available channels.
        
        Returns:
            Dictionary mapping frequencies to metrics
        """
        scan_results = {}
        
        for channel, freq in self.WIFI_24_CHANNELS.items():
            # Skip current channel (we're using it)
            if freq == self.current_channel:
                continue
            
            metrics = self.scan_channel(freq)
            scan_results[freq] = metrics
            
            # Update history
            quality_score = self._calculate_quality_score(metrics)
            self.channel_history[freq].append({
                'quality': quality_score,
                'timestamp': time.time(),
                **metrics
            })
        
        self.last_scan = scan_results
        self.last_scan_time = time.time()
        
        return scan_results
    
    def find_best_channel(self, 
                         exclude_current: bool = True,
                         use_history: bool = True) -> Tuple[int, float]:
        """
        Find the best available channel.
        
        Args:
            exclude_current: Whether to exclude current channel
            use_history: Whether to use historical data
            
        Returns:
            Tuple of (frequency, quality_score)
        """
        candidates = []
        
        for freq in self.WIFI_24_CHANNELS.values():
            if exclude_current and freq == self.current_channel:
                continue
            
            if use_history and len(self.channel_history[freq]) > 0:
                # Use historical average
                recent_scores = [
                    entry['quality'] 
                    for entry in list(self.channel_history[freq])[-10:]
                ]
                avg_quality = np.mean(recent_scores)
                candidates.append((freq, avg_quality))
            elif freq in self.last_scan:
                # Use last scan
                quality = self._calculate_quality_score(self.last_scan[freq])
                candidates.append((freq, quality))
        
        if not candidates:
            return self.current_channel, 0.0
        
        # Sort by quality (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0]
    
    def should_switch_channel(self, current_quality: float) -> bool:
        """
        Determine if channel switch is needed (with hysteresis).
        
        Args:
            current_quality: Current channel quality score
            
        Returns:
            Whether to switch channels
        """
        # Check minimum switch interval
        time_since_switch = time.time() - self.last_switch_time
        if time_since_switch < self.config['min_switch_interval']:
            return False
        
        # Find best alternative
        best_freq, best_quality = self.find_best_channel()
        
        # Apply hysteresis factor
        threshold = current_quality * self.config['hysteresis_factor']
        
        return best_quality > threshold
    
    def recommend_channel_change(self, 
                               current_metrics: Dict[str, float]) -> Optional[Dict]:
        """
        Recommend a channel change if needed.
        
        Args:
            current_metrics: Current channel metrics
            
        Returns:
            Channel change recommendation or None
        """
        current_quality = self._calculate_quality_score(current_metrics)
        
        if not self.should_switch_channel(current_quality):
            return None
        
        best_freq, best_quality = self.find_best_channel()
        
        recommendation = {
            'current_channel': self.current_channel,
            'current_quality': current_quality,
            'recommended_channel': best_freq,
            'expected_quality': best_quality,
            'improvement': best_quality - current_quality,
            'timestamp': time.time()
        }
        
        return recommendation
    
    def execute_channel_change(self, new_frequency: int) -> None:
        """
        Record channel change execution.
        
        Args:
            new_frequency: New channel frequency
        """
        self.switch_history.append({
            'from': self.current_channel,
            'to': new_frequency,
            'timestamp': time.time()
        })
        
        self.current_channel = new_frequency
        self.last_switch_time = time.time()
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate channel quality score from metrics.
        
        Args:
            metrics: Channel metrics
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Normalize energy (assuming -100 to -60 dBm range)
        energy_norm = (metrics['energy_dbm'] + 100) / 40
        energy_score = 1 - np.clip(energy_norm, 0, 1)
        
        # Occupancy score (lower is better)
        occupancy_score = 1 - metrics.get('occupancy', 0)
        
        # Interference score (lower is better)
        interference_score = 1 - metrics.get('interference_score', 0)
        
        # Weighted combination
        quality = (
            energy_score * 0.4 +
            occupancy_score * 0.3 +
            interference_score * 0.3
        )
        
        return np.clip(quality, 0, 1)
    
    def get_channel_report(self) -> Dict:
        """
        Get comprehensive channel status report.
        
        Returns:
            Channel status report
        """
        report = {
            'current_channel': self.current_channel,
            'last_scan_time': self.last_scan_time,
            'time_since_scan': time.time() - self.last_scan_time,
            'channels': {}
        }
        
        for freq in self.WIFI_24_CHANNELS.values():
            if len(self.channel_history[freq]) > 0:
                recent = list(self.channel_history[freq])[-10:]
                report['channels'][freq] = {
                    'avg_quality': np.mean([e['quality'] for e in recent]),
                    'std_quality': np.std([e['quality'] for e in recent]),
                    'last_measurement': recent[-1]['timestamp'],
                    'num_measurements': len(self.channel_history[freq])
                }
        
        return report


class ChannelChangeProtocol:
    """
    Protocol for communicating channel changes to HackRF.
    
    TODO: Implement actual HackRF control interface
    """
    
    def __init__(self, hackrf_address: str = "localhost", port: int = 5000):
        """
        Initialize channel change protocol.
        
        Args:
            hackrf_address: HackRF control address
            port: Control port
        """
        self.hackrf_address = hackrf_address
        self.port = port
    
    def send_channel_change(self, recommendation: Dict) -> bool:
        """
        Send channel change command to HackRF.
        
        Args:
            recommendation: Channel change recommendation
            
        Returns:
            Success status
        """
        # TODO: Implement actual protocol
        # This would send a message like:
        # {
        #     "command": "change_channel",
        #     "frequency": recommendation['recommended_channel'],
        #     "timestamp": time.time(),
        #     "reason": "jamming_detected"
        # }
        
        print(f"Would send channel change to {recommendation['recommended_channel']} MHz")
        return True
    
    def verify_channel_change(self, expected_frequency: int) -> bool:
        """
        Verify that channel change was successful.
        
        Args:
            expected_frequency: Expected new frequency
            
        Returns:
            Verification status
        """
        # TODO: Query HackRF for current frequency
        return True