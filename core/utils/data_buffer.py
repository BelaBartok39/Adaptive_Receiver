"""
Efficient data buffering utilities for streaming RF data.
"""

import numpy as np
from collections import deque
from typing import Optional, List, Tuple
import threading


class CircularBuffer:
    """
    Thread-safe circular buffer for RF I/Q data.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize circular buffer.
        
        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.i_buffer = deque(maxlen=capacity)
        self.q_buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        
    def append(self, i_samples: np.ndarray, q_samples: np.ndarray) -> None:
        """
        Append I/Q samples to buffer.
        
        Args:
            i_samples: In-phase samples
            q_samples: Quadrature samples
        """
        with self.lock:
            self.i_buffer.extend(i_samples)
            self.q_buffer.extend(q_samples)
    
    def get_window(self, size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get a window of samples from buffer.
        
        Args:
            size: Window size
            
        Returns:
            Tuple of (i_data, q_data) or None if insufficient data
        """
        with self.lock:
            if len(self.i_buffer) < size:
                return None
            
            i_data = np.array(list(self.i_buffer)[:size])
            q_data = np.array(list(self.q_buffer)[:size])
            
            return i_data, q_data
    
    def consume(self, size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get and remove samples from buffer.
        
        Args:
            size: Number of samples to consume
            
        Returns:
            Tuple of (i_data, q_data) or None if insufficient data
        """
        with self.lock:
            if len(self.i_buffer) < size:
                return None
            
            i_data = []
            q_data = []
            
            for _ in range(size):
                i_data.append(self.i_buffer.popleft())
                q_data.append(self.q_buffer.popleft())
            
            return np.array(i_data), np.array(q_data)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.i_buffer.clear()
            self.q_buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.i_buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size() >= self.capacity


class SlidingWindowBuffer:
    """
    Sliding window buffer for continuous processing.
    """
    
    def __init__(self, window_size: int = 1024, stride: int = 512):
        """
        Initialize sliding window buffer.
        
        Args:
            window_size: Size of each window
            stride: Stride between windows
        """
        self.window_size = window_size
        self.stride = stride
        self.buffer = CircularBuffer(capacity=window_size * 10)
        
    def process_samples(self, i_samples: np.ndarray, q_samples: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Process samples and return available windows.
        
        Args:
            i_samples: New in-phase samples
            q_samples: New quadrature samples
            
        Returns:
            List of (i_window, q_window) tuples
        """
        self.buffer.append(i_samples, q_samples)
        
        windows = []
        while self.buffer.size() >= self.window_size:
            window_data = self.buffer.consume(self.stride)
            if window_data is not None:
                # Get full window
                full_window = self.buffer.get_window(self.window_size - self.stride)
                if full_window is not None:
                    i_data = np.concatenate([window_data[0], full_window[0]])
                    q_data = np.concatenate([window_data[1], full_window[1]])
                    windows.append((i_data[:self.window_size], q_data[:self.window_size]))
        
        return windows