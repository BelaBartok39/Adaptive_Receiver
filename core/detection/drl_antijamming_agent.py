"""
DRL-based anti-jamming agent inspired by the paper.
Integrates with the existing anomaly detection system.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional
import time

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class AntiJammingDQN(nn.Module):
    """
    Deep Q-Network for anti-jamming channel selection.
    Based on the paper's architecture.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        
        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer for DDQN.
    Samples experiences based on TD error priority.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        
    def push(self, experience: Experience, td_error: float):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities.append(priority)
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
            
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority


class DRLAntiJammingAgent:
    """
    DRL-based anti-jamming agent using DDQN with prioritized replay.
    Designed for resource-constrained IoT devices.
    """
    
    def __init__(self, 
                 num_channels: int = 8,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.005,
                 channel_switch_cost: float = 0.05,
                 device: Optional[str] = None):
        """
        Initialize the anti-jamming agent.
        
        Args:
            num_channels: Number of available channels
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            channel_switch_cost: Penalty for switching channels (Γ)
            device: Device to run on
        """
        self.num_channels = num_channels
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.channel_switch_cost = channel_switch_cost
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Neural networks (DDQN uses two networks)
        self.q_network = AntiJammingDQN(num_channels, num_channels).to(self.device)
        self.target_network = AntiJammingDQN(num_channels, num_channels).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        
        # Training parameters
        self.batch_size = 32
        self.target_update_freq = 100
        self.steps = 0
        
        # State tracking
        self.current_channel = 0
        self.last_channel = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select channel using epsilon-greedy policy.
        
        Args:
            state: Current state (channel powers)
            training: Whether in training mode
            
        Returns:
            Selected channel index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_channels - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def calculate_reward(self, 
                        channel: int, 
                        jammed: bool, 
                        throughput: float = 1.0) -> float:
        """
        Calculate reward based on the paper's formula.
        
        Args:
            channel: Selected channel
            jammed: Whether channel is jammed
            throughput: Normalized throughput achieved
            
        Returns:
            Reward value
        """
        if jammed:
            return 0.0
        
        # Apply channel switching cost
        switch_penalty = 0.0
        if channel != self.last_channel:
            switch_penalty = self.channel_switch_cost
            
        return throughput - switch_penalty
    
    def train_step(self, experience: Experience):
        """
        Perform one training step.
        
        Args:
            experience: Experience tuple
        """
        if len(self.memory.buffer) < self.batch_size:
            return
        
        # Sample batch
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        if not experiences:
            return
        
        # Prepare batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # DDQN: Use online network to select action, target network to evaluate
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # TD errors for priority update
        td_errors = (target_q_values - current_q_values.squeeze()).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Weighted loss
        loss = (weights * (current_q_values.squeeze() - target_q_values.detach()) ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
    
    def update(self, 
               state: np.ndarray, 
               action: int, 
               reward: float, 
               next_state: np.ndarray, 
               done: bool):
        """
        Update agent with new experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Calculate initial TD error for prioritization
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            current_q = self.q_network(state_t)[0, action].item()
            next_action = self.q_network(next_state_t).argmax().item()
            next_q = self.target_network(next_state_t)[0, next_action].item()
            
            td_error = reward + self.gamma * next_q * (1 - done) - current_q
        
        # Store experience with priority
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience, td_error)
        
        # Train
        self.train_step(experience)
        
        # Update channel tracking
        self.last_channel = self.current_channel
        self.current_channel = action
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


class IntegratedAntiJammingSystem:
    """
    Integrates DRL anti-jamming with existing anomaly detection.
    """
    
    def __init__(self, 
                 anomaly_detector,
                 num_channels: int = 8,
                 channel_frequencies: Optional[List[int]] = None):
        """
        Initialize integrated system.
        
        Args:
            anomaly_detector: Existing anomaly detector instance
            num_channels: Number of available channels
            channel_frequencies: List of channel center frequencies (MHz)
        """
        self.anomaly_detector = anomaly_detector
        self.num_channels = num_channels
        
        # Default to 5GHz UNII-1 band channels
        if channel_frequencies is None:
            self.channel_frequencies = [
                5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320
            ][:num_channels]
        else:
            self.channel_frequencies = channel_frequencies
        
        # Initialize DRL agent
        self.agent = DRLAntiJammingAgent(num_channels=num_channels)
        
        # Channel state tracking
        self.current_channel_idx = 0
        self.channel_powers = np.zeros(num_channels)
        
    def scan_all_channels(self) -> np.ndarray:
        """
        Scan all channels to get power levels.
        
        Returns:
            Array of received power levels (dBm)
        """
        # This would interface with your HackRF
        # For now, returning placeholder
        powers = []
        for freq in self.channel_frequencies:
            # Simulate scanning
            power = np.random.normal(-80, 5)  # Placeholder
            powers.append(power)
        
        self.channel_powers = np.array(powers)
        return self.channel_powers
    
    def detect_and_avoid(self, i_data: np.ndarray, q_data: np.ndarray) -> Dict:
        """
        Detect jamming and select best channel.
        
        Args:
            i_data: Current channel I data
            q_data: Current channel Q data
            
        Returns:
            Action dictionary with channel recommendation
        """
        # 1. Detect jamming on current channel
        is_jammed, confidence, metrics = self.anomaly_detector.detect(i_data, q_data)
        
        # 2. Get full spectrum state
        state = self.scan_all_channels()
        
        # 3. Select action (channel)
        if is_jammed:
            # High confidence jamming - must switch
            action = self.agent.select_action(state, training=False)
        else:
            # Maybe stay on current channel
            action = self.current_channel_idx
        
        # 4. Calculate reward
        reward = self.agent.calculate_reward(
            channel=action,
            jammed=is_jammed,
            throughput=1.0 if not is_jammed else 0.0
        )
        
        # 5. Update agent (if training)
        next_state = state.copy()  # Would be next scan
        self.agent.update(state, action, reward, next_state, False)
        
        # 6. Prepare response
        response = {
            'current_channel': self.channel_frequencies[self.current_channel_idx],
            'recommended_channel': self.channel_frequencies[action],
            'should_switch': action != self.current_channel_idx,
            'jamming_detected': is_jammed,
            'confidence': confidence,
            'channel_powers': dict(zip(self.channel_frequencies, state)),
            'metrics': metrics
        }
        
        # Update current channel
        self.current_channel_idx = action
        
        return response