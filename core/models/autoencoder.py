"""
Improved RF Autoencoder for anomaly detection.
Optimized for Jetson deployment with mixed precision support.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ImprovedRFAutoencoder(nn.Module):
    """
    Autoencoder with residual connections and attention mechanisms.
    
    This model is designed to learn normal RF signal patterns and detect
    anomalies through reconstruction error analysis.
    """
    
    def __init__(self, input_size: int = 1024, latent_dim: int = 32):
        """
        Initialize the autoencoder.
        
        Args:
            input_size: Size of input signal window
            latent_dim: Dimension of latent representation
        """
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = self._build_encoder()
        
        # Attention mechanism
        self.attention = self._build_attention()
        
        # Bottleneck
        self.encoder_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, latent_dim),
            nn.Tanh()  # Bound latent space
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8))
        )
        
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> nn.ModuleDict:
        """Build encoder layers with residual connections."""
        return nn.ModuleDict({
            'conv1': nn.Sequential(
                nn.Conv1d(2, 32, 7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1)
            ),
            'conv2': nn.Sequential(
                nn.Conv1d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1)
            ),
            'conv3': nn.Sequential(
                nn.Conv1d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1)
            )
        })
    
    def _build_attention(self) -> nn.Sequential:
        """Build attention mechanism for feature weighting."""
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
    
    def _build_decoder(self) -> nn.ModuleDict:
        """Build decoder layers."""
        return nn.ModuleDict({
            'conv1': nn.Sequential(
                nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1)
            ),
            'conv2': nn.Sequential(
                nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1)
            ),
            'conv3': nn.Sequential(
                nn.ConvTranspose1d(32, 2, 7, stride=2, padding=3, output_padding=1),
                nn.Upsample(size=self.input_size, mode='linear', align_corners=False)
            )
        })
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 2, input_size)
            
        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        # Encoder with residual connections
        e1 = self.encoder['conv1'](x)
        e2 = self.encoder['conv2'](e1)
        e3 = self.encoder['conv3'](e2)
        
        # Apply attention
        att_weights = self.attention(e3).unsqueeze(2)
        e3_attended = e3 * att_weights
        
        # Bottleneck
        z = self.encoder_fc(e3_attended)
        
        # Decoder
        d = self.decoder_fc(z)
        d1 = self.decoder['conv1'](d)
        d2 = self.decoder['conv2'](d1)
        reconstruction = self.decoder['conv3'](d2)
        
        return reconstruction, z
    
    @torch.jit.export
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only the latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        e1 = self.encoder['conv1'](x)
        e2 = self.encoder['conv2'](e1)
        e3 = self.encoder['conv3'](e2)
        
        att_weights = self.attention(e3).unsqueeze(2)
        e3_attended = e3 * att_weights
        
        return self.encoder_fc(e3_attended)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder weights for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.encoder_fc.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.attention.parameters():
            param.requires_grad = True
        for param in self.encoder_fc.parameters():
            param.requires_grad = True


class AutoencoderWithRFPUF(ImprovedRFAutoencoder):
    """
    Extended autoencoder that incorporates RF-PUF concepts.
    
    This model can learn device-specific signatures similar to the
    RF-PUF paper, enabling both anomaly detection and device authentication.
    """
    
    def __init__(self, input_size: int = 1024, latent_dim: int = 32, 
                 n_devices: Optional[int] = None):
        """
        Initialize autoencoder with RF-PUF capabilities.
        
        Args:
            input_size: Size of input signal window
            latent_dim: Dimension of latent representation
            n_devices: Number of devices for authentication (optional)
        """
        super().__init__(input_size, latent_dim)
        
        # Additional layers for device signature extraction
        self.signature_extractor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
        
        # Optional device classifier (for RF-PUF mode)
        self.device_classifier = None
        if n_devices is not None:
            self.device_classifier = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(latent_dim * 2, n_devices)
            )
    
    def extract_signature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract device signature from input signal.
        
        Args:
            x: Input tensor
            
        Returns:
            Device signature vector
        """
        z = self.get_latent(x)
        return self.signature_extractor(z)
    
    def forward_with_classification(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional device classification.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, latent, device_logits)
        """
        reconstruction, z = self.forward(x)
        
        device_logits = None
        if self.device_classifier is not None:
            device_logits = self.device_classifier(z)
        
        return reconstruction, z, device_logits