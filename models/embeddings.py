import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_timesteps: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        position = torch.arange(max_timesteps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_timesteps, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.pe[timesteps]


class DifficultyEmbedding(nn.Module):
    """
    Continuous Fourier-based difficulty embedding.
    Uses sinusoidal features to create smooth, distinct representations
    for all difficulty values including intermediate ones like 0.5.
    """
    
    def __init__(self, embedding_dim: int, num_frequencies: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        
        # Create frequency bands that span different scales
        # This helps distinguish 0.0, 0.5, and 1.0 well
        frequencies = torch.linspace(0, 4, num_frequencies)  # 0 to 4*pi cycles
        self.register_buffer('frequencies', frequencies)
        
        # Raw Fourier features: sin + cos for each frequency = 2 * num_frequencies
        fourier_dim = num_frequencies * 2
        
        # Project Fourier features to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim + 1, embedding_dim),  # +1 for raw difficulty value
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        
        # Learnable scale to adjust embedding magnitude
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, difficulty: torch.Tensor) -> torch.Tensor:
        if difficulty.dim() == 0:
            difficulty = difficulty.unsqueeze(0)
        if difficulty.dim() == 2:
            difficulty = difficulty.squeeze(-1)
        
        # Normalize difficulty to [0, 1] (should already be, but ensure)
        difficulty = difficulty.float().clamp(0, 1)
        
        # Create Fourier features: sin(2*pi*freq*difficulty) and cos(2*pi*freq*difficulty)
        # Shape: [batch, num_frequencies]
        angles = 2 * math.pi * difficulty.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        
        # Concatenate: [batch, 2*num_frequencies + 1]
        fourier_features = torch.cat([
            sin_features, 
            cos_features, 
            difficulty.unsqueeze(-1)  # Include raw value for direct access
        ], dim=-1)
        
        # Project to embedding dimension
        emb = self.projection(fourier_features)
        
        return emb * self.scale