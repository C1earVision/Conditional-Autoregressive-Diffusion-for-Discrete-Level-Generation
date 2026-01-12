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


class FourierDifficultyEmbedding(nn.Module):
    """Linear MLP embedding for difficulty (replaces Fourier approach).
    
    Uses a deeper network to learn representations for intermediate values
    without the potential compression issues of Fourier features.
    """
    def __init__(self, embedding_dim: int, num_frequencies: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Deeper MLP for better intermediate value representation
        hidden_dim = embedding_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, difficulty: torch.Tensor) -> torch.Tensor:
        if difficulty.dim() == 1:
            difficulty = difficulty.unsqueeze(-1)
        
        # Scale to [-1, 1] range for better gradient flow
        scaled = difficulty * 2 - 1
        
        return self.projection(scaled)