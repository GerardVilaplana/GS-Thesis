"""Neural network used by the selected HandAffordSplat configuration."""

from __future__ import annotations

import torch
from torch import nn


class PointNetMean(nn.Module):
    """Shared Gaussian encoder with mean-pooled object context."""

    def __init__(self, input_dim: int = 401, latent_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("features must have shape (N, D) with N > 0")
        local = self.encoder(features)
        context = local.mean(dim=0, keepdim=True).expand(len(local), -1)
        return self.head(torch.cat([local, context], dim=1)).squeeze(-1)
