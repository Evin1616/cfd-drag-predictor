"""PointNet regressor: surface point cloud (xyz + normals) -> [Cd, Cl]."""
from __future__ import annotations

import torch
from torch import nn


class PointNetRegressor(nn.Module):
    def __init__(self, in_ch: int = 6, n_out: int = 2, width: int = 512, dropout: float = 0.2):
        super().__init__()
        chans = [in_ch, 64, 128, 256, width]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv1d(a, b, 1), nn.BatchNorm1d(b), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * width, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, n_out),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: (B, N, C) -> (B, n_out), in standardised target units."""
        feat = self.encoder(pts.transpose(1, 2))
        # max-pool picks out sharp features, mean-pool keeps global proportions
        g = torch.cat([feat.max(dim=2).values, feat.mean(dim=2)], dim=1)
        return self.head(g)
