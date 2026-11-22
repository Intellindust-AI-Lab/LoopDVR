"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0 AND MIT

Adapted the learned residual projection for each backbone and decoder layer.
---------------------------------------------------------------------------------
Modified from SReT: Sliced Recursive Transformer (https://github.com/szq0214/SReT)
Copyright (c) 2021 Zhiqiang Shen.
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LearnableCoefficient(nn.Module):
    def __init__(self, initial: float):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([initial]))

    def forward(self, x):
        return x * self.bias


class LoopAdapter(nn.Module):
    """A per-layer residual MLP with two learned scalar coefficients."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, hidden_dim)
        self.coefficient1 = LearnableCoefficient(1.0)
        self.coefficient2 = LearnableCoefficient(0.0)

    def forward(self, x):
        return self.coefficient1(x) + self.coefficient2(self.mlp(self.norm1(x)))
