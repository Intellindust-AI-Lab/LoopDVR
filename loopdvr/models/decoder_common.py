"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified for distribution refinement across detection, segmentation and pose.
---------------------------------------------------------------------------------
Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
from torch import nn
from torch.nn import functional as F


def activation_fn(name):
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported decoder activation: {name}")


def group_route(group_sizes):
    """Map each decoder refinement layer to the index of its sharing group."""
    if not group_sizes or any(type(n) is not int or n < 1 for n in group_sizes):
        raise ValueError("Decoder group sizes must be positive integers")
    return tuple(group_idx for group_idx, size in enumerate(group_sizes) for _ in range(size))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.act = activation_fn(act)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = self.act(x)
        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2 * dim, 2 * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, first, second):
        weights = self.gate(torch.cat([first, second], dim=-1)).sigmoid()
        w1, w2 = weights.chunk(2, dim=-1)
        return self.norm(w1 * first + w2 * second)


def inverse_sigmoid(x, eps):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def regression_bins(reg_max, up, reg_scale):
    """Non-uniform bin centers from the reference FP32 validation path."""
    half = up[0].abs() * reg_scale.abs()
    step = (half + 1) ** (2 / (reg_max - 2))
    left = [-(step ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step ** i - 1 for i in range(1, reg_max // 2)]
    return torch.cat([-2 * half, *left, torch.zeros_like(half), *right, 2 * half])


def expectation(logits, bins):
    probs = F.softmax(logits.reshape(-1, bins.numel()), dim=1)
    return F.linear(probs, bins).reshape(*logits.shape[:-1], -1)


def flatten_features(features):
    shapes = [feature.shape[-2:] for feature in features]
    tokens = [feature.flatten(2).transpose(1, 2) for feature in features]
    return torch.cat(tokens, 1), shapes
