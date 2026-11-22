"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified for single-block LoopSeg inference.
---------------------------------------------------------------------------------
Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RF-DETR (https://github.com/roboflow/rf-detr)
Copyright (c) 2025 Roboflow. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (see LICENSE).
"""

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseConvBlock(nn.Module):
    """Mix spatial features at the mask resolution."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        spatial = self.dwconv(x).permute(0, 2, 3, 1)
        update = self.act(self.pwconv1(self.norm(spatial)))
        return update.permute(0, 3, 1, 2) + x


class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        return x + self.layers(self.norm_in(x))


class SegmentationHead(nn.Module):
    """Predict one mask per final decoder query from the first feature map."""

    def __init__(self, dim, image_size, downsample_ratio):
        super().__init__()
        self.mask_size = tuple(side // downsample_ratio for side in image_size)
        self.blocks = nn.ModuleList([DepthwiseConvBlock(dim)])
        self.spatial_features_proj = nn.Conv2d(dim, dim, 1)
        self.query_features_block = MLPBlock(dim)
        self.query_features_proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, spatial, queries):
        spatial = F.interpolate(spatial, size=self.mask_size, mode="bilinear", align_corners=False)
        spatial = self.spatial_features_proj(self.blocks[0](spatial))
        queries = self.query_features_proj(self.query_features_block(queries))
        return torch.einsum("bchw,bnc->bnhw", spatial, queries) + self.bias
