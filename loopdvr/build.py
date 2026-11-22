"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified the EdgeCrafter feature pipeline for standalone model construction.
---------------------------------------------------------------------------------
Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

from pathlib import Path

import torch
from torch import nn
import yaml

from .models.looped_vit import LoopedViT
from .models.encoder import HybridEncoder
from .models.detection import DetectionDecoder
from .models.pose import PoseDecoder


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["task"] not in {"det", "seg", "pose"}:
        raise ValueError(f"Unknown task: {config['task']}")
    return config


class LoopDVR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.task = config["task"]
        self.looped_vit = LoopedViT(**config["looped_vit"])
        self.encoder = HybridEncoder(**config["encoder"], eval_spatial_size=config["input_size"])
        decoder = PoseDecoder if self.task == "pose" else DetectionDecoder
        self.decoder = decoder(**config["decoder"], eval_spatial_size=config["input_size"])

    def forward(self, images):
        features = self.encoder(self.looped_vit(images))
        if self.task == "seg":
            return self.decoder(features, spatial_feat=features[0])
        return self.decoder(features)


def build_model(config, weights_path, device="cpu"):
    """Strictly load the named release weights and enter evaluation mode."""
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if weights["model_id"] != config["model_id"]:
        raise ValueError(f"Expected {config['model_id']} weights, got {weights['model_id']}")
    model = LoopDVR(config)
    model.load_state_dict(weights["state_dict"], strict=True)
    return model.eval().to(device)
