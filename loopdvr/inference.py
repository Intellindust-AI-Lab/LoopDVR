"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Adapted image preprocessing for the LoopDVR prediction API.
---------------------------------------------------------------------------------
Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
from torchvision import transforms

from .models.postprocess import postprocess


def _scaled_pil_tensor(image):
    # Detection and segmentation validation retain the RGB interleaved strides.
    return transforms.functional.pil_to_tensor(image).float() / 255.0


def image_transform(config):
    return transforms.Compose([
        transforms.Resize(tuple(config["input_size"])),
        transforms.ToTensor() if config["task"] == "pose" else _scaled_pil_tensor,
        transforms.Normalize(config["mean"], config["std"]),
    ])


@torch.inference_mode()
def predict_image(model, config, image):
    device = next(model.parameters()).device
    tensor = image_transform(config)(image.convert("RGB")).unsqueeze(0).to(device)
    sizes = torch.tensor([image.size], device=device)
    return postprocess(model(tensor), sizes, config["task"])[0]
