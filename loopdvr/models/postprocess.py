"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified to return detections, masks and keypoints in original image coordinates.
---------------------------------------------------------------------------------
Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
https://github.com/SebastianJanampa/DETRPose
"""

import torch
from torch.nn import functional as F
from torchvision.ops import box_convert


def postprocess(outputs, sizes_wh, task):
    """Keep the reference top queries; detection labels are contiguous class indices."""
    if task not in {"det", "seg", "pose"}:
        raise ValueError(f"Unknown task: {task}")
    logits = outputs["pred_logits"]
    sizes_wh = torch.as_tensor(sizes_wh, device=logits.device)
    if sizes_wh.shape != (logits.shape[0], 2):
        raise ValueError("Expected one (width, height) per image")
    count, classes = logits.shape[1:]
    scores, indices = logits.sigmoid().flatten(1).topk(count, dim=1)
    labels, queries = indices % classes, indices // classes
    results = []
    if task == "pose":
        points = outputs["pred_keypoints"].reshape(*logits.shape[:2], -1, 2)
        points = points.gather(1, queries[..., None, None].expand(-1, -1, points.shape[2], 2))
        points = points * sizes_wh[:, None, None]
        keypoints = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1).flatten(-2)
        return [dict(scores=s, labels=l, keypoints=k) for s, l, k in zip(scores, labels, keypoints)]
    boxes = box_convert(outputs["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
    boxes = boxes * sizes_wh.repeat(1, 2)[:, None, :]
    boxes = boxes.gather(1, queries[..., None].expand(-1, -1, 4))
    for i, (s, l, b) in enumerate(zip(scores, labels, boxes)):
        pred = dict(scores=s, labels=l, boxes=b)
        if task == "seg":
            w, h = sizes_wh[i].tolist()
            masks = outputs["pred_masks"][i].index_select(0, queries[i]).unsqueeze(1)
            pred["masks"] = F.interpolate(masks, size=(int(h), int(w)), mode="bilinear", align_corners=False) > 0
        results.append(pred)
    return results
