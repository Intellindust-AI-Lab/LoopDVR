"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified for the recurrent LoopDet and LoopSeg decoders.
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

from .decoder_common import (Gate, MLP, activation_fn, expectation, flatten_features,
                             inverse_sigmoid, group_route, regression_bins)
from .loop_adapter import LoopAdapter
from .segmentation import SegmentationHead


class BoxAttention(nn.Module):
    """Sample image features around each query's current reference box."""

    def __init__(self, dim, heads, points):
        super().__init__()
        self.heads = heads
        self.points = tuple(points)
        total = heads * sum(points)
        self.sampling_offsets = nn.Linear(dim, total * 2)
        self.attention_weights = nn.Linear(dim, total)
        scales = [1 / n for n in points for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(scales))

    def forward(self, query, reference, values, shapes):
        batch, count = query.shape[:2]
        n = sum(self.points)
        offsets = self.sampling_offsets(query).reshape(batch, count, self.heads, n, 2)
        weights = self.attention_weights(query).reshape(batch, count, self.heads, n).softmax(-1)
        offset = offsets * self.num_points_scale.to(query.dtype)[:, None] * reference[:, :, None, None, 2:] * 0.5
        locations = reference[:, :, None, None, :2] + offset
        grids = (2 * locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(self.points, dim=-2)
        samples = []
        for value, (height, width), grid in zip(values, shapes, grids):
            feature = value.reshape(batch * self.heads, -1, height, width)
            samples.append(F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
        weights = weights.permute(0, 2, 1, 3).reshape(batch * self.heads, 1, count, n)
        result = (torch.cat(samples, dim=-1) * weights).sum(-1)
        return result.reshape(batch, -1, count).permute(0, 2, 1)


class DetectionLayer(nn.Module):
    def __init__(self, dim, heads, ffn, points, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = BoxAttention(dim, heads, points)
        self.gateway = Gate(dim)
        self.linear1 = nn.Linear(dim, ffn)
        self.activation = activation_fn(activation)
        self.linear2 = nn.Linear(ffn, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, target, pos, refs, values, shapes):
        query = target + pos
        target = self.norm1(target + self.self_attn(query, query, target)[0])
        target = self.gateway(target, self.cross_attn(target + pos, refs, values, shapes))
        update = self.linear2(self.activation(self.linear1(target)))
        return self.norm2((target + update).clamp(-65504, 65504))


class BoxQuality(nn.Module):
    def __init__(self, reg_max, activation):
        super().__init__()
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * 5, 64, 1, 2, activation)

    def forward(self, scores, corners):
        probs = F.softmax(corners.reshape(*corners.shape[:2], 4, self.reg_max + 1), -1)
        top = probs.topk(4, dim=-1).values
        stats = torch.cat([top, top.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stats.flatten(-2))


def refine_boxes(box, delta, reg_scale):
    cx, cy, w, h = box.unbind(-1)
    scale = reg_scale.abs()
    x1 = cx - (0.5 * scale + delta[..., 0]) * (w / scale)
    y1 = cy - (0.5 * scale + delta[..., 1]) * (h / scale)
    x2 = cx + (0.5 * scale + delta[..., 2]) * (w / scale)
    y2 = cy + (0.5 * scale + delta[..., 3]) * (h / scale)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


class BoxGroups(nn.Module):
    def __init__(self, dim, ffn, adapter_dim, heads, points, activation, group_sizes,
                 reg_max, reg_scale, up, mask_head):
        super().__init__()
        self.layer_to_group = group_route(group_sizes)
        self.layers = nn.ModuleList(DetectionLayer(dim, heads, ffn, points, activation) for _ in group_sizes)
        self.loop_adapters = nn.ModuleList(LoopAdapter(dim, adapter_dim) for _ in self.layer_to_group)
        self.lqe_layers = nn.ModuleList(BoxQuality(reg_max, activation) for _ in group_sizes)
        self.up, self.reg_scale = up, reg_scale
        self.segmentation_head = mask_head


class DetectionDecoder(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward, adapter_hidden_dim, num_queries, eval_spatial_size,
                 num_classes=80, num_points=(3, 6, 3), group_sizes=(2, 2), feat_strides=(8, 16, 32),
                 nhead=8, reg_max=32, reg_scale=4, activation="silu", mask_downsample_ratio=None):
        super().__init__()
        if len(num_points) != len(feat_strides) or hidden_dim % nhead != 0:
            raise ValueError("Incompatible decoder feature levels or attention dimensions")
        self.num_queries = num_queries
        self.nhead = nhead
        self.reg_max = reg_max
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale], dtype=torch.float32), requires_grad=False)
        mask_head = (SegmentationHead(hidden_dim, eval_spatial_size, mask_downsample_ratio)
                     if mask_downsample_ratio is not None else None)
        self.decoder = BoxGroups(hidden_dim, dim_feedforward, adapter_hidden_dim, nhead, num_points,
                                 activation, group_sizes, reg_max, self.reg_scale, self.up, mask_head)
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, activation)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation)
        self.dec_score_head = nn.ModuleList(nn.Linear(hidden_dim, num_classes) for _ in group_sizes)
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, activation)])

        anchors = []
        for level, stride in enumerate(feat_strides):
            h, w = [side // stride for side in eval_spatial_size]
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            xy = (torch.stack([x, y], dim=-1) + 0.5) / torch.tensor([w, h], dtype=torch.float32)
            wh = torch.ones_like(xy) * 0.05 * (2.0 ** level)
            anchors.append(torch.cat([xy, wh], dim=-1).reshape(1, h*w, 4))
        anchors = torch.cat(anchors, 1)
        valid = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        self.register_buffer("anchors", torch.where(valid, torch.log(anchors/(1-anchors)), torch.inf))
        self.register_buffer("valid_mask", valid)

    def forward(self, feats, spatial_feat=None):
        memory, shapes = flatten_features(feats)
        proposals = self.valid_mask.to(memory.dtype) * memory
        scores = self.enc_score_head(proposals).max(-1).values
        index = scores.topk(self.num_queries, dim=-1).indices.unsqueeze(-1)
        output = proposals.gather(1, index.expand(-1, -1, memory.shape[-1]))
        anchors = self.anchors.expand(memory.shape[0], -1, -1).gather(1, index.expand(-1, -1, 4))
        refs = (self.enc_bbox_head(output) + anchors).sigmoid()
        pos = self.query_pos_head(refs).clamp(-10, 10)
        values = memory.reshape(*memory.shape[:2], self.nhead, -1).permute(0, 2, 3, 1)
        values = values.split([h*w for h, w in shapes], dim=-1)
        bins = regression_bins(self.reg_max, self.up, self.reg_scale)
        previous, corners = 0, 0
        for layer_idx, group_idx in enumerate(self.decoder.layer_to_group):
            output = self.decoder.layers[group_idx](output, pos, refs, values, shapes)
            output = self.decoder.loop_adapters[layer_idx](output)
            if layer_idx == 0:
                base = (self.pre_bbox_head(output) + inverse_sigmoid(refs, 1e-5)).sigmoid()
            corners = self.dec_bbox_head[0](output + previous) + corners
            refs = refine_boxes(base, expectation(corners, bins), self.reg_scale)
            previous = output
        final_group = self.decoder.layer_to_group[-1]
        scores = self.decoder.lqe_layers[final_group](self.dec_score_head[final_group](output), corners)
        masks = self.decoder.segmentation_head(spatial_feat, output) if self.decoder.segmentation_head is not None else None
        return {"pred_logits": scores, "pred_boxes": refs, "pred_masks": masks}
