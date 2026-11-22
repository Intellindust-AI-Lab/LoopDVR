"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified for the recurrent LoopPose decoder and its group-shared keypoint heads.
---------------------------------------------------------------------------------
Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
https://github.com/SebastianJanampa/DETRPose
---------------------------------------------------------------------------------
Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
Copyright (c) 2021 Microsoft. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
Copyright (c) 2020 SenseTime. All Rights Reserved.
Its attention implementation credits Deformable-Convolution-V2-PyTorch.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .decoder_common import (Gate, MLP, activation_fn, expectation, flatten_features,
                             inverse_sigmoid, group_route, regression_bins)
from .loop_adapter import LoopAdapter


class PoseAttention(nn.Module):
    """Deformable attention sampled around the instance and each joint."""

    def __init__(self, dim, heads, levels, points):
        super().__init__()
        self.heads = heads
        self.levels = levels
        self.points = points
        total = heads * levels * points
        self.sampling_offsets = nn.Linear(dim, total * 2)
        self.attention_weights = nn.Linear(dim, total)

    def forward(self, query, reference, values, shapes):
        batch, count = query.shape[:2]
        offsets = self.sampling_offsets(query).reshape(batch, count, self.heads, self.levels, self.points, 2)
        weights = self.attention_weights(query).reshape(batch, count, self.heads, self.levels * self.points)
        weights = weights.softmax(-1).transpose(1, 2).reshape(batch * self.heads, 1, count, -1)
        sizes = torch.tensor(shapes, device=query.device).flip([1]).reshape(1, 1, 1, self.levels, 1, 2)
        locations = reference.flatten(1, 2)[:, :, None, None, None, :] + offsets / sizes
        grids = (2 * locations - 1).transpose(1, 2).flatten(0, 1)
        sampled = []
        for index, shape in enumerate(shapes):
            sampled.append(F.grid_sample(values[index].unflatten(2, shape), grids[:, :, index],
                                         mode="bilinear", padding_mode="zeros", align_corners=False))
        output = (torch.cat(sampled, dim=-1) * weights).sum(-1)
        return output.view(batch, -1, count).transpose(1, 2)


class PoseLayer(nn.Module):
    def __init__(self, dim, ffn, heads, levels, points, activation):
        super().__init__()
        self.within_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.within_norm = nn.LayerNorm(dim)
        self.across_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.across_norm = nn.LayerNorm(dim)
        self.cross_attn = PoseAttention(dim, heads, levels, points)
        self.gateway = Gate(dim)
        self.linear1 = nn.Linear(dim, ffn)
        self.linear2 = nn.Linear(ffn, dim)
        self.activation = activation_fn(activation)
        self.norm2 = nn.LayerNorm(dim)

    @staticmethod
    def with_position(tgt, position):
        return torch.cat((tgt[:, :, :1], tgt[:, :, 1:] + position), dim=2)

    def forward(self, tgt, pos, references, values, shapes):
        batch, queries, joints, dim = tgt.shape
        q = self.with_position(tgt, pos).flatten(0, 1)
        update = self.within_attn(q, q, tgt.flatten(0, 1))[0].reshape_as(tgt)
        tgt = self.within_norm(tgt + update)
        tgt = tgt.transpose(1, 2).flatten(0, 1)
        update = self.across_attn(tgt, tgt, tgt)[0]
        tgt = self.across_norm(tgt + update).reshape(batch, joints, queries, dim).transpose(1, 2)
        q = self.with_position(tgt, pos).flatten(1, 2)
        update = self.cross_attn(q, references, values, shapes).reshape_as(tgt)
        tgt = self.gateway(tgt, update)
        return self.norm2((tgt + self.linear2(self.activation(self.linear1(tgt)))).clamp(-65504, 65504))


class PoseQuality(nn.Module):
    def __init__(self, joints):
        super().__init__()
        self.reg_conf = MLP(joints * 5, 256, 1, 2)

    def forward(self, logits, coordinates, features):
        samples = F.grid_sample(features, 2 * coordinates - 1, mode="bilinear",
                                padding_mode="zeros", align_corners=False).permute(0, 2, 3, 1)
        top = samples.topk(4, dim=-1).values
        stats = torch.cat([top, top.mean(-1, keepdim=True)], dim=-1)
        return logits + self.reg_conf(stats.flatten(-2))


class PoseGroups(nn.Module):
    def __init__(self, dim, ffn, adapter_dim, heads, levels, points, activation, group_sizes):
        super().__init__()
        self.layer_to_group = group_route(group_sizes)
        self.layers = nn.ModuleList(PoseLayer(dim, ffn, heads, levels, points, activation) for _ in group_sizes)
        self.loop_adapters = nn.ModuleList(LoopAdapter(dim, adapter_dim) for _ in self.layer_to_group)
        self.half_pose_ref_point_head = MLP(dim, dim, dim, 2)
        frequencies = torch.arange(dim // 2, dtype=torch.float32)
        self.register_buffer("dim_t", 10000 ** (2 * (frequencies // 2) / (dim // 2)))

    def query_position(self, coordinates):
        def embedding(axis):
            phase = axis * (2 * math.pi) / self.dim_t
            return torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=4).flatten(3)
        y, x = embedding(coordinates[..., 1:2]), embedding(coordinates[..., :1])
        return self.half_pose_ref_point_head(torch.cat((y, x), dim=3))


class PoseDecoder(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward, adapter_hidden_dim, num_queries, eval_spatial_size,
                 num_classes=2, num_points=4, group_sizes=(2, 2), feat_strides=(8, 16, 32),
                 nhead=8, reg_max=32, reg_scale=4, activation="relu", num_body_points=17):
        super().__init__()
        if hidden_dim % nhead or hidden_dim % 4:
            raise ValueError("Pose dimension must be divisible by the head count and by four")
        self.num_queries = num_queries
        self.num_body_points = num_body_points
        self.nhead = nhead
        self.reg_max = reg_max
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale], dtype=torch.float32), requires_grad=False)
        self.decoder = PoseGroups(hidden_dim, dim_feedforward, adapter_hidden_dim, nhead, len(feat_strides),
                                  num_points, activation, group_sizes)
        self.keypoint_embedding = nn.Embedding(num_body_points, hidden_dim)
        self.instance_embedding = nn.Embedding(1, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_class_embed = nn.Linear(hidden_dim, num_classes)
        self.enc_pose_embed = MLP(hidden_dim, 2 * hidden_dim, 2 * num_body_points, 4)
        self.class_embed = nn.ModuleList(nn.Linear(hidden_dim, num_classes) for _ in group_sizes)
        self.pose_embed = nn.ModuleList(MLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3) for _ in group_sizes)
        self.lqe_embed = nn.ModuleList(PoseQuality(num_body_points) for _ in group_sizes)
        self.pre_pose_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        anchors = []
        for stride in feat_strides:
            h, w = [size // stride for size in eval_spatial_size]
            y, x = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w), indexing="ij")
            xy = (torch.stack([x, y], dim=-1) + 0.5) / torch.tensor([w, h], dtype=torch.float32)
            anchors.append(xy.reshape(1, h*w, 2))
        coords = torch.cat(anchors, 1)
        self.register_buffer("valid_mask", ~((coords > 0.01) & (coords < 0.99)).all(-1, keepdim=True))
        self.register_buffer("anchors", torch.log(coords / (1-coords)))

    def forward(self, feats):
        memory, shapes = flatten_features(feats)
        proposal = self.enc_output_norm(self.enc_output(memory.masked_fill(self.valid_mask, 0.0)))
        index = self.enc_out_class_embed(proposal).max(-1).values.topk(self.num_queries, dim=1).indices.unsqueeze(-1)
        top = proposal.gather(1, index.expand(-1, -1, proposal.shape[-1]))
        batch, count = top.shape[:2]
        anchors = self.anchors.expand(batch, -1, -1).gather(1, index.expand(-1, -1, 2))
        offsets = self.enc_pose_embed(top).reshape(batch, count, self.num_body_points, 2)
        coords = (offsets + anchors.unsqueeze(-2)).sigmoid()
        references = torch.cat([coords.mean(2, keepdim=True), coords], dim=2)
        target = self.tgt_embed.weight[None, :, None, :].expand(batch, -1, -1, -1)
        joints = self.keypoint_embedding.weight[None, None, :, :] + target
        person = self.instance_embedding.weight[None, None, :, :].expand(batch, count, -1, -1)
        output = torch.cat([person, joints], dim=2)
        values = memory.unflatten(2, (self.nhead, -1)).permute(0, 2, 3, 1).flatten(0, 1)
        values = values.split([h*w for h, w in shapes], dim=-1)
        bins = regression_bins(self.reg_max, self.up, self.reg_scale)
        previous, corners = 0, 0
        for layer_idx, group_idx in enumerate(self.decoder.layer_to_group):
            coords = references[:, :, 1:]
            pos = self.decoder.query_position(coords)
            output = self.decoder.layers[group_idx](output, pos, references, values, shapes)
            output = self.decoder.loop_adapters[layer_idx](output)
            joints = output[:, :, 1:]
            if layer_idx == 0:
                base = (self.pre_pose_embed(joints) + inverse_sigmoid(coords, 1e-3)).sigmoid()
            corners = self.pose_embed[group_idx](joints + previous) + corners
            coords = base + expectation(corners, bins) / self.reg_scale.abs()
            references = torch.cat([coords.mean(2, keepdim=True), coords], dim=2)
            previous = joints
        final_group = self.decoder.layer_to_group[-1]
        scores = self.lqe_embed[final_group](self.class_embed[final_group](output[:, :, 0]), coords, feats[0])
        return {"pred_logits": scores, "pred_keypoints": coords.flatten(-2)}
