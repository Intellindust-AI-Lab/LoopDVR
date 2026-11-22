"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

Adapted the COCO evaluators for standalone val2017 evaluation.
---------------------------------------------------------------------------------
Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Based on the TorchVision detection references (https://github.com/pytorch/vision).
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..categories import COCO_CATEGORIES
from ..inference import image_transform
from ..models.postprocess import postprocess

IOU_TYPE = {"det": "bbox", "seg": "segm", "pose": "keypoints"}


class ValidationImages(Dataset):
    def __init__(self, folder, images, transform):
        self.folder = folder
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        with Image.open(self.folder / info["file_name"]) as image:
            if image.size != (info["width"], info["height"]):
                raise ValueError(f"Image dimensions disagree with annotations: {info['file_name']}")
            size = torch.tensor(image.size)
            return self.transform(image.convert("RGB")), size, info["id"]


def coco_records(task, image_id, prediction):
    scores = prediction["scores"].tolist()
    labels = prediction["labels"].tolist()
    if task == "pose":
        keypoints = prediction["keypoints"].reshape(len(scores), -1).tolist()
        return [dict(image_id=image_id, category_id=label, score=score, keypoints=points)
                for label, score, points in zip(labels, scores, keypoints)]
    if task == "seg":
        # pycocotools expects H x W x K in Fortran order. Transpose before the
        # device transfer so the NumPy view is ready for the encoder.
        masks = prediction["masks"][:, 0].transpose(1, 2).to(torch.uint8).contiguous()
        values = mask_utils.encode(masks.cpu().numpy().T)
        field = "segmentation"
    else:
        values = box_convert(prediction["boxes"], "xyxy", "xywh").tolist()
        field = "bbox"
    return [{"image_id": image_id, "category_id": COCO_CATEGORIES[label][0],
             "score": score, field: value} for label, score, value in zip(labels, scores, values)]


def evaluate_chunk(evaluator, ground_truth, image_ids, records):
    # Keep only COCO's per-image matches; full-resolution mask RLEs can be
    # released before reading the next chunk.
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.cocoDt = ground_truth.loadRes(records)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
    shape = (len(evaluator.params.catIds), len(evaluator.params.areaRng), len(image_ids))
    return np.asarray(evaluator.evalImgs, dtype=object).reshape(shape)


@torch.inference_mode()
def evaluate_coco(model, config, root, *, batch_size=4, workers=4):
    task = config["task"]
    root = Path(root)
    split = "person_keypoints" if task == "pose" else "instances"
    ground_truth = COCO(str(root / "annotations" / f"{split}_val2017.json"))
    image_ids = sorted(ground_truth.getImgIds())
    if not image_ids:
        raise ValueError("Validation annotations contain no images")
    device = next(model.parameters()).device
    dataset = ValidationImages(root / "val2017", ground_truth.loadImgs(image_ids), image_transform(config))
    batches = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    evaluator = COCOeval(ground_truth, iouType=IOU_TYPE[task])
    chunks, ids, records = [], [], []
    completed = 0
    for images, sizes, batch_ids in batches:
        outputs = model(images.to(device, non_blocking=True))
        predictions = postprocess(outputs, sizes.to(device), task)
        for image_id, prediction in zip(batch_ids.tolist(), predictions):
            ids.append(image_id)
            records.extend(coco_records(task, image_id, prediction))
        completed += len(batch_ids)
        if len(ids) >= 64 or completed == len(dataset):
            chunks.append(evaluate_chunk(evaluator, ground_truth, ids, records))
            ids, records = [], []
            print(f"Evaluated {completed}/{len(dataset)}", flush=True)
    evaluator.params.imgIds = image_ids
    evaluator._paramsEval = copy.deepcopy(evaluator.params)
    evaluator.evalImgs = np.concatenate(chunks, axis=2).ravel().tolist()
    evaluator.accumulate()
    evaluator.summarize()
    return dict(model_id=config["model_id"], iou_type=IOU_TYPE[task], images=completed,
                stats=evaluator.stats.tolist())
