"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors.
SPDX-License-Identifier: Apache-2.0

Evaluate a LoopDVR model on COCO val2017.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopdvr.build import build_model, load_config
from loopdvr.evaluation.coco import evaluate_coco


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LoopDVR model on COCO val2017.")
    for name in ("config", "weights", "coco-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Write COCO summary statistics as JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers nonnegative")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(args.config)
    model = build_model(config, args.weights, args.device)
    metrics = evaluate_coco(model, config, args.coco_root, batch_size=args.batch_size, workers=args.workers)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
