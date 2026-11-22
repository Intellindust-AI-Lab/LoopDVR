"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors.
SPDX-License-Identifier: Apache-2.0

Run a LoopDVR model on one image.
"""

import argparse
from pathlib import Path
import sys

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopdvr.build import build_model, load_config
from loopdvr.inference import predict_image
from loopdvr.visualization import draw_predictions


def main():
    parser = argparse.ArgumentParser(description="Run a LoopDVR model on one image.")
    for name in ("config", "weights", "input", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.4, help="Minimum score for display")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(args.config)
    model = build_model(config, args.weights, args.device)
    with Image.open(args.input) as image:
        prediction = predict_image(model, config, image)
        result = draw_predictions(image, prediction, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Inference completed successfully. Output saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
