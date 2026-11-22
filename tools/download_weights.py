"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors.
SPDX-License-Identifier: Apache-2.0

Download the v1.0 inference checkpoints into the repository's weights directory.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve


RELEASE = "https://github.com/Siruuui/LoopDVR_Weights/releases/download/v1.0"
WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


def download_weights():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for model in ("loopdet", "loopseg", "looppose"):
        for size in ("s", "m", "l", "x"):
            path = WEIGHTS / f"{model}-{size}.pth"
            if path.is_file() and path.stat().st_size > 0:
                continue
            print(f"Downloading {path.name}...", flush=True)
            with TemporaryDirectory(dir=WEIGHTS) as tmp:
                partial = Path(tmp) / path.name
                urlretrieve(f"{RELEASE}/{path.name}", partial)
                if partial.stat().st_size == 0:
                    raise OSError(f"Empty checkpoint: {path.name}")
                partial.replace(path)


if __name__ == "__main__":
    download_weights()
