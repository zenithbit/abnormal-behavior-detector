from pathlib import Path

import cv2
import numpy as np
import pytest


def write_video(path: Path, n_frames: int = 60, fast: bool = False, size=(160, 120), fps: int = 15, seed: int = 0):
    """Synthetic clip: a square that drifts slowly (normal) or jumps around erratically (abnormal)."""
    rng = np.random.default_rng(seed)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    x, y = w // 2, h // 2
    for _ in range(n_frames):
        frame = np.full((h, w, 3), 40, np.uint8)
        step = 25 if fast else 2
        x = int(np.clip(x + rng.integers(-step, step + 1), 10, w - 30))
        y = int(np.clip(y + rng.integers(-step, step + 1), 10, h - 30))
        cv2.rectangle(frame, (x, y), (x + 20, y + 20), (0, 200, 255) if fast else (200, 200, 200), -1)
        writer.write(frame)
    writer.release()


@pytest.fixture
def toy_dataset(tmp_path) -> Path:
    root = tmp_path / "Ano-AAD"
    (root / "normal").mkdir(parents=True)
    for cls in ("Fighting", "Theft"):
        (root / "abnormal" / cls).mkdir(parents=True)
    for i in range(6):
        write_video(root / "normal" / f"video_{i:02d}.mp4", fast=False, seed=i)
        write_video(root / "abnormal" / "Fighting" / f"video_{i:02d}.mp4", fast=True, seed=100 + i)
        write_video(root / "abnormal" / "Theft" / f"video_{i:02d}.mp4", fast=True, seed=200 + i)
    (root / "abnormal" / "Theft" / "broken.mp4").write_bytes(b"not a video")
    return root
