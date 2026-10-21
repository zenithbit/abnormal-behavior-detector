"""Frozen spatio-temporal backbones (Kinetics-400 pretrained) that map a clip to one vector."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

# torchvision's Kinetics-400 normalisation (shared by s3d / r3d_18)
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
INPUT_SIZE = 224

FEATURE_DIMS = {"s3d": 1024, "r3d_18": 512}


def build_backbone(name: str) -> nn.Module:
    from torchvision.models import video as tvv

    if name == "s3d":
        model = tvv.s3d(weights=tvv.S3D_Weights.KINETICS400_V1)
        model.classifier = nn.Identity()  # forward then averages features over (T, H, W) -> [B, 1024]
    elif name == "r3d_18":
        model = tvv.r3d_18(weights=tvv.R3D_18_Weights.KINETICS400_V1)
        model.fc = nn.Identity()          # -> [B, 512]
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(FEATURE_DIMS)}")
    return model.eval()


class ClipEncoder:
    """Encode RGB uint8 clips [T, H, W, 3] into feature vectors, in batches."""

    def __init__(self, name: str, device: torch.device, batch_size: int = 8):
        self.name = name
        self.dim = FEATURE_DIMS[name]
        self.device = device
        self.batch_size = batch_size
        self.model = build_backbone(name).to(device)
        self.mean = torch.tensor(KINETICS_MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(KINETICS_STD, device=device).view(1, 3, 1, 1, 1)

    @staticmethod
    def prepare(clip: np.ndarray) -> np.ndarray:
        # Squash to a square instead of centre-cropping: surveillance frames are wide and
        # the event can happen near the border. Person crops are already square.
        return np.stack([cv2.resize(f, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA) for f in clip])

    @torch.inference_mode()
    def encode(self, clips: list[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(clips), self.batch_size):
            batch = np.stack([self.prepare(c) for c in clips[i:i + self.batch_size]])  # B T H W C
            x = torch.from_numpy(batch).to(self.device).permute(0, 4, 1, 2, 3).float().div_(255)
            x = (x - self.mean) / self.std
            out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
