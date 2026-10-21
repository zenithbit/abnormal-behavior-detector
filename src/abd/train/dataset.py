from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.index import VideoRecord


class FeatureDataset(Dataset):
    """Pre-extracted segment features of a list of videos, held in memory."""

    def __init__(self, records: list[VideoRecord], feature_dir: str | Path, behavior_classes: list[str],
                 input_mode: str):
        self.records = records
        self.items = []
        cls_index = {c: i for i, c in enumerate(behavior_classes)}
        need_person = input_mode in {"person", "both"}
        missing = []
        for r in records:
            path = Path(feature_dir) / f"{r.video_id}.npz"
            if not path.exists():
                missing.append(r.video_id)
                continue
            z = np.load(path)
            if need_person and "person" not in z:
                raise ValueError(f"{path} has no person stream; re-run extract with streams [frame, person]")
            item = {
                "video_id": r.video_id,
                "frame": torch.from_numpy(z["frame"]),
                "anomaly": torch.tensor(r.anomaly),
                "behavior": torch.tensor(cls_index.get(r.behavior, -1) if r.anomaly else -1),
            }
            if need_person:
                item["person"] = torch.from_numpy(z["person"])
                item["person_mask"] = torch.from_numpy(z["person_mask"])
            self.items.append(item)
        if missing:
            raise FileNotFoundError(f"{len(missing)} video(s) have no features, e.g. {missing[:3]}; run `abd extract`")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]

    @property
    def feat_dim(self) -> int:
        return self.items[0]["frame"].shape[-1]


def collate(items: list[dict]) -> dict:
    out = {"video_id": [it["video_id"] for it in items]}
    for k in items[0]:
        if k != "video_id":
            out[k] = torch.stack([it[k] for it in items])
    return out


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
