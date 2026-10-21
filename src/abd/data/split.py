"""Video-level stratified split. Clips are never split across sets (no leakage)."""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from .index import NORMAL, VideoRecord

log = logging.getLogger(__name__)


def stratified_split(
    records: list[VideoRecord], val_ratio: float, test_ratio: float, seed: int
) -> dict:
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_class[r.behavior].append(r.video_id)

    split = {"train": [], "val": [], "test": []}
    for cls in sorted(by_class):
        ids = sorted(by_class[cls])
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            log.warning("class %r has only %d video(s); all go to train", cls, n)
            split["train"] += ids
            continue
        n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
        n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test:n_test + n_val]
        split["train"] += ids[n_test + n_val:]

    classes = sorted(c for c in by_class if c != NORMAL)
    return {"seed": seed, "behavior_classes": classes, **{k: sorted(v) for k, v in split.items()}}


def save_split(split: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def kfold_splits(records: list[VideoRecord], k: int, val_ratio: float, seed: int) -> list[dict]:
    """Stratified k-fold over videos: fold i is the test set, val is carved from the rest.

    Every video is tested exactly once, which matters when a class has only a few dozen videos.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_class[r.behavior].append(r.video_id)
    folds: list[list[str]] = [[] for _ in range(k)]
    offset = 0
    for cls in sorted(by_class):
        ids = sorted(by_class[cls])
        rng.shuffle(ids)
        for j, vid in enumerate(ids):
            folds[(j + offset) % k].append(vid)  # rotate start so small classes don't pile into fold 0
        offset += len(ids)

    classes = sorted(c for c in by_class if c != NORMAL)
    out = []
    for i in range(k):
        rest = [v for j, f in enumerate(folds) if j != i for v in f]
        rest_ids = set(rest)
        inner = stratified_split([r for r in records if r.video_id in rest_ids], val_ratio, 0.0, seed + i)
        out.append({"seed": seed, "fold": i, "behavior_classes": classes,
                    "train": inner["train"], "val": inner["val"], "test": sorted(folds[i])})
    return out
