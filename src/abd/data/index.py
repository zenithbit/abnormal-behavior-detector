"""Scan a dataset laid out as `normal/*.mp4` and `abnormal/<Behavior>/*.mp4` into a manifest CSV."""
from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg", ".webm", ".m4v"}
NORMAL = "Normal"


@dataclass
class VideoRecord:
    video_id: str
    path: str          # relative to dataset root
    anomaly: int       # 0 normal, 1 abnormal
    behavior: str      # "Normal" or the abnormal class name
    num_frames: int
    fps: float
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


def probe(path: Path) -> tuple[int, float, int, int] | None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok or n <= 0:
            return None
        return n, fps, w, h
    finally:
        cap.release()


def _label_for(rel: Path) -> tuple[int, str] | None:
    top = rel.parts[0].lower()
    if top == "normal":
        return 0, NORMAL
    if top == "abnormal":
        # abnormal/<Behavior>/video.mp4 ; a video directly under abnormal/ has unknown class
        return 1, rel.parts[1] if len(rel.parts) > 2 else "Abnormal"
    return None


def scan_dataset(root: str | Path) -> list[VideoRecord]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    records: list[VideoRecord] = []
    skipped: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS):
        rel = path.relative_to(root)
        label = _label_for(rel)
        if label is None:
            skipped.append(f"{rel} (not under normal/ or abnormal/)")
            continue
        meta = probe(path)
        if meta is None:
            skipped.append(f"{rel} (unreadable)")
            continue
        anomaly, behavior = label
        video_id = "__".join(rel.with_suffix("").parts)
        records.append(VideoRecord(video_id, rel.as_posix(), anomaly, behavior, *meta))
    for s in skipped:
        log.warning("skipped %s", s)
    return records


def write_manifest(records: list[VideoRecord], out: str | Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(VideoRecord)])
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def read_manifest(path: str | Path) -> list[VideoRecord]:
    with open(path, newline="") as f:
        return [
            VideoRecord(
                video_id=row["video_id"], path=row["path"], anomaly=int(row["anomaly"]),
                behavior=row["behavior"], num_frames=int(row["num_frames"]), fps=float(row["fps"]),
                width=int(row["width"]), height=int(row["height"]),
            )
            for row in csv.DictReader(f)
        ]


def summarize(records: list[VideoRecord]) -> str:
    by_class = Counter(r.behavior for r in records)
    dur = Counter()
    for r in records:
        dur[r.behavior] += r.duration
    lines = [f"{'behavior':<24}{'videos':>8}{'hours':>9}{'avg sec':>9}"]
    for cls, n in sorted(by_class.items(), key=lambda kv: (kv[0] != NORMAL, kv[0])):
        lines.append(f"{cls:<24}{n:>8}{dur[cls] / 3600:>9.2f}{dur[cls] / n:>9.1f}")
    lines.append(f"{'TOTAL':<24}{len(records):>8}{sum(dur.values()) / 3600:>9.2f}")
    return "\n".join(lines)
