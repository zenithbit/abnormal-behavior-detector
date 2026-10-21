"""YOLO person detection + multi-object tracking. YOLO only answers "where are the people"."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..data.video import iter_frames

# Columns of a tracks array: frame_idx, track_id, x1, y1, x2, y2, conf (original-resolution pixels)
TRACK_COLS = ("frame", "track_id", "x1", "y1", "x2", "y2", "conf")
PERSON_CLASS = 0


class PersonTracker:
    def __init__(self, model: str = "yolo11n.pt", tracker: str = "bytetrack.yaml",
                 conf: float = 0.3, fps: float = 10, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.tracker = tracker
        self.conf = conf
        self.fps = fps
        self.device = device

    def track_video(self, path: str | Path, min_track_len: int = 1) -> np.ndarray:
        rows = []
        first = True
        for frame_idx, frame in iter_frames(path, self.fps):
            res = self.model.track(
                frame, persist=not first, classes=[PERSON_CLASS], conf=self.conf,
                tracker=self.tracker, device=self.device, verbose=False,
            )[0]
            first = False
            boxes = res.boxes
            if boxes is None or boxes.id is None:
                continue
            ids = boxes.id.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for tid, (x1, y1, x2, y2), c in zip(ids, xyxy, confs):
                rows.append((frame_idx, tid, x1, y1, x2, y2, c))
        tracks = np.asarray(rows, dtype=np.float32).reshape(-1, len(TRACK_COLS))
        return filter_short_tracks(tracks, min_track_len)


def filter_short_tracks(tracks: np.ndarray, min_len: int) -> np.ndarray:
    if len(tracks) == 0 or min_len <= 1:
        return tracks
    ids, counts = np.unique(tracks[:, 1], return_counts=True)
    keep = set(ids[counts >= min_len].tolist())
    return tracks[np.isin(tracks[:, 1], list(keep))]


def save_tracks(tracks: np.ndarray, path: str | Path) -> None:
    np.save(path, tracks)


def load_tracks(path: str | Path) -> np.ndarray:
    return np.load(path).reshape(-1, len(TRACK_COLS))


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id))
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def render_tracks(video_path: str | Path, tracks: np.ndarray, out_path: str | Path,
                  caption: Callable[[int], tuple[str, tuple[int, int, int]]] | None = None) -> None:
    """Draw tracked boxes onto the video for visual QA.

    `caption(frame_idx) -> (text, bgr_color)` adds a banner, e.g. the anomaly score.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(tracks[:, 0]).astype(int):
        by_frame[f] = tracks[tracks[:, 0] == f]
    tracked_frames = np.array(sorted(by_frame)) if by_frame else np.array([], dtype=int)
    i = 0
    current = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # tracks exist only on subsampled frames; hold the latest boxes in between
        if len(tracked_frames):
            pos = np.searchsorted(tracked_frames, i, side="right") - 1
            current = by_frame[tracked_frames[pos]] if pos >= 0 else None
        if current is not None:
            for _, tid, x1, y1, x2, y2, _c in current:
                col = _color(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(frame, f"ID {int(tid)}", (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if caption:
            text, col = caption(i)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
