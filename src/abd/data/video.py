"""Video decoding and segment/clip sampling."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def segment_bounds(num_frames: int, num_segments: int) -> np.ndarray:
    """[S, 2] (start, end) frame indices, end exclusive; S equal-length segments covering the video."""
    edges = np.linspace(0, num_frames, num_segments + 1)
    starts = np.floor(edges[:-1]).astype(int)
    ends = np.maximum(np.floor(edges[1:]).astype(int), starts + 1)
    return np.stack([starts, np.minimum(ends, num_frames)], axis=1)


def segment_clip_indices(num_frames: int, num_segments: int, clip_len: int) -> np.ndarray:
    """[S, clip_len] frame indices: clip_len frames uniformly spread inside each segment."""
    bounds = segment_bounds(num_frames, num_segments)
    idx = [np.linspace(s, e - 1, clip_len).round().astype(int) for s, e in bounds]
    return np.clip(np.stack(idx), 0, num_frames - 1)


def _resize(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side:
        return frame
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def read_frames(path: str | Path, indices: np.ndarray, max_side: int | None = None) -> dict[int, np.ndarray]:
    """Decode only the requested frame indices (RGB uint8), in one sequential pass.

    Container frame counts are often slightly wrong; indices past the real end
    are filled with the last decoded frame.
    """
    wanted = set(int(i) for i in np.unique(indices))
    last_needed = max(wanted)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    last = None
    i = 0
    try:
        while i <= last_needed and cap.grab():
            if i in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    last = _resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), max_side)
                    out[i] = last
            i += 1
    finally:
        cap.release()
    if last is None:
        raise RuntimeError(f"could not decode any frame from {path}")
    fill = out[max(out)]
    for j in sorted(wanted - out.keys()):
        prev = [k for k in out if k < j]
        out[j] = out[max(prev)] if prev else fill
    return out


def iter_frames(path: str | Path, target_fps: float, max_side: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) subsampled to roughly `target_fps`."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, round(src_fps / target_fps))
    i = 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield i, _resize(frame, max_side)
            i += 1
    finally:
        cap.release()
