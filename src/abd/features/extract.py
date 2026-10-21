"""Per-video feature extraction for the frame stream and the person (track) stream.

Output per video (`<video_id>.npz`):
    frame        [S, D]       full-frame clip feature per segment
    person       [S, K, D]    feature of up to K person tracks per segment (zeros where absent)
    person_mask  [S, K] bool  which person slots are real
    person_ids   [S, K]       track id per slot (-1 where absent)
    bounds       [S, 2]       segment (start, end) frame indices
    fps          scalar
"""
from __future__ import annotations

import numpy as np

from ..data.video import read_frames, segment_bounds, segment_clip_indices
from .backbone import ClipEncoder


def select_tracks(tracks: np.ndarray, start: int, end: int, k: int) -> list[np.ndarray]:
    """Up to k tracks present in [start, end), ranked by total box area (presence x size)."""
    seg = tracks[(tracks[:, 0] >= start) & (tracks[:, 0] < end)]
    if len(seg) == 0:
        return []
    area = (seg[:, 4] - seg[:, 2]) * (seg[:, 5] - seg[:, 3])
    ids = np.unique(seg[:, 1])
    score = {tid: area[seg[:, 1] == tid].sum() for tid in ids}
    ranked = sorted(ids, key=lambda t: -score[t])[:k]
    # Return the full track (not only the in-segment part) so boxes can be interpolated
    # for clip frames that fall between two detector frames.
    return [tracks[tracks[:, 1] == tid] for tid in ranked]


def person_crop_boxes(track: np.ndarray, frame_idx: np.ndarray, margin: float) -> np.ndarray:
    """Square, fixed-size boxes [T, 4] following the person's centre at the given frames.

    Size is constant within the clip (largest box side + margin) so the backbone sees motion
    of the person, not zoom jitter from the detector.
    """
    order = np.argsort(track[:, 0])
    f, boxes = track[order, 0], track[order, 2:6]
    cx = np.interp(frame_idx, f, (boxes[:, 0] + boxes[:, 2]) / 2)
    cy = np.interp(frame_idx, f, (boxes[:, 1] + boxes[:, 3]) / 2)
    w = np.interp(frame_idx, f, boxes[:, 2] - boxes[:, 0])
    h = np.interp(frame_idx, f, boxes[:, 3] - boxes[:, 1])
    side = max(w.max(), h.max()) * (1 + margin)
    half = side / 2
    return np.stack([cx - half, cy - half, cx + half, cy + half], axis=1)


def crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop with zero padding where the box leaves the frame (keeps the crop square)."""
    h, w = frame.shape[:2]
    # round the origin and the size separately so equal-size boxes give equal-size crops
    x1, y1 = int(round(box[0])), int(round(box[1]))
    x2 = x1 + max(1, int(round(box[2] - box[0])))
    y2 = y1 + max(1, int(round(box[3] - box[1])))
    out = np.zeros((y2 - y1, x2 - x1, 3), dtype=frame.dtype)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return out


def extract_video(
    video_path: str,
    num_frames: int,
    fps: float,
    orig_width: int,
    encoder: ClipEncoder,
    num_segments: int,
    clip_len: int,
    max_frame_side: int,
    tracks: np.ndarray | None = None,
    max_persons: int = 4,
    crop_margin: float = 0.2,
) -> dict[str, np.ndarray]:
    idx = segment_clip_indices(num_frames, num_segments, clip_len)
    bounds = segment_bounds(num_frames, num_segments)
    frames = read_frames(video_path, idx, max_frame_side)

    out: dict[str, np.ndarray] = {"bounds": bounds, "fps": np.float32(fps)}
    out["frame"] = encoder.encode([np.stack([frames[i] for i in seg]) for seg in idx])

    if tracks is not None:
        scale = next(iter(frames.values())).shape[1] / orig_width  # tracks are in original pixels
        person = np.zeros((num_segments, max_persons, encoder.dim), dtype=np.float32)
        mask = np.zeros((num_segments, max_persons), dtype=bool)
        pids = np.full((num_segments, max_persons), -1, dtype=np.int32)
        clips, slots = [], []
        for s, ((a, b), seg) in enumerate(zip(bounds, idx)):
            for k, tr in enumerate(select_tracks(tracks, a, b, max_persons)):
                boxes = person_crop_boxes(tr, seg, crop_margin) * scale
                clips.append(np.stack([crop(frames[i], bx) for i, bx in zip(seg, boxes)]))
                slots.append((s, k, int(tr[0, 1])))
        if clips:
            feats = encoder.encode(clips)
            for (s, k, tid), f in zip(slots, feats):
                person[s, k], mask[s, k], pids[s, k] = f, True, tid
        out.update(person=person, person_mask=mask, person_ids=pids)
    return out
