"""Turn per-segment anomaly scores into candidate abnormal time ranges (pseudo temporal labels).

These are *candidates*, not ground truth: they require high score AND temporal consistency
(several consecutive segments after smoothing). Inspect them visually before training on them.
"""
from __future__ import annotations

import numpy as np


def smooth(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    pad = window // 2
    padded = np.pad(scores, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(scores)]


def candidate_segments(scores: np.ndarray, threshold: float, min_consecutive: int = 2,
                       window: int = 3) -> list[tuple[int, int]]:
    """Runs [first, last] (inclusive segment indices) of smoothed score >= threshold, length >= min_consecutive."""
    above = smooth(scores, window) >= threshold
    runs, start = [], None
    for i, a in enumerate(np.append(above, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_consecutive:
                runs.append((start, i - 1))
            start = None
    return runs


def runs_to_time(runs: list[tuple[int, int]], bounds: np.ndarray, fps: float, scores: np.ndarray) -> list[dict]:
    return [
        {
            "start_sec": round(float(bounds[a, 0]) / fps, 2),
            "end_sec": round(float(bounds[b, 1]) / fps, 2),
            "segments": [int(a), int(b)],
            "mean_score": round(float(scores[a:b + 1].mean()), 4),
        }
        for a, b in runs
    ]
