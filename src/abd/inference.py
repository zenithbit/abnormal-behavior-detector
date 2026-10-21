"""Full pipeline on one video (track -> extract -> model), shared by `abd infer` and the web demo."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from .config import resolve_device
from .train.pseudo_label import candidate_segments, runs_to_time


class Analyzer:
    """Loads the checkpoint, tracker and backbone once; `analyze` can then be called per video."""

    def __init__(self, cfg: dict, ckpt_path: str | Path):
        from .detection.tracker import PersonTracker
        from .features.backbone import ClipEncoder
        from .train.trainer import load_checkpoint

        self.device = resolve_device(cfg["train"]["device"])
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        # Sampling / backbone / tracker settings must match training, so they come from the checkpoint.
        # Machine-specific settings (devices, batch size, detector weights path) come from this run's config.
        self.cfg = copy.deepcopy(ckpt["config"])
        self.cfg["features"].update(device=cfg["features"]["device"], batch_size=cfg["features"]["batch_size"])
        self.cfg["tracking"]["model"] = cfg["tracking"]["model"]
        self.threshold = ckpt["threshold"]
        self.classes = ckpt["behavior_classes"]
        self.mode = self.cfg["model"]["input"]
        t, f = self.cfg["tracking"], self.cfg["features"]
        self.tracker = PersonTracker(t["model"], t["tracker"], t["conf"], t["fps"])
        self.encoder = ClipEncoder(f["backbone"], resolve_device(f["device"]), f["batch_size"])

    def analyze(self, video: str | Path, render: str | Path | None = None) -> dict:
        from .data.index import probe
        from .detection.tracker import render_tracks
        from .features.extract import extract_video

        video = str(video)
        meta = probe(Path(video))
        if meta is None:
            raise ValueError(f"cannot read {video}")
        n, fps, w, _h = meta
        tracks = None
        if self.mode != "frame" or render:
            tracks = self.tracker.track_video(video, self.cfg["tracking"]["min_track_len"])
        smp, f = self.cfg["sampling"], self.cfg["features"]
        feats = extract_video(video, n, fps, w, self.encoder, smp["num_segments"], smp["clip_len"],
                              smp["max_frame_side"], tracks if self.mode != "frame" else None,
                              f["max_persons"], f["crop_margin"])
        batch = {k: torch.from_numpy(v)[None].to(self.device) for k, v in feats.items()
                 if k in {"frame", "person", "person_mask"}}
        with torch.no_grad():
            o = self.model(batch)
        score = float(torch.sigmoid(o["video_logit"])[0])
        seg = torch.sigmoid(o["segment_logits"])[0].cpu().numpy()
        probs = torch.softmax(o["behavior_logits"], -1)[0].cpu().numpy()
        thr = self.threshold
        is_anom = score >= thr
        behavior = self.classes[int(probs.argmax())] if is_anom and self.classes else "Normal"
        bounds = feats["bounds"]
        result = {
            "video": video,
            "anomaly": bool(is_anom),
            "behavior": behavior,
            "confidence": round(float(probs.max()) * score if is_anom else 1 - score, 4),
            "anomaly_score": round(score, 4),
            "threshold": round(thr, 4),
            "behavior_probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "segments": runs_to_time(candidate_segments(seg, thr), bounds, fps, seg),
            "timeline": [{"start_sec": round(float(a) / fps, 2), "end_sec": round(float(b) / fps, 2),
                          "score": round(float(s), 4)} for (a, b), s in zip(bounds, seg)],
        }
        if render:
            def caption(i):
                s = int(np.clip(np.searchsorted(bounds[:, 1], i, side="right"), 0, len(seg) - 1))
                col = (0, 0, 255) if seg[s] >= thr else (0, 200, 0)
                return f"{behavior if is_anom else 'Normal'} | segment score {seg[s]:.2f}", col

            render_tracks(video, tracks, render, caption)
        return result
