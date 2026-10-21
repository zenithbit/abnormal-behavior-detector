"""Logistic-regression baseline on pooled segment features (no temporal model).

Each video becomes one vector: mean + max over its segments (frame stream) and/or over all
valid (segment, person) slots (person stream). Two independent classifiers:
anomaly (binary, all videos) and behavior (multiclass, abnormal videos with a known class).
`predict` returns the same dict as `trainer.predict`, so `trainer.evaluate` scores both alike.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import FeatureDataset
from .metrics import select_threshold


def pooled_features(ds: FeatureDataset, input_mode: str) -> np.ndarray:
    rows = []
    for it in ds.items:
        parts = []
        if input_mode in {"frame", "both"}:
            f = it["frame"].numpy()
            parts += [f.mean(0), f.max(0)]
        if input_mode in {"person", "both"}:
            p, m = it["person"].numpy(), it["person_mask"].numpy()
            valid = p[m]  # [N, D] over all segments and tracked persons
            if len(valid):
                parts += [valid.mean(0), valid.max(0)]
            else:  # no person ever tracked
                parts += [np.zeros(p.shape[-1], p.dtype)] * 2
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def _labels(ds: FeatureDataset) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([int(it["anomaly"]) for it in ds.items]),
            np.array([int(it["behavior"]) for it in ds.items]))


def _logreg(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def fit(cfg: dict, train_ds: FeatureDataset, val_ds: FeatureDataset, classes: list[str]) -> dict:
    """Fit both classifiers on train, pick the anomaly threshold on val. Returns a model dict."""
    mode, c = cfg["model"]["input"], float(cfg["linear"]["C"])
    x = pooled_features(train_ds, mode)
    anom, beh = _labels(train_ds)
    known = beh >= 0
    model = {
        "input": mode, "behavior_classes": classes,
        "anomaly": _logreg(c).fit(x, anom),
        "behavior": _logreg(c).fit(x[known], beh[known]),
    }
    val_pred = predict(model, val_ds)
    model["threshold"] = select_threshold(val_pred["anomaly"], val_pred["score"], cfg["train"].get("threshold", "youden"))
    return model


def predict(model: dict, ds: FeatureDataset) -> dict:
    x = pooled_features(ds, model["input"])
    anom, beh = _labels(ds)
    clf = model["behavior"]
    # classes absent from train get probability 0
    prob = np.zeros((len(x), len(model["behavior_classes"])))
    prob[:, clf.classes_] = clf.predict_proba(x)
    return {"video_id": [it["video_id"] for it in ds.items], "anomaly": anom, "behavior": beh,
            "score": model["anomaly"].predict_proba(x)[:, 1], "behavior_prob": prob}


def save(model: dict, path: Path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
