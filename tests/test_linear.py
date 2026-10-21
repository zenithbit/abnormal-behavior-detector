from types import SimpleNamespace

import numpy as np
import pytest
import torch

from abd.train import linear
from abd.train.trainer import evaluate


def _ds(n=30, s=8, k=4, d=16, seed=0):
    """Normal videos near 0; abnormal ones shifted along a class-specific axis. Video 0 has no person."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        anomaly, behavior = (0, -1) if i % 3 == 0 else (1, i % 2)
        shift = np.zeros(d, np.float32)
        if anomaly:
            shift[behavior] = 3.0
        mask = np.zeros((s, k), bool)
        if i:
            mask[:, :2] = True
        items.append({
            "video_id": f"v{i}",
            "frame": torch.from_numpy(rng.normal(size=(s, d)).astype(np.float32) + shift),
            "person": torch.from_numpy(rng.normal(size=(s, k, d)).astype(np.float32) + shift),
            "person_mask": torch.from_numpy(mask),
            "anomaly": torch.tensor(anomaly), "behavior": torch.tensor(behavior),
        })
    return SimpleNamespace(items=items)


@pytest.mark.parametrize("mode", ["frame", "person", "both"])
def test_linear_fit_predict_evaluate(mode, tmp_path):
    cfg = {"model": {"input": mode}, "linear": {"C": 1.0}, "train": {"threshold": "youden"}}
    classes = ["A", "B"]
    model = linear.fit(cfg, _ds(seed=0), _ds(seed=1), classes)
    linear.save(model, tmp_path / "model.pkl")
    pred = linear.predict(linear.load(tmp_path / "model.pkl"), _ds(seed=2))
    assert pred["behavior_prob"].shape == (30, 2)
    res = evaluate(pred, classes, model["threshold"])
    assert res["anomaly"]["roc_auc"] > 0.9
    assert res["behavior"]["accuracy"] > 0.9


def test_pooled_features_dims():
    ds = _ds(n=5)
    assert linear.pooled_features(ds, "frame").shape == (5, 32)
    assert linear.pooled_features(ds, "both").shape == (5, 64)
    assert np.isfinite(linear.pooled_features(ds, "person")).all()  # video 0 has no person
