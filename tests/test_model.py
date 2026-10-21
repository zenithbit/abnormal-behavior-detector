import numpy as np
import pytest
import torch

from abd.features.extract import crop, person_crop_boxes, select_tracks
from abd.models.head import AnomalyBehaviorModel, multitask_loss
from abd.train.metrics import best_f1_threshold, end_to_end_metrics
from abd.train.pseudo_label import candidate_segments


def _batch(b=3, s=8, k=4, d=32):
    mask = torch.zeros(b, s, k, dtype=torch.bool)
    mask[0, :, :2] = True
    mask[1, ::2, 0] = True  # some segments with nobody
    return {"frame": torch.randn(b, s, d), "person": torch.randn(b, s, k, d), "person_mask": mask}


@pytest.mark.parametrize("mode", ["frame", "person", "both"])
def test_model_shapes_and_no_nan(mode):
    m = AnomalyBehaviorModel(feat_dim=32, num_behaviors=3, input_mode=mode, d_model=32, n_heads=4)
    out = m(_batch())
    assert out["segment_logits"].shape == (3, 8)
    assert out["video_logit"].shape == (3,)
    assert out["behavior_logits"].shape == (3, 3)
    assert all(torch.isfinite(v).all() for v in out.values())  # sample 2 has no person at all


def test_behavior_loss_ignores_normal():
    m = AnomalyBehaviorModel(feat_dim=32, num_behaviors=3, d_model=32)
    out = m(_batch())
    _, parts = multitask_loss(out, torch.tensor([0, 0, 0]), torch.tensor([-1, -1, -1]))
    assert parts["loss_behavior"] == 0.0
    loss, parts = multitask_loss(out, torch.tensor([1, 0, 1]), torch.tensor([2, -1, 0]))
    loss.backward()
    assert parts["loss_behavior"] > 0


def test_select_tracks_and_crops():
    # track 1 big & present, track 2 tiny, track 3 outside the segment
    rows = [(f, 1, 10, 10, 60, 110, 0.9) for f in range(0, 20, 2)]
    rows += [(f, 2, 0, 0, 5, 5, 0.9) for f in range(0, 20, 2)]
    rows += [(f, 3, 0, 0, 100, 100, 0.9) for f in range(40, 50, 2)]
    tracks = np.array(rows, np.float32)
    sel = select_tracks(tracks, 0, 20, k=4)
    assert [int(t[0, 1]) for t in sel] == [1, 2]
    boxes = person_crop_boxes(sel[0], np.arange(0, 16), margin=0.2)
    side = boxes[:, 2] - boxes[:, 0]
    assert np.allclose(side, side[0]) and np.allclose(side, boxes[:, 3] - boxes[:, 1])  # fixed square
    frame = np.ones((50, 50, 3), np.uint8)
    c = crop(frame, np.array([-10, -10, 20, 20]))
    assert c.shape == (30, 30, 3) and c[:10].sum() == 0 and c[10:, 10:].all()
    # sub-pixel offsets must not change the crop size (clips are stacked)
    shapes = {crop(frame, np.array([x, x, x + 20.0, x + 20.0])).shape for x in (0.4, 0.5, 0.6, 1.49)}
    assert shapes == {(20, 20, 3)}


def test_candidate_segments_requires_consecutive():
    s = np.array([0.1, 0.9, 0.1, 0.1, 0.8, 0.9, 0.95, 0.9, 0.1, 0.1])
    assert candidate_segments(s, 0.5, min_consecutive=2, window=1) == [(4, 7)]


def test_threshold_and_end_to_end():
    y = np.array([0, 0, 1, 1])
    thr = best_f1_threshold(y, np.array([0.1, 0.2, 0.7, 0.9]))
    assert 0.2 < thr <= 0.7
    e = end_to_end_metrics(y, np.array([-1, -1, 0, 1]), np.array([0.1, 0.2, 0.7, 0.9]),
                           np.array([0, 0, 0, 1]), thr, ["Fighting", "Theft"])
    assert e["accuracy"] == 1.0
