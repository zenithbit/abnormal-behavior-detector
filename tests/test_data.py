import numpy as np

from abd.data.index import read_manifest, scan_dataset, write_manifest
from abd.data.split import stratified_split
from abd.data.video import read_frames, segment_bounds, segment_clip_indices


def test_scan_labels_and_skips_broken(toy_dataset, tmp_path):
    recs = scan_dataset(toy_dataset)
    assert len(recs) == 18  # broken.mp4 skipped
    by_beh = {r.behavior for r in recs}
    assert by_beh == {"Normal", "Fighting", "Theft"}
    assert all((r.anomaly == 0) == (r.behavior == "Normal") for r in recs)
    write_manifest(recs, tmp_path / "m.csv")
    assert read_manifest(tmp_path / "m.csv") == recs


def test_split_is_by_video_and_stratified(toy_dataset):
    recs = scan_dataset(toy_dataset)
    s = stratified_split(recs, 0.15, 0.15, seed=1)
    sets = [set(s[k]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == {r.video_id for r in recs}
    beh = {r.video_id: r.behavior for r in recs}
    for k in ("val", "test"):
        assert {beh[v] for v in s[k]} == {"Normal", "Fighting", "Theft"}
    assert s["behavior_classes"] == ["Fighting", "Theft"]
    assert stratified_split(recs, 0.15, 0.15, seed=1) == s  # deterministic


def test_segments_cover_video():
    b = segment_bounds(100, 8)
    assert b[0, 0] == 0 and b[-1, 1] == 100
    assert (b[1:, 0] == b[:-1, 1]).all()
    idx = segment_clip_indices(100, 8, 16)
    assert idx.shape == (8, 16)
    assert ((idx >= b[:, :1]) & (idx < b[:, 1:])).all()
    # video shorter than the number of segments still yields valid indices
    idx = segment_clip_indices(5, 32, 16)
    assert idx.min() >= 0 and idx.max() <= 4


def test_read_frames_handles_overrun(toy_dataset):
    path = toy_dataset / "normal" / "video_00.mp4"
    frames = read_frames(path, np.array([0, 10, 59, 500]), max_side=80)
    assert set(frames) == {0, 10, 59, 500}
    assert frames[0].shape == (60, 80, 3)
    assert np.array_equal(frames[500], frames[59])  # past the end -> last real frame
