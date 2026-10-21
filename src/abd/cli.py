"""Command line entry point: `abd [--config cfg.yaml] [--set a.b=value]... <command>`.

Pipeline order:  index -> split -> track -> extract -> train -> evaluate / pseudo-label / infer
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_config, resolve_device, work_path

log = logging.getLogger("abd")


# ---------------------------------------------------------------- helpers
def _readable(path: Path, need: str | None = None) -> bool:
    """True if a previous run left a complete .npy/.npz here (and it has key `need`, for .npz)."""
    if not path.exists():
        return False
    try:
        z = np.load(path)
        return need is None or need in z.files
    except Exception:
        log.warning("unreadable %s (interrupted run?) -> recomputing", path)
        return False


def _manifest(cfg):
    from .data.index import read_manifest

    path = work_path(cfg, "manifest.csv")
    if not path.exists():
        raise SystemExit(f"{path} not found; run `abd index` first")
    return read_manifest(path)


def _split(cfg):
    from .data.split import load_split

    path = work_path(cfg, "splits.json")
    if not path.exists():
        raise SystemExit(f"{path} not found; run `abd split` first")
    return load_split(path)


def _feature_dir(cfg) -> Path:
    d = work_path(cfg, "features", cfg["features"]["backbone"], "x").parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tracks_dir(cfg) -> Path:
    d = work_path(cfg, "tracks", "x").parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _video_path(cfg, rec) -> Path:
    return Path(cfg["paths"]["dataset_root"]) / rec.path


def _tracker(cfg):
    from .detection.tracker import PersonTracker

    t = cfg["tracking"]
    return PersonTracker(t["model"], t["tracker"], t["conf"], t["fps"])


def _encoder(cfg):
    from .features.backbone import ClipEncoder

    f = cfg["features"]
    return ClipEncoder(f["backbone"], resolve_device(f["device"]), f["batch_size"])


def _subset(records, ids):
    ids = set(ids)
    return [r for r in records if r.video_id in ids]


# ---------------------------------------------------------------- commands
def cmd_index(cfg, args):
    from .data.index import scan_dataset, summarize, write_manifest

    records = scan_dataset(cfg["paths"]["dataset_root"])
    out = work_path(cfg, "manifest.csv")
    write_manifest(records, out)
    print(summarize(records))
    print(f"\nmanifest -> {out}")


def cmd_split(cfg, args):
    from collections import Counter

    from .data.split import save_split, stratified_split

    records = _manifest(cfg)
    s = cfg["split"]
    split = stratified_split(records, s["val_ratio"], s["test_ratio"], s["seed"])
    out = work_path(cfg, "splits.json")
    save_split(split, out)
    beh = {r.video_id: r.behavior for r in records}
    for name in ("train", "val", "test"):
        c = Counter(beh[v] for v in split[name])
        print(f"{name:<6}{len(split[name]):>6}  " + ", ".join(f"{k}:{v}" for k, v in sorted(c.items())))
    print(f"behavior classes: {split['behavior_classes']}\nsplit -> {out}")


def cmd_track(cfg, args):
    from .detection.tracker import render_tracks, save_tracks

    records = _manifest(cfg)
    if args.only:
        records = [r for r in records if r.video_id in set(args.only)]
    tracker = _tracker(cfg)
    out_dir = _tracks_dir(cfg)
    vis_dir = work_path(cfg, "vis", "x").parent
    rendered = 0
    for rec in tqdm(records, desc="track"):
        out = out_dir / f"{rec.video_id}.npy"
        if args.overwrite or not _readable(out):
            tracks = tracker.track_video(_video_path(cfg, rec), cfg["tracking"]["min_track_len"])
            save_tracks(tracks, out)
        if rendered < args.render:
            vis_dir.mkdir(parents=True, exist_ok=True)
            render_tracks(_video_path(cfg, rec), np.load(out), vis_dir / f"{rec.video_id}_tracks.mp4")
            rendered += 1
    print(f"tracks -> {out_dir}" + (f"\nvisualisations -> {vis_dir}" if rendered else ""))


def cmd_extract(cfg, args):
    from .detection.tracker import load_tracks
    from .features.extract import extract_video

    records = _manifest(cfg)
    if args.only:
        records = [r for r in records if r.video_id in set(args.only)]
    f, smp = cfg["features"], cfg["sampling"]
    use_person = "person" in f["streams"]
    encoder = _encoder(cfg)
    out_dir = _feature_dir(cfg)
    log.info("backbone %s on %s, streams %s", f["backbone"], encoder.device, f["streams"])
    for rec in tqdm(records, desc="extract"):
        out = out_dir / f"{rec.video_id}.npz"
        # a frame-only file is incomplete once the person stream is requested
        if not args.overwrite and _readable(out, "person" if use_person else "frame"):
            continue
        tracks = None
        if use_person:
            tp = _tracks_dir(cfg) / f"{rec.video_id}.npy"
            if not tp.exists():
                raise SystemExit(f"missing {tp}; run `abd track` first (or set features.streams=[frame])")
            tracks = load_tracks(tp)
        feats = extract_video(
            str(_video_path(cfg, rec)), rec.num_frames, rec.fps, rec.width, encoder,
            smp["num_segments"], smp["clip_len"], smp["max_frame_side"],
            tracks, f["max_persons"], f["crop_margin"],
        )
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **feats)
        tmp.replace(out)
    print(f"features -> {out_dir}")


def _datasets(cfg, split, names, input_mode):
    from .train.dataset import FeatureDataset

    records = _manifest(cfg)
    return [FeatureDataset(_subset(records, split[n]), _feature_dir(cfg), split["behavior_classes"], input_mode)
            for n in names]


def cmd_train(cfg, args):
    from .train.trainer import evaluate, load_checkpoint, predict, train

    split = _split(cfg)
    train_ds, val_ds, test_ds = _datasets(cfg, split, ("train", "val", "test"), cfg["model"]["input"])
    device = resolve_device(cfg["train"]["device"])
    run_dir = work_path(cfg, "runs", args.name, "x").parent
    log.info("train %d / val %d / test %d videos on %s -> %s", len(train_ds), len(val_ds), len(test_ds), device, run_dir)
    result = train(cfg, train_ds, val_ds, split["behavior_classes"], device, run_dir)
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    result["test"] = evaluate(predict(model, test_ds, device), split["behavior_classes"], ckpt["threshold"])
    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    _print_metrics(result)
    print(f"\nrun -> {run_dir}")


def cmd_cv(cfg, args):
    """Stratified k-fold CV: every video is tested once; reports mean +- std over folds."""
    from .data.split import kfold_splits
    from .train.dataset import FeatureDataset
    from .train.trainer import evaluate, load_checkpoint, predict, train

    records = _manifest(cfg)
    device = resolve_device(cfg["train"]["device"])
    folds = kfold_splits(records, args.folds, cfg["split"]["val_ratio"], cfg["split"]["seed"])
    base = work_path(cfg, "runs", args.name, "x").parent
    per_fold = []
    for fold in folds:
        classes = fold["behavior_classes"]
        train_ds, val_ds, test_ds = (
            FeatureDataset(_subset(records, fold[n]), _feature_dir(cfg), classes, cfg["model"]["input"])
            for n in ("train", "val", "test"))
        run_dir = base / f"fold{fold['fold']}"
        log.info("fold %d: train %d / val %d / test %d", fold["fold"], len(train_ds), len(val_ds), len(test_ds))
        result = train(cfg, train_ds, val_ds, classes, device, run_dir)
        model, ckpt = load_checkpoint(run_dir / "best.pt", device)
        result["test"] = evaluate(predict(model, test_ds, device), classes, ckpt["threshold"])
        with open(run_dir / "metrics.json", "w") as fh:
            json.dump(result, fh, indent=2)
        per_fold.append(result["test"])
    _cv_summary(per_fold, base)


def _cv_summary(per_fold, base: Path):
    keys = [("anomaly", "precision"), ("anomaly", "recall"), ("anomaly", "specificity"),
            ("anomaly", "balanced_accuracy"), ("anomaly", "f1"), ("anomaly", "roc_auc"),
            ("anomaly", "pr_auc"), ("behavior", "accuracy"), ("behavior", "macro_f1"),
            ("end_to_end", "accuracy"), ("end_to_end", "macro_f1")]
    summary = {}
    for task, metric in keys:
        vals = np.array([f[task].get(metric, np.nan) for f in per_fold], dtype=float)
        summary[f"{task}.{metric}"] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                                       "folds": vals.round(4).tolist()}
        print(f"{task + '.' + metric:<24} {np.nanmean(vals):.3f} +- {np.nanstd(vals):.3f}")
    with open(base / "cv_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\ncv -> {base}")


def cmd_linear(cfg, args):
    """Logistic-regression baseline: same split / threshold rule / metrics as `train` (or `cv` with --folds)."""
    from .data.split import kfold_splits
    from .train import linear
    from .train.dataset import FeatureDataset
    from .train.trainer import evaluate

    records = _manifest(cfg)
    mode = cfg["model"]["input"]
    if args.folds:
        folds = kfold_splits(records, args.folds, cfg["split"]["val_ratio"], cfg["split"]["seed"])
    else:
        folds = [_split(cfg)]
    base = work_path(cfg, "runs", args.name, "x").parent
    per_fold = []
    for fold in folds:
        classes = fold["behavior_classes"]
        train_ds, val_ds, test_ds = (FeatureDataset(_subset(records, fold[n]), _feature_dir(cfg), classes, mode)
                                     for n in ("train", "val", "test"))
        run_dir = base / f"fold{fold['fold']}" if args.folds else base
        run_dir.mkdir(parents=True, exist_ok=True)
        model = linear.fit(cfg, train_ds, val_ds, classes)
        linear.save(model, run_dir / "model.pkl")
        result = {"threshold": model["threshold"],
                  "val": evaluate(linear.predict(model, val_ds), classes, model["threshold"]),
                  "test": evaluate(linear.predict(model, test_ds), classes, model["threshold"])}
        with open(run_dir / "metrics.json", "w") as fh:
            json.dump(result, fh, indent=2)
        per_fold.append(result["test"])
        if not args.folds:
            _print_metrics(result)
            print(f"\nrun -> {run_dir}")
    if args.folds:
        _cv_summary(per_fold, base)


def _print_metrics(result):
    for name in ("val", "test"):
        if name not in result:
            continue
        a, b, e = result[name]["anomaly"], result[name]["behavior"], result[name]["end_to_end"]
        print(f"[{name}] anomaly  P {a['precision']:.3f} R {a['recall']:.3f} Spec {a['specificity']:.3f} F1 {a['f1']:.3f} "
              f"ROC-AUC {a['roc_auc']:.3f} PR-AUC {a['pr_auc']:.3f} (thr {a['threshold']:.3f})")
        if b.get("n"):
            print(f"[{name}] behavior acc {b['accuracy']:.3f} macro-F1 {b['macro_f1']:.3f} (n={b['n']})")
        print(f"[{name}] end-to-end acc {e['accuracy']:.3f} macro-F1 {e['macro_f1']:.3f}")


def cmd_evaluate(cfg, args):
    from .train.trainer import evaluate, load_checkpoint, predict

    device = resolve_device(cfg["train"]["device"])
    model, ckpt = load_checkpoint(args.ckpt, device)
    split = _split(cfg)
    (ds,) = _datasets(cfg, split, (args.split,), ckpt["config"]["model"]["input"])
    res = {args.split: evaluate(predict(model, ds, device), ckpt["behavior_classes"], ckpt["threshold"])}
    _print_metrics({"val" if args.split == "val" else "test": res[args.split]})
    print(json.dumps(res, indent=2) if args.json else "")


def cmd_pseudo_label(cfg, args):
    from .train.pseudo_label import candidate_segments, runs_to_time
    from .train.trainer import load_checkpoint, predict

    device = resolve_device(cfg["train"]["device"])
    model, ckpt = load_checkpoint(args.ckpt, device)
    split = _split(cfg)
    (ds,) = _datasets(cfg, split, (args.split,), ckpt["config"]["model"]["input"])
    pred = predict(model, ds, device)
    thr = args.threshold if args.threshold is not None else ckpt["threshold"]
    by_id = {r.video_id: r for r in _manifest(cfg)}
    out = {"threshold": thr, "min_consecutive": args.min_consecutive, "window": args.window,
           "note": "pseudo labels from a model, not ground truth", "videos": {}}
    for vid, anom, scores in zip(pred["video_id"], pred["anomaly"], pred["segment_scores"]):
        if not anom:
            continue
        z = np.load(_feature_dir(cfg) / f"{vid}.npz")
        runs = candidate_segments(scores, thr, args.min_consecutive, args.window)
        out["videos"][vid] = {"behavior": by_id[vid].behavior,
                              "segments": runs_to_time(runs, z["bounds"], float(z["fps"]), scores),
                              "segment_scores": np.round(scores, 4).tolist()}
    path = Path(args.ckpt).parent / f"pseudo_labels_{args.split}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    found = sum(1 for v in out["videos"].values() if v["segments"])
    print(f"{found}/{len(out['videos'])} abnormal videos got candidate segments -> {path}")


def cmd_infer(cfg, args):
    from .inference import Analyzer

    try:
        result = Analyzer(cfg, args.ckpt).analyze(args.video, args.render)
    except ValueError as e:
        raise SystemExit(str(e))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.render:
        print(f"rendered -> {args.render}")


# ---------------------------------------------------------------- argparse
def main(argv=None):
    p = argparse.ArgumentParser(prog="abd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML merged over configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable: --set model.input=both --set train.epochs=30")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="scan dataset -> manifest.csv + statistics")
    sub.add_parser("split", help="video-level stratified train/val/test split")

    t = sub.add_parser("track", help="YOLO person detection + tracking for every video")
    t.add_argument("--render", type=int, default=0, help="also render the first N videos with boxes")
    t.add_argument("--only", nargs="*", help="restrict to these video_ids")
    t.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("extract", help="backbone features per segment (frame / person streams)")
    e.add_argument("--only", nargs="*")
    e.add_argument("--overwrite", action="store_true")

    tr = sub.add_parser("train", help="train temporal transformer + heads on extracted features")
    tr.add_argument("--name", default="baseline")

    cv = sub.add_parser("cv", help="stratified k-fold cross-validation (more reliable on small data)")
    cv.add_argument("--name", default="cv")
    cv.add_argument("--folds", type=int, default=5)

    lr = sub.add_parser("linear", help="logistic-regression baseline on mean+max pooled features")
    lr.add_argument("--name", default="linear")
    lr.add_argument("--folds", type=int, default=0, help="k-fold CV instead of the fixed split")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--split", default="test", choices=["train", "val", "test"])
    ev.add_argument("--json", action="store_true")

    pl = sub.add_parser("pseudo-label", help="candidate abnormal time ranges for abnormal videos")
    pl.add_argument("--ckpt", required=True)
    pl.add_argument("--split", default="train", choices=["train", "val", "test"])
    pl.add_argument("--threshold", type=float)
    pl.add_argument("--min-consecutive", type=int, default=2)
    pl.add_argument("--window", type=int, default=3)

    inf = sub.add_parser("infer", help="run the full pipeline on one video")
    inf.add_argument("--video", required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--render", help="write an annotated mp4 here")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.set)
    {
        "index": cmd_index, "split": cmd_split, "track": cmd_track, "extract": cmd_extract,
        "train": cmd_train, "cv": cmd_cv, "linear": cmd_linear, "evaluate": cmd_evaluate, "pseudo-label": cmd_pseudo_label, "infer": cmd_infer,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
