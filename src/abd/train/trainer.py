from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.head import AnomalyBehaviorModel, multitask_loss
from .dataset import FeatureDataset, collate, to_device
from .metrics import anomaly_metrics, behavior_metrics, end_to_end_metrics, select_threshold

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict, feat_dim: int, num_behaviors: int) -> AnomalyBehaviorModel:
    m = cfg["model"]
    return AnomalyBehaviorModel(
        feat_dim=feat_dim, num_behaviors=num_behaviors, input_mode=m["input"], d_model=m["d_model"],
        n_layers=m["n_layers"], n_heads=m["n_heads"], dropout=m["dropout"], topk_ratio=m["topk_ratio"],
        max_segments=max(512, cfg["sampling"]["num_segments"]),
    )


@torch.no_grad()
def predict(model: AnomalyBehaviorModel, ds: FeatureDataset, device: torch.device, batch_size: int = 64) -> dict:
    model.eval()
    out = {"video_id": [], "anomaly": [], "behavior": [], "score": [], "segment_scores": [], "behavior_prob": []}
    for batch in DataLoader(ds, batch_size=batch_size, collate_fn=collate):
        o = model(to_device(batch, device))
        out["video_id"] += batch["video_id"]
        out["anomaly"].append(batch["anomaly"].numpy())
        out["behavior"].append(batch["behavior"].numpy())
        out["score"].append(torch.sigmoid(o["video_logit"]).cpu().numpy())
        out["segment_scores"].append(torch.sigmoid(o["segment_logits"]).cpu().numpy())
        out["behavior_prob"].append(torch.softmax(o["behavior_logits"], -1).cpu().numpy())
    return {k: (v if k == "video_id" else np.concatenate(v)) for k, v in out.items()}


def evaluate(pred: dict, classes: list[str], threshold: float) -> dict:
    beh_pred = pred["behavior_prob"].argmax(1)
    return {
        "anomaly": anomaly_metrics(pred["anomaly"], pred["score"], threshold),
        "behavior": behavior_metrics(pred["behavior"], beh_pred, classes),
        "end_to_end": end_to_end_metrics(pred["anomaly"], pred["behavior"], pred["score"], beh_pred, threshold, classes),
    }


def _selection_key(pred: dict) -> tuple[float, float]:
    """Model selection on val: (anomaly ROC-AUC + behavior macro-F1, -val loss).

    ROC-AUC rather than PR-AUC: with ~90% abnormal videos PR-AUC is near 1 for any model.

    The anomaly task often saturates early; including behavior F1 and breaking ties on
    loss keeps the behavior head from being frozen at an untrained epoch.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    y, s = pred["anomaly"], pred["score"]
    score = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float(((s >= 0.5) == y).mean())
    beh = pred["behavior"]
    valid = beh >= 0
    eps = 1e-7
    loss = -np.mean(y * np.log(s + eps) + (1 - y) * np.log(1 - s + eps))
    if valid.any():
        score += float(f1_score(beh[valid], pred["behavior_prob"][valid].argmax(1), average="macro"))
        loss += -np.mean(np.log(pred["behavior_prob"][valid, beh[valid]] + eps))
    return round(score, 4), -float(loss)


def train(cfg: dict, train_ds: FeatureDataset, val_ds: FeatureDataset, classes: list[str],
          device: torch.device, run_dir: Path) -> dict:
    tc = cfg["train"]
    seed_everything(tc["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, train_ds.feat_dim, len(classes)).to(device)

    # Class balancing: anomaly pos_weight and inverse-frequency behavior weights
    n_pos = sum(int(it["anomaly"]) for it in train_ds.items)
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor(n_neg / max(1, n_pos), device=device)
    beh_counts = Counter(int(it["behavior"]) for it in train_ds.items if it["behavior"] >= 0)
    beh_w = torch.tensor([1.0 / beh_counts[i] if beh_counts[i] else 0.0 for i in range(len(classes))])
    beh_w = (beh_w / beh_w[beh_w > 0].mean()).to(device) if (beh_w > 0).any() else None

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, collate_fn=collate, drop_last=False)
    total_steps = tc["epochs"] * max(1, len(loader))
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    best, best_epoch, history = (-np.inf, -np.inf), -1, []
    for epoch in range(tc["epochs"]):
        model.train()
        agg = Counter()
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            loss, parts = multitask_loss(out, batch["anomaly"], batch["behavior"], tc["lambda_anomaly"],
                                         tc["lambda_behavior"], pos_weight, beh_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg.update({k: v * len(batch["video_id"]) for k, v in parts.items()})
        train_stats = {k: v / len(train_ds) for k, v in agg.items()}

        val_pred = predict(model, val_ds, device)
        key = _selection_key(val_pred)
        history.append({"epoch": epoch, **train_stats, "val_score": key[0], "val_loss": -key[1]})
        log.info("epoch %3d  loss %.4f (anom %.4f beh %.4f)  val score %.4f  val loss %.4f", epoch,
                 train_stats["loss"], train_stats["loss_anomaly"], train_stats["loss_behavior"], key[0], -key[1])
        if key > best:
            best, best_epoch = key, epoch
            save_checkpoint(run_dir / "best.pt", model, cfg, classes, train_ds.feat_dim)
        elif epoch - best_epoch >= tc["patience"]:
            log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
            break

    model = load_checkpoint(run_dir / "best.pt", device)[0]
    val_pred = predict(model, val_ds, device)
    threshold = select_threshold(val_pred["anomaly"], val_pred["score"], tc.get("threshold", "youden"))
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    ckpt["threshold"] = threshold
    torch.save(ckpt, run_dir / "best.pt")
    result = {"best_epoch": best_epoch, "threshold": threshold, "val": evaluate(val_pred, classes, threshold)}
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return result


def save_checkpoint(path: Path, model: AnomalyBehaviorModel, cfg: dict, classes: list[str], feat_dim: int) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg, "behavior_classes": classes,
                "feat_dim": feat_dim, "threshold": 0.5}, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[AnomalyBehaviorModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], ckpt["feat_dim"], len(ckpt["behavior_classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
