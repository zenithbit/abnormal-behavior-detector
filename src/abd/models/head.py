"""Multi-person aggregation + temporal Transformer + anomaly (MIL) and behavior heads."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PersonAttentionPool(nn.Module):
    """Pool a variable set of person features per segment with a learned query (masked)."""

    def __init__(self, dim: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(dim, d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.empty = nn.Parameter(torch.zeros(d_model))  # used when a segment has nobody

    def forward(self, person: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # person [B, S, K, D], mask [B, S, K] -> [B, S, d_model]
        b, s, k, _ = person.shape
        x = self.proj(person).reshape(b * s, k, -1)
        m = mask.reshape(b * s, k)
        has_any = m.any(dim=1)
        # rows with no person would make softmax NaN; give them a dummy valid slot, then overwrite
        safe = m.clone()
        safe[~has_any, 0] = True
        pooled, _ = self.attn(self.query.expand(b * s, -1, -1), x, x, key_padding_mask=~safe)
        pooled = pooled.squeeze(1)
        pooled = torch.where(has_any[:, None], pooled, self.empty.expand_as(pooled))
        return pooled.reshape(b, s, -1)


class AnomalyBehaviorModel(nn.Module):
    """
    Input  : segment features of one video (frame stream, person stream, or both).
    Output : per-segment anomaly logits [B, S], video anomaly logit [B] (top-k MIL),
             behavior logits [B, C] (pooled with anomaly-score attention).
    """

    def __init__(self, feat_dim: int, num_behaviors: int, input_mode: str = "frame",
                 d_model: int = 256, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.3, topk_ratio: float = 0.125, max_segments: int = 512,
                 behavior_pool: str = "attn"):
        super().__init__()
        if input_mode not in {"frame", "person", "both"}:
            raise ValueError(f"input_mode must be frame|person|both, got {input_mode!r}")
        self.input_mode = input_mode
        self.topk_ratio = topk_ratio

        if input_mode in {"frame", "both"}:
            self.frame_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d_model))
        if input_mode in {"person", "both"}:
            self.person_norm = nn.LayerNorm(feat_dim)
            self.person_pool = PersonAttentionPool(feat_dim, d_model, n_heads, dropout)
        if input_mode == "both":
            self.fuse = nn.Linear(2 * d_model, d_model)

        self.pos = nn.Parameter(torch.randn(1, max_segments, d_model) * 0.02)
        self.in_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        # n_layers=0 -> no temporal mixing (per-segment MLP); a useful ablation on small data
        self.temporal = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False) if n_layers else nn.Identity()
        if behavior_pool not in {"attn", "mean"}:
            raise ValueError(f"behavior_pool must be attn|mean, got {behavior_pool!r}")
        self.behavior_pool = behavior_pool
        self.norm = nn.LayerNorm(d_model)
        self.anomaly_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.behavior_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, max(1, num_behaviors)))

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.input_mode in {"frame", "both"}:
            parts.append(self.frame_proj(batch["frame"]))
        if self.input_mode in {"person", "both"}:
            parts.append(self.person_pool(self.person_norm(batch["person"]), batch["person_mask"]))
        x = self.fuse(torch.cat(parts, -1)) if len(parts) == 2 else parts[0]
        x = self.in_drop(x + self.pos[:, : x.shape[1]])
        return self.norm(self.temporal(x))

    def topk(self, s: int) -> int:
        return max(1, math.ceil(s * self.topk_ratio))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.encode(batch)                                   # [B, S, d]
        seg_logits = self.anomaly_head(h).squeeze(-1)            # [B, S]
        k = self.topk(seg_logits.shape[1])
        video_logit = seg_logits.topk(k, dim=1).values.mean(1)   # MIL: mean of top-k segments
        if self.behavior_pool == "attn":
            attn = torch.softmax(seg_logits.detach(), dim=1)     # focus behavior on suspicious parts
            pooled = (attn.unsqueeze(-1) * h).sum(1)
        else:
            pooled = h.mean(1)
        behavior_logits = self.behavior_head(pooled)
        return {"segment_logits": seg_logits, "video_logit": video_logit, "behavior_logits": behavior_logits}


def multitask_loss(out: dict[str, torch.Tensor], anomaly: torch.Tensor, behavior: torch.Tensor,
                   lambda_anomaly: float = 1.0, lambda_behavior: float = 1.0,
                   pos_weight: torch.Tensor | None = None,
                   behavior_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE on the video anomaly logit + CE on behavior for abnormal videos only (behavior=-1 ignored)."""
    l_anom = F.binary_cross_entropy_with_logits(out["video_logit"], anomaly.float(), pos_weight=pos_weight)
    valid = behavior >= 0
    if valid.any():
        l_beh = F.cross_entropy(out["behavior_logits"][valid], behavior[valid], weight=behavior_weight)
    else:
        l_beh = out["behavior_logits"].sum() * 0.0
    total = lambda_anomaly * l_anom + lambda_behavior * l_beh
    return total, {"loss": total.item(), "loss_anomaly": l_anom.item(), "loss_behavior": l_beh.item()}
