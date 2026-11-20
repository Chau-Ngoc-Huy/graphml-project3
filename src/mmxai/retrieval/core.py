"""Truy vấn text→image: top-k theo cosine CLIP, xếp hạng lại α·cos + (1−α)·graph_score, metric R@K / MRR / nDCG@10.

Dùng chung cho mọi mô hình (CLIP, CLIP+Graph, phương pháp cấp 2) để cùng một cách đánh giá trên cùng split.
Với tập ảnh lớn có FAISS index (bundle VG có sẵn): thay `clip_topk` bằng index.search — đầu ra giữ nguyên (scores, idx).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

KS = (1, 5, 10)


def clip_topk(q: np.ndarray, gallery: np.ndarray, k: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """q [Q,D], gallery [M,D] đã L2-normalize → (scores [Q,k], idx [Q,k]) giảm dần."""
    k = min(k, gallery.shape[0])
    g = torch.from_numpy(gallery)
    S, I = [], []
    for s in range(0, len(q), chunk):
        sim = torch.from_numpy(q[s:s + chunk]) @ g.T
        v, i = sim.topk(k, dim=1)
        S.append(v.numpy()); I.append(i.numpy())
    return np.concatenate(S), np.concatenate(I)


def combine(cos: np.ndarray, graph: np.ndarray | None, alpha: float) -> np.ndarray:
    return cos if graph is None else alpha * cos + (1 - alpha) * graph


def ranks_of_gold(scores: np.ndarray, cand_idx: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Hạng (1-based) của ảnh đúng trong danh sách ứng viên đã chấm điểm; 0 = không nằm trong top-k ứng viên."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.take_along_axis(cand_idx, order, axis=1)
    hit = ranked == gold[:, None]
    r = hit.argmax(1) + 1
    r[~hit.any(1)] = 0
    return r


def metrics_from_ranks(r: np.ndarray) -> dict:
    found = r > 0
    out = {f"R@{k}": float(np.mean(found & (r <= k))) for k in KS}
    out["MRR"] = float(np.mean(np.where(found, 1.0 / np.maximum(r, 1), 0.0)))
    out["nDCG@10"] = float(np.mean([1 / math.log2(x + 1) if 0 < x <= 10 else 0.0 for x in r]))
    out["n"] = int(len(r))
    return out


def grouped_metrics(r: np.ndarray, groups: list[str | None]) -> dict:
    """Metric toàn bộ + theo nhóm truy vấn (direct / indirect nếu captions có trường group)."""
    out = {"all": metrics_from_ranks(r)}
    by = defaultdict(list)
    for i, g in enumerate(groups):
        if g:
            by[g].append(i)
    for g, idx in sorted(by.items()):
        out[g] = metrics_from_ranks(r[idx])
    return out


def select_alpha(cos, graph, cand_idx, gold, grid, key: str = "MRR") -> tuple[float, list[dict]]:
    """Chọn α trên VALIDATION (đề mục 6). Trả về α tốt nhất và đường cong độ nhạy."""
    curve = []
    for a in grid:
        m = metrics_from_ranks(ranks_of_gold(combine(cos, graph, a), cand_idx, gold))
        curve.append({"alpha": round(float(a), 4), **m})
    best = max(curve, key=lambda c: (c[key], c["alpha"]))  # hòa → ưu tiên α lớn (gần CLIP)
    return best["alpha"], curve
