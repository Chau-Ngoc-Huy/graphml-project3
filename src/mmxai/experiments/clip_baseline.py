"""Baseline CLIP đóng băng (baseline CLIP): xếp hạng ảnh test chỉ bằng cosine CLIP. Không có tham số học → không phụ thuộc seed.

Dùng cùng loader, cùng split, cùng metric với clip_graph để bảng so sánh công bằng.
"""

from __future__ import annotations

import numpy as np

from ..core.env import ROOT
from ..data.contract import load, resolve_data
from ..retrieval.core import clip_topk, grouped_metrics, ranks_of_gold


def evaluate_clip(b, split: str, k: int, limit: int | None = None):
    rows = np.array([b.image_ids.index(i) for i in b.splits[split]])
    row_of = {int(r): j for j, r in enumerate(rows)}
    qi = b.captions_of(split, limit)
    q = b.caption_emb[qi]
    gold = np.array([row_of[b.image_ids.index(b.captions[i]["image_id"])] for i in qi])
    cos, cand = clip_topk(q, b.image_emb[rows], k)
    r = ranks_of_gold(cos, cand, gold)
    return grouped_metrics(r, [b.captions[i].get("group") for i in qi]), r, qi


def run(ctx) -> dict:
    cfg = ctx.cfg
    data_dir = resolve_data(cfg, ctx.root / "cache")
    b = load(data_dir, max_images=cfg.get("max_images"))
    out = {}
    for split in ("val", "test"):
        m, r, qi = evaluate_clip(b, split, int(cfg.get("topk", 100)), ctx.limit)
        out[split] = m
        if split == "test":
            ctx.save_json("ranks_test.json", [{"caption_id": b.captions[i]["caption_id"], "rank": int(x)} for i, x in zip(qi, r)])
    return {"metrics": out, "n": out["test"]["all"]["n"], "extra": {"data": str(data_dir), "n_images": len(b.image_ids)}}
