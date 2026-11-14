"""Thí nghiệm khói: không cần GPU, không cần repo paper. Chứng minh harness chạy từ đầu đến cuối (local, Kaggle, Colab).

Giữ lại trong mọi project: `make local EXP=smoke` và `make push EXP=smoke` là cách rẻ nhất để kiểm tra
môi trường (pip, dataset gắn vào kernel, secret) trước khi tốn GPU cho thí nghiệm thật.
"""

from __future__ import annotations

import random


def run(ctx) -> dict:
    n = ctx.limit or int(ctx.cfg.get("n", 100))
    rng = random.Random(ctx.seed)
    xs = [rng.random() for _ in range(n)]
    preds = [x > 0.5 for x in xs]
    gold = [(x + rng.gauss(0, ctx.cfg.get("noise", 0.1))) > 0.5 for x in xs]
    acc = sum(p == g for p, g in zip(preds, gold)) / n
    ctx.save_json("predictions.json", [{"x": x, "pred": p, "gold": g} for x, p, g in zip(xs, preds, gold)])
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        gpu = None
    return {"metrics": {"accuracy": acc}, "n": n, "extra": {"gpu": gpu, "data": str(ctx.data_dir())}}
