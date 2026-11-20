"""CLIP + Graph (Nhiệm vụ 2, cấp 1): top-k CLIP → graph_score từ GNN → xếp hạng lại α·cos + (1−α)·graph_score.

Một lượt chạy = một cấu hình đồ thị/GNN × nhiều seed (cfg.seeds, mặc định 3) → báo cáo mean ± std (đề mục 6).
Ablation = cùng file này, chỉ đổi cfg.graph.variant (configs/experiments/ablation_*.yaml kế thừa clip_graph.yaml).

Quy trình (đúng đề mục 4.2):
  1. Nạp dữ liệu (thật: cfg.data; giả: data=null), dựng đồ thị theo variant.
  2. CLIP top-k ứng viên cho caption val/test (cố định, không phụ thuộc seed).
  3. Mỗi seed: huấn luyện GNN tương phản trên caption TRAIN, early stopping theo MRR(graph) trên val.
  4. Chọn α trên val (lưới cfg.alpha_grid), báo cáo test; lưu đường cong độ nhạy α.
  5. Giải thích + fidelity cho cfg.n_explain truy vấn test (seed đầu) → explanations.json cho Nhiệm vụ 3.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.env import set_seed
from ..data.contract import load, resolve_data
from ..graph.build import build_graph
from ..graph.explain import explain
from ..graph.gnn import GraphEncoder, graph_scores, train
from ..retrieval.core import (clip_topk, combine, grouped_metrics, metrics_from_ranks, ranks_of_gold,
                              select_alpha)


def _split_queries(b, split, k, limit):
    rows = np.array([b.image_ids.index(i) for i in b.splits[split]])   # đỉnh ảnh của split (ảnh = đỉnh 0..N_img-1)
    row_pos = {int(r): j for j, r in enumerate(rows)}
    qi = b.captions_of(split, limit)
    q = b.caption_emb[qi]
    gold = np.array([b.image_ids.index(b.captions[i]["image_id"]) for i in qi])
    cos, cand_local = clip_topk(q, b.image_emb[rows], k)
    cand = rows[cand_local]                                             # chỉ số đỉnh toàn cục
    groups = [b.captions[i].get("group") for i in qi]
    return {"qi": qi, "q": q, "gold": gold, "cos": cos, "cand": cand, "groups": groups}


def _mean_std(dicts):
    """[{g: {metric: v}}] → ({g: {metric: mean}}, {g: {metric: std}})."""
    mean, std = {}, {}
    for g in dicts[0]:
        mean[g], std[g] = {}, {}
        for k in dicts[0][g]:
            vals = [d[g][k] for d in dicts]
            mean[g][k] = float(np.mean(vals))
            if k != "n":
                std[g][k] = float(np.std(vals))
    return mean, std


def run(ctx) -> dict:
    cfg = ctx.cfg
    gcfg, mcfg, tcfg = cfg.get("graph", {}), dict(cfg.get("gnn", {})), dict(cfg.get("train", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":  # nhiều luồng làm đồ thị nhỏ chậm hơn ~15 lần (đo trên máy 16 nhân)
        torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    k = int(cfg.get("topk", 100))
    seeds = [ctx.seed] if ctx.args.get("seed") is not None else list(cfg.get("seeds", [0, 1, 2]))
    if ctx.limit:  # lượt thử: ít truy vấn, ít epoch, 1 seed
        tcfg["epochs"] = min(int(tcfg.get("epochs", 20)), 2)
        seeds = seeds[:1]

    data_dir = resolve_data(cfg, ctx.root / "cache")
    b = load(data_dir, max_images=cfg.get("max_images"))
    print(f"  dữ liệu: {data_dir} · {len(b.image_ids)} ảnh · {len(b.captions)} caption · {len(b.labels)} nhãn · device {device}")

    val = _split_queries(b, "val", k, ctx.limit)
    test = _split_queries(b, "test", k, ctx.limit)
    tr_qi = b.captions_of("train", ctx.limit * 50 if ctx.limit else None)
    tr_q = b.caption_emb[tr_qi]
    tr_rows = np.array([b.image_ids.index(b.captions[i]["image_id"]) for i in tr_qi])
    grid = list(np.round(np.arange(0, 1.0001, float(cfg.get("alpha_step", 0.1))), 4))

    clip_test = grouped_metrics(ranks_of_gold(test["cos"], test["cand"], test["gold"]), test["groups"])
    per_seed, graph_only, curves, train_info, graph_stats = [], [], [], [], None
    explanations = []
    for si, seed in enumerate(seeds):
        set_seed(seed)
        g = build_graph(b, gcfg.get("variant", "full"), int(gcfg.get("min_rel_count", 1)), seed=seed,
                        rel_merge=gcfg.get("rel_merge", "directed"))
        graph_stats = graph_stats or g.stats
        print(f"  seed {seed}: đồ thị {g.stats}")
        model = GraphEncoder(b.image_emb.shape[1], **mcfg)

        def val_fn(m):
            gs = graph_scores(m, g.data, val["q"], val["cand"], device)
            return metrics_from_ranks(ranks_of_gold(gs, val["cand"], val["gold"]))["MRR"]

        info = train(model, g.data, g.n_img, tr_q, tr_rows, val_fn, tcfg, device)
        gs_val = graph_scores(model, g.data, val["q"], val["cand"], device)
        alpha, curve = select_alpha(val["cos"], gs_val, val["cand"], val["gold"], grid)
        gs_test = graph_scores(model, g.data, test["q"], test["cand"], device)
        r = ranks_of_gold(combine(test["cos"], gs_test, alpha), test["cand"], test["gold"])
        m = grouped_metrics(r, test["groups"])
        per_seed.append(m)
        graph_only.append(grouped_metrics(ranks_of_gold(gs_test, test["cand"], test["gold"]), test["groups"]))
        curves.append({"seed": seed, "alpha": alpha, "val_curve": curve})
        train_info.append({"seed": seed, "alpha": alpha, **{k2: v for k2, v in info.items() if k2 != "history"},
                           "history": info["history"]})
        print(f"  seed {seed}: α={alpha} test {m['all']}")

        if si == 0:
            ctx.save_json("ranks_test.json", [{"caption_id": b.captions[i]["caption_id"], "rank_clip_graph": int(x)}
                                              for i, x in zip(test["qi"], r)])
            final = combine(test["cos"], gs_test, alpha)
            for j in range(min(int(cfg.get("n_explain", 10)), len(test["qi"]))):
                top1 = int(test["cand"][j][int(np.argmax(final[j]))])
                e = explain(model, g, test["q"][j], top1, top_m=int(cfg.get("explain_top_m", 3)), seed=seed, device=device)
                cap = b.captions[test["qi"][j]]
                explanations.append({"caption_id": cap["caption_id"], "query": cap.get("text"), "group": cap.get("group"),
                                     "top1_image_id": b.image_ids[top1], "correct": top1 == int(test["gold"][j]), **e})
            if cfg.get("save_model", True):
                torch.save(model.state_dict(), ctx.run_dir / "model.pt")

    mean, std = _mean_std(per_seed)
    gmean, _ = _mean_std(graph_only)
    ctx.save_json("alpha_curves.json", curves)
    ctx.save_json("train_info.json", train_info)
    ctx.save_json("explanations.json", explanations)
    fid = [e["fidelity"] for e in explanations]
    metrics = {
        "clip": clip_test,                         # baseline trên đúng các truy vấn này
        "clip_graph": mean, "clip_graph_std": std,  # mean ± std qua seed
        "graph_only": gmean,                       # α = 0, để thấy riêng sức mạnh của nhúng đồ thị
        "alpha_mean": float(np.mean([c["alpha"] for c in curves])),
        "fidelity": {"delta_expl": float(np.mean([f["delta_explanation"] for f in fid])) if fid else None,
                     "delta_random": float(np.mean([f["delta_random_mean"] for f in fid])) if fid else None,
                     "sparsity": float(np.mean([f["sparsity"] for f in fid])) if fid else None},
    }
    return {"metrics": metrics, "n": mean["all"]["n"],
            "extra": {"seeds": seeds, "graph": graph_stats, "data": str(data_dir), "device": str(device),
                      "per_seed": per_seed}}
