"""CLIP + điểm đồ thị phụ thuộc truy vấn (graph/qa.py) trên bundle VG, đánh giá bằng mmxai_eval.Protocol.

Cùng giao thức với vg_graph: seed s = gallery s + khởi tạo s; α chọn bằng tune_alpha trên VAL; test chỉ báo cáo.
Khái niệm của caption lấy từ trường `concepts` trong captions.jsonl (đã chuẩn hóa về bộ nhãn object_vocab_train khi dựng bundle).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..core.env import set_seed
from ..data.bundle import load_bundle
from ..graph.build import build_graph
from ..graph.qa import QAData, QAScorer, qa_scores, train_qa
from .vg_graph import load_eval_module


def run(ctx) -> dict:
    cfg = ctx.cfg
    bundle = Path(cfg["data"])
    ev = load_eval_module(bundle)
    P = ev.Protocol.load(bundle)
    seeds = [ctx.seed] if ctx.args.get("seed") is not None else list(cfg.get("seeds", [0, 1, 2]))
    if ctx.limit:
        seeds = seeds[:1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    gcfg, mcfg, tcfg = cfg.get("graph", {}), dict(cfg.get("gnn", {})), dict(cfg.get("train", {}))
    if ctx.limit:
        tcfg["epochs"] = min(int(tcfg.get("epochs", 20)), 2)
    b = load_bundle(bundle, cfg.get("extra"))
    row_of = {iid: r for r, iid in enumerate(b.image_ids)}
    to_rows = np.vectorize(lambda i: row_of[int(i)])
    cap_concepts = {c["caption_id"]: c["concepts"] for c in b.captions}
    tr_qi = b.captions_of("train", ctx.limit * 200 if ctx.limit else None)
    tr_q = b.caption_emb[tr_qi]
    tr_cid = np.array([b.captions[i]["caption_id"] for i in tr_qi])
    tr_rows = np.array([row_of[b.captions[i]["image_id"]] for i in tr_qi])
    x_img = torch.from_numpy(b.image_emb).to(device)

    per_seed, full_runs, alphas, infos, stats = [], [], [], [], None
    for seed in seeds:
        set_seed(seed)
        g = build_graph(b, gcfg.get("variant", "full"), int(gcfg.get("min_rel_count", 1)), seed=seed,
                        rel_merge=gcfg.get("rel_merge", "directed"))
        qd = QAData(g, cap_concepts, device)
        stats = stats or {**g.stats, **qd.stats, "concepts_dropped": qd.n_dropped}
        print(f"  seed {seed}: đồ thị {g.stats} | qa {qd.stats}")
        model = QAScorer(b.image_emb.shape[1], **mcfg).to(device)
        cv = P.candidates("val", seed)

        def gs_for(c, m=model):
            return qa_scores(m, qd, x_img, P.cap[c["q_cap"]], c["q_cap"], to_rows(c["cand"]), device)

        def val_fn(m):
            return ev.rank_metrics(ev.rerank_ranks(cv, gs_for(cv, m)))["MRR"] / 100

        info = {}
        if cfg.get("mode", "graph") == "graph":
            info = train_qa(model, qd, x_img, tr_q, tr_cid, tr_rows, val_fn, tcfg, device,
                            log=lambda s: print(s, flush=True))
        alpha, curve = ev.tune_alpha(cv, gs_for(cv), step=float(cfg.get("alpha_step", 0.05)))
        alphas.append(alpha)
        infos.append({"seed": seed, "alpha": alpha, "alpha_curve": curve, **info})
        score_fn = lambda c, p, a=alpha: ev.fuse(c["sim"], gs_for(c), a)  # noqa: E731
        res = P.evaluate(score_fn, name=f"CLIP+QA s{seed}", split="test", seeds=[seed], full=True)
        per_seed.append(res["per_seed"][0])
        full_runs.append(res["full"])
        if seed == seeds[0] and cfg.get("save_model", True):
            torch.save(model.state_dict(), ctx.run_dir / "model.pt")

    agg = ev.aggregate(per_seed)
    groups = {gn: ev.aggregate([r["by_group"][gn] for r in per_seed if gn in r["by_group"]])
              for gn in sorted({k for r in per_seed for k in r.get("by_group", {})})}
    ctx.save_json("train_info.json", infos)
    ctx.save_json("protocol_result.json", {"per_seed": [{k: v for k, v in r.items() if k != "ranks"} for r in per_seed],
                                           "full": [{k: v for k, v in r.items() if k != "ranks"} for r in full_runs]})
    ctx.save_json("ranks_test_full.json", {"seed": seeds[0], "ranks": full_runs[0]["ranks"]})
    return {"metrics": {"test": agg, "test_full": ev.aggregate(full_runs), "by_group": groups,
                        "alpha": {"mean": float(np.mean(alphas)), "values": alphas}},
            "n": int(full_runs[0]["n"]), "extra": {"seeds": seeds, "graph": stats, "device": str(device)}}
