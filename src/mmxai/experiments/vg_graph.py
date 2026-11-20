"""CLIP + Graph trên dữ liệu THẬT (bundle VG), đánh giá bằng mmxai_eval.Protocol.

Giao thức chung (mmxai_eval): seed s = gallery s (1000 ảnh lấy mẫu từ pool test/val) VÀ khởi tạo GNN seed s;
`full` = toàn bộ pool (tất định với một mô hình). Điểm cuối = fuse(cos, graph_score, α) = α·z(cos) + (1−α)·z(s_G),
α chọn bằng tune_alpha trên VAL (gallery cùng seed). Test chỉ để báo cáo.

cfg.mode:
  clip      chỉ CLIP (điểm = cos) — để kiểm tra khớp baseline (B2)
  untrained GNN khởi tạo đơn vị, không học — kiểm tra đường dữ liệu qua GNN (B3)
  graph     huấn luyện GNN như clip_graph (mặc định)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from ..core.env import set_seed
from ..data.bundle import load_bundle
from ..graph.build import build_graph
from ..graph.gnn import GraphEncoder, graph_scores, train


def load_eval_module(bundle: Path | None = None):
    """Bộ đánh giá chung của giao thức đánh giá (src/mmxai_eval.py — MỘT bản cho mọi phương pháp, mọi bộ dữ liệu).

    Bundle cũ có kèm bản chép `mmxai_eval.py`; nếu phần code (bỏ docstring) khác bản trong repo thì cảnh báo để không âm thầm
    chấm theo hai cách khác nhau. Không có bản trong repo (vd môi trường lạ) thì mới dùng bản chép trong bundle.
    """
    try:
        import mmxai_eval as mod
    except ImportError:
        if bundle is None:
            raise
        spec = importlib.util.spec_from_file_location("mmxai_eval", Path(bundle) / "mmxai_eval.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mmxai_eval"] = mod
        spec.loader.exec_module(mod)
        return mod
    copy = Path(bundle) / "mmxai_eval.py" if bundle is not None else None
    if copy is not None and copy.exists():
        def code(t):  # bỏ docstring đầu file
            return t.split('"""', 2)[2] if t.lstrip().startswith('"""') else t
        if code(copy.read_text(encoding="utf-8")) != code(Path(mod.__file__).read_text(encoding="utf-8")):
            print(f"[mmxai] CẢNH BÁO: {copy} khác src/mmxai_eval.py — đang dùng bản trong repo", flush=True)
    return mod


def _strip(res):
    """Bỏ danh sách ranks dài khỏi kết quả Protocol trước khi ghi metrics.json."""
    out = {k: v for k, v in res.items() if k not in ("per_seed", "full")}
    out["full"] = {k: v for k, v in (res.get("full") or {}).items() if k not in ("ranks",)}
    out["per_seed"] = [{k: v for k, v in r.items() if k != "ranks"} for r in res["per_seed"]]
    return out


def run(ctx) -> dict:
    cfg = ctx.cfg
    bundle = Path(cfg["data"])
    ev = load_eval_module(bundle)
    P = ev.Protocol.load(bundle)
    mode = cfg.get("mode", "graph")
    seeds = [ctx.seed] if ctx.args.get("seed") is not None else list(cfg.get("seeds", [0, 1, 2]))
    if ctx.limit:
        seeds = seeds[:1]

    if mode == "clip":
        res = P.evaluate(lambda c, p: c["sim"], name="CLIP", split="test", seeds=seeds)
        ctx.save_json("protocol_result.json", _strip(res))
        return {"metrics": {"test": res["mean_std"], "test_full": {k: res["full"][k] for k in ev.METRICS},
                            "by_group": res.get("by_group")}, "n": res["full"]["n"], "extra": {"mode": mode}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    gcfg, mcfg, tcfg = cfg.get("graph", {}), dict(cfg.get("gnn", {})), dict(cfg.get("train", {}))
    if ctx.limit:
        tcfg["epochs"] = min(int(tcfg.get("epochs", 20)), 2)
    b = load_bundle(bundle, cfg.get("extra"))
    row_of = {iid: r for r, iid in enumerate(b.image_ids)}
    to_rows = np.vectorize(lambda i: row_of[int(i)])
    tr_qi = b.captions_of("train", ctx.limit * 200 if ctx.limit else None)
    tr_q, tr_rows = b.caption_emb[tr_qi], np.array([row_of[b.captions[i]["image_id"]] for i in tr_qi])

    per_seed, full_runs, alphas, infos, graph_stats = [], [], [], [], None
    for seed in seeds:
        set_seed(seed)
        g = build_graph(b, gcfg.get("variant", "full"), int(gcfg.get("min_rel_count", 1)), seed=seed,
                        rel_merge=gcfg.get("rel_merge", "directed"))
        graph_stats = graph_stats or g.stats
        print(f"  seed {seed}: đồ thị {g.stats}")
        model = GraphEncoder(b.image_emb.shape[1], **mcfg)
        cv = P.candidates("val", seed)

        def gs_for(c, m=model):
            return graph_scores(m, g.data, P.cap[c["q_cap"]], to_rows(c["cand"]), device)

        def val_fn(m):
            return ev.rank_metrics(ev.rerank_ranks(cv, gs_for(cv, m)))["MRR"] / 100

        info = {}
        if mode == "graph":
            info = train(model, g.data, g.n_img, tr_q, tr_rows, val_fn, tcfg, device)
        model.to(device)
        alpha, curve = ev.tune_alpha(cv, gs_for(cv), step=float(cfg.get("alpha_step", 0.05)))
        alphas.append(alpha)
        infos.append({"seed": seed, "alpha": alpha, "alpha_curve": curve, **info})
        score_fn = lambda c, p, a=alpha: ev.fuse(c["sim"], gs_for(c), a)  # noqa: E731
        res = P.evaluate(score_fn, name=f"CLIP+Graph s{seed}", split="test", seeds=[seed], full=True)
        per_seed.append(res["per_seed"][0])
        full_runs.append(res["full"])
        if seed == seeds[0] and cfg.get("save_model", True) and mode == "graph":
            torch.save(model.state_dict(), ctx.run_dir / "model.pt")

    agg = ev.aggregate(per_seed)
    groups = {}
    for gname in sorted({k for r in per_seed for k in r.get("by_group", {})}):
        groups[gname] = ev.aggregate([r["by_group"][gname] for r in per_seed if gname in r["by_group"]])
    full_agg = ev.aggregate(full_runs)
    ctx.save_json("train_info.json", infos)
    ctx.save_json("protocol_result.json", {"per_seed": [{k: v for k, v in r.items() if k != "ranks"} for r in per_seed],
                                           "full": [{k: v for k, v in r.items() if k != "ranks"} for r in full_runs]})
    ctx.save_json("ranks_test_full.json", {"seed": seeds[0], "ranks": full_runs[0]["ranks"]})
    return {"metrics": {"test": agg, "test_full": full_agg, "by_group": groups,
                        "alpha": {"mean": float(np.mean(alphas)), "values": alphas}},
            "n": int(full_runs[0]["n"]),
            "extra": {"mode": mode, "seeds": seeds, "graph": graph_stats, "device": str(device)}}
