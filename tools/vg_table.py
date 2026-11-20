"""Bảng kết quả chính thức (3 seed, Kaggle, tag kaggle-3seed) cho Nhiệm vụ 2 cấp 1 trên bundle của Thắng.

Đọc results/<exp>/*<tag>*.json mới nhất của mỗi thí nghiệm → results/tables/vg_official.md:
  - Bảng chính + ablation + GCN/GAT/SAGE: mean ± std 3 seed (gallery 1000) và toàn pool test (2138 ảnh).
  - MRR theo nhóm direct / indirect (định nghĩa nhóm của Thắng).
  - Kiểm định ghép cặp (bootstrap theo truy vấn, seed 0, toàn pool) so với CLIP+Adapter: ΔMRR và CI95.
Dòng CLIP lấy từ baseline của Thắng (cùng giao thức; vg_clip của mình khớp tuyệt đối, JOURNAL #2).

Dùng: python tools/vg_table.py [--tag kaggle-3seed]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--tag", default="kaggle-3seed")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()

ROWS = [  # (nhóm bảng, exp, tên hiển thị)
    ("main", "vg_qa_ablation_no_edge", "CLIP + Adapter (không đồ thị)"),
    ("main", "vg_qa", "+ QA-Graph (contains + sg_rel)"),
    ("main", "vg_qa_ablation_no_rel", "+ QA-Graph (chỉ contains)"),
    ("main", "vg_qa_cn", "+ QA-Graph + ConceptNet (contains + sg_rel + CN)"),
    ("main", "vg_qa_cn_no_sg", "+ QA-Graph + ConceptNet, bỏ sg_rel (contains + CN)"),
    ("ablation", "vg_qa_cn_shuffle", "QA-Graph + CN, xáo cạnh (shuffle)"),
    ("gnn", "vg_qa_cn_gcn", "contains + sg_rel + CN · GCN"),
    ("gnn", "vg_qa_cn", "contains + sg_rel + CN · GAT"),
    ("gnn", "vg_qa_cn_sage", "contains + sg_rel + CN · SAGE"),
    ("gnn", "vg_qa_cn_no_sg_gcn", "contains + CN · GCN"),
    ("gnn", "vg_qa_cn_no_sg", "contains + CN · GAT"),
    ("gnn", "vg_qa_cn_no_sg_sage", "contains + CN · SAGE"),
]
CLIP = {"gal": {"R@1": (57.36, 0.95), "R@5": (84.08, 0.24), "R@10": (91.76, 0.33), "MRR": (69.23, 0.68),
                "nDCG@10": (74.32, 0.55)},
        "full": {"R@1": 48.31, "MRR": 60.40}, "direct": (69.33, 0.89), "indirect": (69.24, 0.65)}


def latest(exp):
    fs = sorted(glob.glob(str(ROOT / "results" / exp / f"*{a.tag}*.json")))
    return (json.loads(Path(fs[-1]).read_text()), Path(fs[-1]).stem) if fs else (None, None)


def ranks(run_id, exp):
    f = ROOT / "runs" / exp / run_id / "ranks_test_full.json"
    return np.asarray(json.loads(f.read_text())["ranks"], float) if f.exists() else None


def ms(d):
    return f"{d['mean']:.2f} ± {d['std']:.2f}"


res = {e: latest(e) for _, e, _ in ROWS}
base_r = ranks(res["vg_qa_ablation_no_edge"][1], "vg_qa_ablation_no_edge") if res["vg_qa_ablation_no_edge"][0] else None
rng = np.random.default_rng(0)


def paired(exp):
    r = ranks(res[exp][1], exp) if res[exp][0] else None
    if r is None or base_r is None or len(r) != len(base_r) or exp == "vg_qa_ablation_no_edge":
        return "—"
    d = 1 / r - 1 / base_r
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
    return f"{d.mean() * 100:+.2f} [{lo:+.2f}, {hi:+.2f}]"


out = [f"# Kết quả chính thức Nhiệm vụ 2 cấp 1 (tag `{a.tag}`, 3 seed, Kaggle T4)\n",
       "Gallery 1000: mean ± std 3 seed (seed = gallery + khởi tạo). Toàn pool: 2138 ảnh test, mean ± std 3 mô hình.",
       "ΔMRR vs Adapter: seed 0, toàn pool, bootstrap 2000 lần theo truy vấn, [CI95]. MRR/R@k tính theo %.\n"]
for table, title in (("main", "Bảng chính"), ("ablation", "Đối chứng"), ("gnn", "So sánh GNN")):
    out += [f"## {title}\n",
            "| Mô hình | R@1 (1K) | R@5 (1K) | R@10 (1K) | MRR (1K) | nDCG@10 (1K) | R@1 (pool) | MRR (pool) | MRR direct | MRR indirect | ΔMRR vs Adapter |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    if table == "main":
        g = CLIP["gal"]
        out.append("| CLIP (baseline Thắng) | " + " | ".join(f"{g[k][0]:.2f} ± {g[k][1]:.2f}" for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10"))
                   + f" | {CLIP['full']['R@1']:.2f} | {CLIP['full']['MRR']:.2f} | {CLIP['direct'][0]:.2f} ± {CLIP['direct'][1]:.2f}"
                   f" | {CLIP['indirect'][0]:.2f} ± {CLIP['indirect'][1]:.2f} | — |")
    for t, exp, name in ROWS:
        if t != table:
            continue
        d, _ = res[exp]
        if d is None:
            out.append(f"| {name} | (chưa có `{exp}`) |" + " |" * 9)
            continue
        m = d["metrics"]
        bg = m.get("by_group", {})
        out.append(f"| {name} | " + " | ".join(ms(m["test"][k]) for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10"))
                   + f" | {ms(m['test_full']['R@1'])} | {ms(m['test_full']['MRR'])}"
                   + f" | {ms(bg['direct']['MRR']) if 'direct' in bg else '—'} | {ms(bg['indirect']['MRR']) if 'indirect' in bg else '—'}"
                   + f" | {paired(exp)} |")
    out.append("")
out.append("Nguồn: " + ", ".join(f"`{res[e][1]}`" for _, e, _ in ROWS if res[e][0]))
dst = ROOT / "results" / "tables" / "vg_official.md"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"\n→ {dst}")
