"""QA-Graph theo nhóm truy vấn (direct / indirect) — Δ so với CLIP + adapter (không cạnh), CI95 ghép cặp.

Đầu vào: hạng toàn pool test seed 0 đã commit trong results/ranks/<ds>_*.seed0.test_full.json
         và nhãn nhóm của từng truy vấn từ giao thức chung trên data/<ds>_bundle.
Dùng:    python tools/groups_vs_adapter.py --ds psg   → results/psg_groups.{md,json}   (Bảng VI)
         python tools/groups_vs_adapter.py --ds vg    → results/vg_groups.{md,json}    (Hình 5)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--ds", choices=["vg", "psg"], default="psg")
a = ap.parse_args()
a.bundle = str(ROOT / "data" / f"{a.ds}_bundle")
a.out = str(ROOT / "results" / f"{a.ds}_groups")
sys.path.insert(0, a.bundle)
from mmxai_eval import Protocol, paired_bootstrap  # noqa: E402

group = np.asarray(Protocol.load(a.bundle).candidates("test", None)["group"])
n = len(group)
masks = {"all": np.ones(n, bool), "direct": group == "direct", "indirect": group == "indirect"}

REF = (f"{a.ds}_qa_ablation_no_edge", "CLIP + adapter (không cạnh)")
ROWS = {"psg": [("psg_qa_ablation_no_rel", "chỉ contains"),
                ("psg_qa", "contains + sg_rel"),
                ("psg_qa_cn", "contains + sg_rel + ConceptNet"),
                ("psg_qa_cn_shuffle", "contains + sg_rel + ConceptNet, xáo cạnh")],
        # VG: cấu hình "Đồ thị" của Hình 5 là contains + ConceptNet (SAGE); xáo cạnh như Hình 5
        "vg": [("vg_qa_ablation_no_rel", "chỉ contains"),
               ("vg_qa", "contains + sg_rel"),
               ("vg_qa_cn_no_sg_sage", "contains + ConceptNet (SAGE)"),
               ("vg_qa_cn_shuffle", "contains + ConceptNet, xáo cạnh")]}[a.ds]
FULL = {"psg": "psg_qa_cn", "vg": "vg_qa_cn_no_sg_sage"}[a.ds]


def ranks(exp):
    r = np.asarray(json.load(open(ROOT / "results" / "ranks" / f"{exp}.seed0.test_full.json"))["ranks"], float)
    assert len(r) == n, f"{exp}: {len(r)} hạng, giao thức có {n} truy vấn"
    return r


def mrr(r):
    return float((100 / r).mean())


ref = ranks(REF[0])
out = {"n": {g: int(m.sum()) for g, m in masks.items()},
       REF[1]: {"MRR": {g: mrr(ref[m]) for g, m in masks.items()}}}
for exp, name in ROWS:
    r = ranks(exp)
    out[name] = {"MRR": {g: mrr(r[m]) for g, m in masks.items()},
                 "vs_adapter": {g: paired_bootstrap(ref[m], r[m])["MRR"] for g, m in masks.items()}}

# Cạnh quan hệ có thêm gì so với chỉ contains? (cấu hình − chỉ contains, ghép cặp)
only_c, full = ranks(f"{a.ds}_qa_ablation_no_rel"), ranks(FULL)
out["relations_vs_contains_only"] = {g: paired_bootstrap(only_c[m], full[m])["MRR"] for g, m in masks.items()}
# So cùng GNN (GAT) để không lẫn hiệu ứng kiến trúc: mọi cấu hình GAT có cạnh nhãn–nhãn − chỉ contains (GAT)
SAME_GNN = {"psg": [("psg_qa", "contains + sg_rel"), ("psg_qa_cn_no_sg", "contains + ConceptNet"),
                    ("psg_qa_cn", "contains + sg_rel + ConceptNet")],
            "vg": [("vg_qa", "contains + sg_rel"), ("vg_qa_cn_no_sg", "contains + ConceptNet"),
                   ("vg_qa_cn", "contains + sg_rel + ConceptNet")]}[a.ds]
out["same_gnn_vs_contains_only"] = {}
for exp, name in SAME_GNN:
    if (ROOT / "results" / "ranks" / f"{exp}.seed0.test_full.json").exists():
        r = ranks(exp)
        out["same_gnn_vs_contains_only"][name] = {g: paired_bootstrap(only_c[m], r[m])["MRR"] for g, m in masks.items()}

N = out["n"]
lines = [f"# QA-Graph trên {a.ds.upper()} theo nhóm truy vấn — toàn pool test, seed 0, MRR %", "",
         f"{N['all']} truy vấn ({N['direct']} direct, {N['indirect']} indirect). "
         "Δ = (hàng) − CLIP + adapter (không cạnh), CI95 bootstrap ghép cặp theo truy vấn (2000 lần, `paired_bootstrap`).", "",
         "| Cấu hình | MRR direct | Δ direct [CI95] | MRR indirect | Δ indirect [CI95] |", "|---|---|---|---|---|"]
m0 = out[REF[1]]["MRR"]
lines.append(f"| {REF[1]} | {m0['direct']:.2f} | – | {m0['indirect']:.2f} | – |")
for _, name in ROWS:
    m, d = out[name]["MRR"], out[name]["vs_adapter"]
    c = lambda g: f"{d[g]['delta']:+.2f} [{d[g]['ci95'][0]:+.2f}, {d[g]['ci95'][1]:+.2f}]"
    lines.append(f"| {name} | {m['direct']:.2f} | {c('direct')} | {m['indirect']:.2f} | {c('indirect')} |")
d = out["relations_vs_contains_only"]
c = lambda g: f"{d[g]['delta']:+.2f} [{d[g]['ci95'][0]:+.2f}, {d[g]['ci95'][1]:+.2f}]"
lines += ["", f"Cạnh quan hệ ({ROWS[2][1]} − chỉ contains): tổng {c('all')}, "
          f"direct {c('direct')}, indirect {c('indirect')}."]
lines += ["", "Cùng GNN (GAT), Δ = (cấu hình) − chỉ contains:", "",
          "| Cấu hình (GAT) | Δ tổng [CI95] | Δ direct [CI95] | Δ indirect [CI95] |", "|---|---|---|---|"]
for name, d in out["same_gnn_vs_contains_only"].items():
    c = lambda g: f"{d[g]['delta']:+.2f} [{d[g]['ci95'][0]:+.2f}, {d[g]['ci95'][1]:+.2f}]"
    lines.append(f"| {name} | {c('all')} | {c('direct')} | {c('indirect')} |")
Path(a.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path(a.out + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("\n".join(lines))
