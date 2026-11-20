"""Kiểm định ghép cặp (bootstrap theo truy vấn) giữa AdaMF-MAT của Khánh và các cấu hình của Huy, toàn pool test (2138 ảnh, 10 696 caption).

Hai bên phải cùng thứ tự truy vấn: script kiểm tra hạng CLIP của Khánh trùng `gt_rank` do Protocol của Thắng sinh ra, và độ dài khớp.
Dùng:  python tools/paired_vs_qagraph.py --bundle <thư mục bundle của Thắng>   (bundle chỉ cần để lấy nhóm direct/indirect + kiểm tra thứ tự)
Ra: results/vs_qagraph_paired.md  (+ .json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT  # repo thống nhất: tools/ nằm ngay dưới gốc
sys.path.insert(0, str(REPO / "src"))
from mmxai_eval import Protocol, paired_bootstrap, rank_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--mine", default=str(REPO / "results" / "adamf" / "protocol_coco_full.json"))
ap.add_argument("--huy", default=str(REPO / "results" / "ranks"))
ap.add_argument("--out", default=str(REPO / "results" / "adamf" / "vs_qagraph_paired"))
a = ap.parse_args()

mine = {k: np.array(v["full"]["ranks"]) for k, v in json.load(open(a.mine)).items()}
huy = {p.name.split(".seed")[0]: np.array(json.load(open(p))["ranks"]) for p in sorted(Path(a.huy).glob("*.json"))}

P = Protocol.load(a.bundle)
c = P.candidates("test", None)
group = np.asarray(c["group"])
n = len(group)
assert all(len(v) == n for v in mine.values()), "độ dài xếp hạng của Khánh không khớp"
assert all(len(v) == n for v in huy.values()), "độ dài xếp hạng của Huy không khớp"
assert np.array_equal(mine["CLIP"], c["gt_rank"]), "thứ tự truy vấn của Khánh KHÔNG khớp giao thức của Thắng"
print(f"{n} truy vấn; thứ tự khớp giao thức của Thắng (hạng CLIP trùng gt_rank)")

ref = mine["CLIP+AdaMF-MAT"]
rows, out = [], {}
masks = {"all": np.ones(n, bool), "direct": group == "direct", "indirect": group == "indirect"}


def mrr(r):
    return float((100 / r).mean())


for name, r in [("CLIP (Thắng)", mine["CLIP"]), ("CLIP+khớp tên (Thắng/Khánh)", mine["CLIP+presence (no relations)"])] + list(huy.items()):
    rec = {"MRR": {g: mrr(r[m]) for g, m in masks.items()}}
    rec["vs_AdaMF"] = {g: paired_bootstrap(ref[m], r[m])["MRR"] for g, m in masks.items()}  # delta = (phương pháp này) - AdaMF-MAT
    out[name] = rec
    rows.append((name, rec))
out["CLIP+AdaMF-MAT (Khánh)"] = {"MRR": {g: mrr(ref[m]) for g, m in masks.items()}}

lines = ["# AdaMF-MAT (Khánh) vs các cấu hình của Huy: kiểm định ghép cặp, toàn pool test (seed 0), MRR %", "",
         f"{n} truy vấn, cùng thứ tự với giao thức của Thắng. Δ = (phương pháp trong hàng) − AdaMF-MAT; CI95 bootstrap theo truy vấn; "
         "Δ > 0 nghĩa là phương pháp trong hàng tốt hơn AdaMF-MAT.", "",
         "| Phương pháp | MRR all | MRR direct | MRR indirect | Δ all [CI95] | Δ direct [CI95] | Δ indirect [CI95] |", "|---|---|---|---|---|---|---|"]
a_m = out["CLIP+AdaMF-MAT (Khánh)"]["MRR"]
lines.append(f"| **CLIP+AdaMF-MAT (Khánh)** | {a_m['all']:.2f} | {a_m['direct']:.2f} | {a_m['indirect']:.2f} | | | |")
for name, rec in rows:
    d = rec["vs_AdaMF"]
    cell = lambda g: f"{d[g]['delta']:+.2f} [{d[g]['ci95'][0]:+.2f}, {d[g]['ci95'][1]:+.2f}]"
    lines.append(f"| {name} | {rec['MRR']['all']:.2f} | {rec['MRR']['direct']:.2f} | {rec['MRR']['indirect']:.2f} | {cell('all')} | {cell('direct')} | {cell('indirect')} |")
Path(a.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path(a.out + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("\n".join(lines))
