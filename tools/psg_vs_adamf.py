"""AdaMF-MAT (phương pháp cấp độ 2 của Khánh) trên PSG vs các cấu hình QA-Graph của Huy — kiểm định ghép cặp, toàn pool test.

Đầu vào:
  - kết quả Kaggle `huychau/mmxai-adamf-psg` (notebook của Khánh, chỉ đổi input = PSG bundle, bật ablation):
    results_PSG_{full,shuf,drop}.json (có hạng từng truy vấn ở ["full"]["ranks"])
  - hạng toàn pool seed 0 của Huy: runs/psg_*/*-s0*/ranks_test_full.json
Kiểm tra thứ tự truy vấn: hạng CLIP của Khánh phải trùng gt_rank của Protocol trên PSG bundle.

Dùng: python tools/psg_vs_adamf.py --adamf runs/adamf_psg/kaggle-v1   → results/psg_vs_adamf.{md,json}
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--adamf", required=True)
ap.add_argument("--bundle", default=str(ROOT / "data" / "psg_bundle"))
ap.add_argument("--out", default=str(ROOT / "results" / "psg_vs_adamf"))
a = ap.parse_args()
sys.path.insert(0, a.bundle)
from mmxai_eval import Protocol, paired_bootstrap  # noqa: E402

AD = {v: json.load(open(Path(a.adamf) / f"results_PSG_{v}.json")) for v in ("full", "shuf", "drop")}
P = Protocol.load(a.bundle)
c = P.candidates("test", None)
group = np.asarray(c["group"])
n = len(group)
clip = np.array(AD["full"]["CLIP"]["full"]["ranks"])
gt = np.asarray(c["gt_rank"])
# GPU (Kaggle) và CPU tính cosine lệch ở chữ số cuối → vài ảnh hoà điểm đổi chỗ (lệch đúng 1 hạng); thứ tự truy vấn vẫn phải khớp
assert len(clip) == n and np.abs(clip - gt).max() <= 1 and (clip != gt).mean() < 0.02, \
    "thứ tự truy vấn của kết quả AdaMF KHÔNG khớp giao thức PSG"
print(f"{n} truy vấn khớp giao thức; {(clip != gt).sum()} hạng CLIP lệch 1 bậc do hoà điểm GPU/CPU")
masks = {"all": np.ones(n, bool), "direct": group == "direct", "indirect": group == "indirect"}


def mrr(r):
    return float((100 / r).mean())


def ranks_of(exp):
    fs = sorted(glob.glob(str(ROOT / "runs" / exp / f"*{exp}-s0*" / "ranks_test_full.json")))
    return np.asarray(json.load(open(fs[0]))["ranks"], float) if fs else None


meth = {
    "CLIP": clip,
    "CLIP + khớp tên (không quan hệ)": AD["full"]["CLIP+presence (no relations)"]["full"]["ranks"],
    "CLIP + AdaMF-MAT": AD["full"]["CLIP+AdaMF-MAT"]["full"]["ranks"],
    "CLIP + AdaMF-MAT, xáo cạnh": AD["shuf"]["CLIP+AdaMF-MAT"]["full"]["ranks"],
    "CLIP + AdaMF-MAT, bỏ quan hệ": AD["drop"]["CLIP+AdaMF-MAT"]["full"]["ranks"],
    "CLIP + Adapter (Khánh)": AD["full"]["CLIP+Adapter"]["full"]["ranks"],
    "CLIP + Adapter + khớp tên": AD["full"]["CLIP+Adapter+presence (no relations)"]["full"]["ranks"],
    "CLIP + Adapter + AdaMF-MAT": AD["full"]["CLIP+Adapter+AdaMF-MAT"]["full"]["ranks"],
    "CLIP + Adapter + AdaMF-MAT, xáo cạnh": AD["shuf"]["CLIP+Adapter+AdaMF-MAT"]["full"]["ranks"],
    "CLIP + Adapter + AdaMF-MAT, bỏ quan hệ": AD["drop"]["CLIP+Adapter+AdaMF-MAT"]["full"]["ranks"],
}
for exp, name in [("psg_qa_ablation_no_edge", "Huy: CLIP + Adapter (không đồ thị)"), ("psg_qa", "Huy: QA-Graph (contains + sg_rel)"),
                  ("psg_qa_ablation_no_rel", "Huy: QA-Graph (chỉ contains)"), ("psg_qa_cn", "Huy: QA-Graph + ConceptNet"),
                  ("psg_qa_ablation_shuffle", "Huy: QA-Graph, xáo cạnh")]:
    r = ranks_of(exp)
    if r is not None:
        assert len(r) == n, exp
        meth[name] = r
meth = {k: np.asarray(v, float) for k, v in meth.items()}
ref_name = "CLIP + Adapter + AdaMF-MAT"
ref = meth[ref_name]

out, lines = {}, [
    "# AdaMF-MAT (Khánh) trên PSG vs QA-Graph (Huy) — kiểm định ghép cặp, toàn pool test (2186 ảnh), MRR %", "",
    f"{n} truy vấn, cùng thứ tự với giao thức (hạng CLIP khớp gt_rank; 78 truy vấn lệch 1 bậc do hoà điểm GPU/CPU). AdaMF-MAT: notebook của Khánh trên Kaggle T4 "
    "(`huychau/mmxai-adamf-psg`), 300 epoch, 1 seed huấn luyện; MMKG = 133 nhãn PSG + 9186 ảnh (6000 train + val/test). "
    "Huy: seed 0. Δ = (hàng) − CLIP+Adapter+AdaMF-MAT, CI95 bootstrap theo truy vấn.", "",
    "| Phương pháp | MRR all | MRR direct | MRR indirect | Δ all [CI95] | Δ direct [CI95] | Δ indirect [CI95] |",
    "|---|---|---|---|---|---|---|"]
for name, r in meth.items():
    rec = {"MRR": {g: mrr(r[m]) for g, m in masks.items()}}
    if name != ref_name:
        rec["vs_ref"] = {g: paired_bootstrap(ref[m], r[m])["MRR"] for g, m in masks.items()}
        d = rec["vs_ref"]
        cell = lambda g: f"{d[g]['delta']:+.2f} [{d[g]['ci95'][0]:+.2f}, {d[g]['ci95'][1]:+.2f}]"
        tail = f"{cell('all')} | {cell('direct')} | {cell('indirect')}"
        nm = name
    else:
        tail, nm = " | | ", f"**{name}**"
    out[name] = rec
    m = rec["MRR"]
    lines.append(f"| {nm} | {m['all']:.2f} | {m['direct']:.2f} | {m['indirect']:.2f} | {tail} |")
Path(a.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path(a.out + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("\n".join(lines))
