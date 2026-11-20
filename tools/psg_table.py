"""Bảng kết quả PSG (máy local, tools/run_psg_local.sh) + so sánh cạnh nhau với VG (results/tables/t1_official.md).

Mỗi biến thể có 2 lượt: tag s0 (seed 0, lưu model) và s12 (seed 1, 2) → gộp thành 3 seed:
  - gallery 1000: mean ± std trên 3 seed (seed = gallery + khởi tạo)
  - toàn pool test (2186 ảnh): mean ± std 3 mô hình; MRR theo nhóm direct/indirect
  - ΔMRR vs Adapter: seed 0, toàn pool, bootstrap 2000 lần theo truy vấn [CI95]
Biến thể chưa đủ 3 seed vẫn hiện (ghi số seed).

Dùng: python tools/psg_table.py   → results/tables/psg_official.md
"""

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROWS = [  # (bảng, exp, tên, exp VG tương ứng trong t1_official)
    ("main", "psg_clip", "CLIP (đóng băng)", None),
    ("main", "psg_qa_ablation_no_edge", "CLIP + Adapter (không đồ thị)", "CLIP + Adapter (không đồ thị)"),
    ("main", "psg_qa", "+ QA-Graph (contains + sg_rel)", "+ QA-Graph (contains + sg_rel)"),
    ("main", "psg_qa_ablation_no_rel", "+ QA-Graph (chỉ contains)", "+ QA-Graph (chỉ contains)"),
    ("main", "psg_qa_cn", "+ QA-Graph + ConceptNet (contains + sg_rel + CN)", "+ QA-Graph + ConceptNet (contains + sg_rel + CN)"),
    ("main", "psg_qa_cn_no_sg", "+ QA-Graph + ConceptNet, bỏ sg_rel (contains + CN)",
     "+ QA-Graph + ConceptNet, bỏ sg_rel (contains + CN)"),
    ("ablation", "psg_qa_ablation_shuffle", "QA-Graph, xáo cạnh (shuffle)", None),
    ("ablation", "psg_qa_cn_shuffle", "QA-Graph + CN, xáo cạnh (shuffle)", "QA-Graph + CN, xáo cạnh (shuffle)"),
]
KEYS = ("R@1", "R@5", "R@10", "MRR", "nDCG@10")


def runs_of(exp):
    out = []
    for d in sorted(glob.glob(str(ROOT / "runs" / exp / "*"))):
        d = Path(d)
        if "smoke" in d.name or not (d / "protocol_result.json").exists():
            continue
        out.append(d)
    return out


def collect(exp):
    """per_seed (gallery) theo seed, full (toàn pool) mỗi mô hình, ranks toàn pool của seed 0."""
    per, full, r0 = {}, [], None
    for d in runs_of(exp):
        p = json.loads((d / "protocol_result.json").read_text())
        for r in p.get("per_seed", []):
            if r.get("seed") is not None:
                per[int(r["seed"])] = r
        f = p.get("full")
        full += f if isinstance(f, list) else ([f] if f else [])
        rk = d / "ranks_test_full.json"
        if rk.exists():
            j = json.loads(rk.read_text())
            if r0 is None or j.get("seed") == 0:
                r0 = np.asarray(j["ranks"], float)
    return per, full, r0


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"


def vg_pool_mrr():
    """MRR toàn pool của VG (Kaggle 3 seed) từ results/tables/t1_official.md, theo tên dòng."""
    f = ROOT / "results" / "tables" / "t1_official.md"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            c = [x.strip() for x in line.split("|")]
            if len(c) > 12 and c[1] and not c[1].startswith(("Mô hình", "---")):
                out.setdefault(c[1], (c[8], c[11]))       # MRR (pool), ΔMRR vs Adapter
    return out


data = {e: collect(e) for _, e, _, _ in ROWS}
base = data["psg_qa_ablation_no_edge"][2]
rng = np.random.default_rng(0)


def paired(exp):
    r = data[exp][2]
    if r is None or base is None or len(r) != len(base) or exp in ("psg_qa_ablation_no_edge",):
        return "—"
    d = 1 / r - 1 / base
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
    return f"{d.mean() * 100:+.2f} [{lo:+.2f}, {hi:+.2f}]"


vg = vg_pool_mrr()
stats = json.loads((ROOT / "data" / "psg_bundle" / "psg_stats.json").read_text()) \
    if (ROOT / "data" / "psg_bundle" / "psg_stats.json").exists() else {}
out = ["# Kết quả PSG (Panoptic Scene Graph ∩ caption COCO) — máy local, GAT, 3 seed\n",
       f"Dữ liệu: {stats.get('splits')}, {stats.get('image_labels')} nhãn ảnh + {stats.get('concepts_ext')} khái niệm "
       f"mở rộng (ConceptNet ∩ caption train); nhóm truy vấn test {stats.get('test_groups')}. "
       "Cạnh sg_rel gộp KHÔNG hướng (graph.rel_merge=undirected).",
       "Gallery 1000: mean ± std theo seed (seed = gallery + khởi tạo). Toàn pool: 2186 ảnh test, mean ± std các mô hình.",
       "ΔMRR vs Adapter: seed 0, toàn pool, bootstrap 2000 lần theo truy vấn [CI95]. Cột VG: Kaggle 3 seed (t1_official.md).\n"]
for table, title in (("main", "Bảng chính"), ("ablation", "Đối chứng (xáo cạnh)")):
    out += [f"## {title}\n",
            "| Mô hình | seed | R@1 (1K) | R@5 (1K) | MRR (1K) | R@1 (pool) | MRR (pool) | MRR direct | MRR indirect | ΔMRR vs Adapter | VG: MRR (pool) | VG: ΔMRR |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, exp, name, vgname in ROWS:
        if t != table:
            continue
        per, full, _ = data[exp]
        if not per and not full:
            out.append(f"| {name} | 0 | (chưa chạy `{exp}`) |" + " |" * 9)
            continue
        ps = [per[s] for s in sorted(per)]
        g = lambda k, grp=None: [((r.get("by_group") or {}).get(grp) or {}).get(k) if grp else r.get(k) for r in ps]
        v = vg.get(vgname, ("—", "—")) if vgname else ("60.40" if exp == "psg_clip" else "—", "—")
        out.append(f"| {name} | {','.join(map(str, sorted(per)))} | {ms(g('R@1'))} | {ms(g('R@5'))} | {ms(g('MRR'))} | "
                   f"{ms([f['R@1'] for f in full])} | {ms([f['MRR'] for f in full])} | {ms(g('MRR', 'direct'))} | "
                   f"{ms(g('MRR', 'indirect'))} | {paired(exp)} | {v[0]} | {v[1]} |")
    out.append("")
dst = ROOT / "results" / "tables" / "psg_official.md"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"→ {dst}")
