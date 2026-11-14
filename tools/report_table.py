"""Gom lượt ĐẦY ĐỦ mới nhất của mỗi thí nghiệm trong results/ thành bảng (Markdown + LaTeX) cho báo cáo.

Bảng main:     CLIP vs CLIP+Graph (GCN/GAT/SAGE), mean ± std, tách all / direct / indirect.
Bảng ablation: full vs no_rel / no_concept / no_edge / shuffle / random.
Dùng: python tools/report_table.py [--metrics R@1 R@5 R@10 MRR] [--group all] [--out results/tables]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = [("CLIP (đóng băng)", "clip_graph", "clip"), ("CLIP+Graph (GCN)", "clip_graph_gcn", "clip_graph"),
        ("CLIP+Graph (GAT)", "clip_graph", "clip_graph"), ("CLIP+Graph (SAGE)", "clip_graph_sage", "clip_graph")]
ABL = [("Đồ thị đầy đủ", "clip_graph"), ("Bỏ cạnh quan hệ", "ablation_no_rel"), ("Bỏ cạnh ConceptNet", "ablation_no_concept"),
       ("Không cạnh (chỉ lớp chiếu)", "ablation_no_edge"), ("Xáo trộn cạnh", "ablation_shuffle"), ("Đồ thị ngẫu nhiên", "ablation_random")]


def latest(exp):
    files = sorted((ROOT / "results" / exp).glob("*.json"))
    runs = [json.loads(f.read_text()) for f in files]
    runs = [r for r in runs if not r["run"]["args"].get("limit") and not r["run"].get("error")]
    return runs[-1] if runs else None


def cell(run, key, group, metric):
    if not run:
        return "—"
    m = run["metrics"].get(key, {}).get(group, {})
    if metric not in m:
        return "—"
    std = run["metrics"].get(f"{key}_std", {}).get(group, {}).get(metric)
    return f"{100 * m[metric]:.1f}" + (f" ± {100 * std:.1f}" if std is not None and key != "clip" else "")


def table(rows, groups, metrics):
    head = ["Mô hình"] + [f"{m} ({g})" for g in groups for m in metrics]
    body = [[name] + [cell(latest(exp), key, g, m) for g in groups for m in metrics] for name, exp, key in rows]
    return head, body


def write(name, head, body, out):
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(r) + " |\n" for r in body)
    tex = ("\\begin{tabular}{l" + "c" * (len(head) - 1) + "}\n\\toprule\n" + " & ".join(head).replace("@", "@").replace("_", "\\_")
           + " \\\\\n\\midrule\n" + "".join(" & ".join(r).replace("±", "$\\pm$") + " \\\\\n" for r in body) + "\\bottomrule\n\\end{tabular}\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"tab_{name}.md").write_text(md, encoding="utf-8")
    (out / f"tab_{name}.tex").write_text(tex, encoding="utf-8")
    print(f"## {name}\n{md}")


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--metrics", nargs="+", default=["R@1", "R@5", "R@10", "MRR"])
p.add_argument("--groups", nargs="+", default=["all", "direct", "indirect"])
p.add_argument("--out", default=str(ROOT / "results" / "tables"))
a = p.parse_args()
write("main", *table(MAIN, ["all"], a.metrics), Path(a.out))
write("groups", *table(MAIN, a.groups, ["R@10", "MRR"]), Path(a.out))
write("ablation", *table([(n, e, "clip_graph") for n, e in ABL], a.groups, ["MRR"]), Path(a.out))
