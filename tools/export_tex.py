"""Xuất bảng LaTeX (tabularx, đúng kiểu skill latex-report) từ results/REPRODUCTION.json hoặc từ các lượt chạy.

Dùng: python tools/export_tex.py --table table2 [--out report/tables/tab_table2.tex]
      python tools/export_tex.py --exps exp_a,exp_b --metrics accuracy,f1 --caption "..." --out tables/tab_ext.tex
Số lấy từ file kết quả, không gõ tay (tránh lệch nguồn). Sau đó \\input{tables/tab_<id>} trong báo cáo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from make_kernel import PKG  # noqa: E402

results = __import__(f"{PKG}.core.results", fromlist=["load_results", "flatten"])


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def table_tex(label: str, caption: str, head: list[str], rows: list[list[str]], spec: str | None = None) -> str:
    spec = spec or "l" + " c" * (len(head) - 1)
    out = ["% Sinh bởi tools/export_tex.py — đừng sửa tay, chạy lại `make tex` khi có lượt mới.",
           r"\begin{table}[H]", r"\centering", f"\\caption{{{esc(caption).rstrip('.')}.}}", f"\\label{{tab:{label}}}", r"\small",
           f"\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}", r"\toprule",
           " & ".join(f"\\textbf{{{esc(h)}}}" for h in head) + r"\\", r"\midrule"]
    out += [" & ".join(esc(c) for c in r) + r"\\" for r in rows]
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(out)


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else ("–" if v is None else str(v))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", help="id bảng trong results/REPRODUCTION.json (chạy make compare trước)")
    p.add_argument("--exps", help="danh sách thí nghiệm, cách nhau bằng dấu phẩy (lượt đầy đủ mới nhất của mỗi cái)")
    p.add_argument("--metrics", help="khóa metric (dạng phẳng a.b), cách nhau bằng dấu phẩy")
    p.add_argument("--caption", default="")
    p.add_argument("--label")
    p.add_argument("--nd", type=int, default=2)
    p.add_argument("--out")
    args = p.parse_args()

    if args.table:
        rep = json.loads((ROOT / "results" / "REPRODUCTION.json").read_text(encoding="utf-8"))
        t = next((t for t in rep["tables"] if t["id"] == args.table), None) or sys.exit(f"không có bảng {args.table}")
        head = ["Cấu hình"] + [f"{k} (paper)" for k in t["metrics"]] + [f"{k} (tái lập)" for k in t["metrics"]]
        def f(r, k, v):
            return (r.get("fmt") or {}).get(k, "{:.2f}").format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else ("–" if v is None else str(v))
        rows = [[r["name"], *[f(r, k, (r["paper"] or {}).get(k)) for k in t["metrics"]],
                 *[f(r, k, (r["ours"] or {}).get(k)) for k in t["metrics"]]] for r in t["rows"]]
        label, caption = args.label or args.table, args.caption or (t.get("caption") or args.table)
        spec = "X" + " c" * (len(head) - 1)
    else:
        exps = [e for e in (args.exps or "").split(",") if e] or sys.exit("cần --table hoặc --exps")
        keys = [k for k in (args.metrics or "").split(",") if k]
        rows = []
        for e in exps:
            runs = [m for m in results.load_results(e) if not m["run"]["args"].get("limit") and not m["run"].get("error")]
            m = max(runs, key=lambda m: m["run"].get("finished_at", ""), default=None)
            flat = results.flatten(m["metrics"]) if m else {}
            keys = keys or list(flat)
            rows.append([e, *[fmt(flat.get(k), args.nd) for k in keys]])
        head, label, caption, spec = ["Thí nghiệm", *keys], args.label or "ext", args.caption or "Kết quả mở rộng", "X" + " c" * len(keys)
    tex = table_tex(label, caption, head, rows, spec)
    out = Path(args.out) if args.out else ROOT / "results" / "tex" / f"tab_{label}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
