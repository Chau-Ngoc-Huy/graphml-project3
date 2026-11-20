"""So số trong paper với lượt chạy của mình: docs/paper_numbers.yaml + results/<exp>/*.json → results/REPRODUCTION.md (+ .json).

Dùng: python tools/compare.py [--table table2] [--tolerance 1.0] [--include-smoke]
Mỗi dòng: lấy lượt đầy đủ mới nhất (không --limit, không lỗi) của `exp`, hoặc `run_id` ghim sẵn. Xem docs/paper_numbers.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from make_kernel import PKG  # noqa: E402

results = __import__(f"{PKG}.core.results", fromlist=["load_results", "flatten"])


def pick_run(exp: str, run_id: str | None, tag: str | None, include_smoke: bool) -> dict | None:
    runs = results.load_results(exp)
    if run_id:
        hits = [m for m in runs if m["run"]["run_id"] == run_id]
        return hits[0] if hits else None
    runs = [m for m in runs if not m["run"].get("error") and (include_smoke or not m["run"]["args"].get("limit"))]
    if tag:
        runs = [m for m in runs if m["run"].get("tag") == tag]
    runs.sort(key=lambda m: m["run"].get("finished_at", ""))
    return runs[-1] if runs else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=str(ROOT / "docs" / "paper_numbers.yaml"))
    p.add_argument("--table", help="chỉ một bảng (id)")
    p.add_argument("--tolerance", type=float, help="ghi đè sai số cho phép (tuyệt đối, đơn vị của paper)")
    p.add_argument("--include-smoke", action="store_true")
    p.add_argument("--out", default=str(ROOT / "results" / "REPRODUCTION.md"))
    args = p.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8")) or {}
    default_tol = args.tolerance if args.tolerance is not None else float(spec.get("tolerance", 1.0))
    lines = [f"# Tái lập: {spec.get('paper', {}).get('title', '')}", "",
             f"Sinh tự động bởi `make compare` lúc {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC từ `docs/paper_numbers.yaml` + `results/`. "
             "Nhận xét ghi ở `results/JOURNAL.md`.", "",
             "Trạng thái: ✓ chênh ≤ sai số cho phép · ✗ chênh lớn hơn · – chưa có lượt đầy đủ. Δ = mình − paper.", ""]
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tables": []}
    n_ok = n_bad = n_missing = 0
    for table in spec.get("tables", []):
        if args.table and table["id"] != args.table:
            continue
        metrics_def: dict = table.get("metrics", {})
        tol_table = float(table.get("tolerance", default_tol))
        head = ["Dòng", *(f"{k} paper" for k in metrics_def), *(f"{k} mình" for k in metrics_def), *(f"Δ {k}" for k in metrics_def), "Trạng thái", "run_id"]
        lines += [f"## {table['id']} — {table.get('caption', '')}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        trows = []
        for row in table.get("rows", []):
            m = pick_run(row["exp"], row.get("run_id"), row.get("tag"), args.include_smoke) if row.get("exp") else None
            flat = results.flatten(m.get("metrics", {})) if m else {}
            paper_vals, ours_vals, deltas, status, ours_scaled, fmts = [], [], [], [], {}, {}
            for k, mdef in metrics_def.items():
                mdef = mdef if isinstance(mdef, dict) else {"ours": mdef}
                fmt_ = mdef.get("fmt", "{:.2f}")
                tol = float(mdef.get("tolerance", tol_table))
                pv = (row.get("paper") or {}).get(k)
                ov = flat.get(mdef.get("ours", k))
                if ov is not None and isinstance(ov, (int, float)):
                    ov = ov * float(mdef.get("paper_scale", 1))
                ours_scaled[k], fmts[k] = ov, fmt_
                paper_vals.append(fmt_.format(pv) if isinstance(pv, (int, float)) else ("–" if pv is None else str(pv)))
                ours_vals.append(fmt_.format(ov) if isinstance(ov, (int, float)) else "–")
                if isinstance(pv, (int, float)) and isinstance(ov, (int, float)):
                    d = ov - pv
                    deltas.append(("+" if d >= 0 else "") + fmt_.format(d))
                    status.append(abs(d) <= tol)
                else:
                    deltas.append("–")
            if m is None:
                st = "–"
                n_missing += 1
            elif status and all(status):
                st = "✓"
                n_ok += 1
            elif status:
                st = "✗"
                n_bad += 1
            else:
                st = "?"
            lines.append("| " + " | ".join([row["name"], *paper_vals, *ours_vals, *deltas, st, m["run"]["run_id"] if m else "–"]) + " |")
            trows.append({"name": row["name"], "exp": row.get("exp"), "run_id": m["run"]["run_id"] if m else None,
                          "paper": row.get("paper"), "ours": ours_scaled, "fmt": fmts, "status": st})
        if table.get("note"):
            lines += ["", f"_{table['note']}_"]
        lines.append("")
        report["tables"].append({"id": table["id"], "caption": table.get("caption"), "metrics": list(metrics_def), "rows": trows})
    lines.insert(5, f"**Tổng:** {n_ok} ✓ · {n_bad} ✗ · {n_missing} chưa chạy (sai số mặc định ±{default_tol}).")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {out.relative_to(ROOT)}: {n_ok} ✓ · {n_bad} ✗ · {n_missing} chưa chạy")


if __name__ == "__main__":
    main()
