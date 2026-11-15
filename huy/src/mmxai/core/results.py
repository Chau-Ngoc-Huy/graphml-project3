"""results/<exp>/<run_id>.json (chỉ metrics, commit) + SUMMARY.md, index.csv sinh lại từ các file đó.

python -m mmxai.core.results [--exp <tên>] [--publish-all]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .env import ROOT

RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"
PRESERVED_KEYS = ("notes",)  # ghi tay sau khi chạy, không có trong metrics.json của lượt


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def publish(run_dir: Path | str, results_dir: Path | str = RESULTS_DIR) -> Path:
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    exp, run_id = metrics["run"]["exp"], metrics["run"]["run_id"]
    dst = Path(results_dir) / exp / f"{run_id}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        old = json.loads(dst.read_text(encoding="utf-8"))
        metrics.update({k: old[k] for k in PRESERVED_KEYS if k in old})
    dst.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    write_summary(exp, results_dir)
    return dst


def load_results(exp: str, results_dir: Path | str = RESULTS_DIR) -> list[dict]:
    out = []
    for f in sorted((Path(results_dir) / exp).glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"Bỏ qua file hỏng: {f}")
    return out


def write_summary(exp: str, results_dir: Path | str = RESULTS_DIR) -> Path:
    runs = load_results(exp, results_dir)
    runs.sort(key=lambda m: m["run"].get("finished_at", ""), reverse=True)
    metric_keys: list[str] = []
    for m in runs:
        for k in flatten(m.get("metrics", {})):
            if k not in metric_keys:
                metric_keys.append(k)
    head = ["run_id", "người chạy", "n", *metric_keys, "s/mẫu", "VRAM GB", "compute", "git", "ghi chú"]

    def row(m: dict) -> list[str]:
        r, flat = m["run"], flatten(m.get("metrics", {}))
        env = m.get("environment", {})
        return [r["run_id"], r.get("user", ""), str(m.get("n", "")),
                *[fmt(flat.get(k)) for k in metric_keys],
                fmt(r.get("sec_per_sample"), 1), fmt(r.get("peak_vram_gb"), 1), env.get("compute", ""),
                (env.get("git_commit") or "")[:7], str(m.get("notes", {}).get("text", ""))]

    def table(selected):
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        return lines + ["| " + " | ".join(row(m)) + " |" for m in selected]

    full = [m for m in runs if not m["run"]["args"].get("limit")]
    smoke = [m for m in runs if m["run"]["args"].get("limit")]
    lines = [f"# {exp}", "", "Sinh tự động bởi `make summary`, đừng sửa tay. Nhận xét viết vào results/JOURNAL.md.", ""]
    lines += ["## Lượt đầy đủ (mới nhất trước)", ""] + table(full)
    lines += ["", "## Lượt thử (--limit)", ""] + table(smoke)
    out = Path(results_dir) / exp / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out.with_name("index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head + ["limit", "tag"])
        for m in runs:
            w.writerow(row(m) + [m["run"]["args"].get("limit"), m["run"].get("tag")])
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", help="mặc định: mọi thí nghiệm có trong results/")
    p.add_argument("--publish-all", action="store_true", help="chép lại metrics của mọi lượt trong runs/ sang results/")
    args = p.parse_args(argv)
    exps = [args.exp] if args.exp else sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir()) if RESULTS_DIR.exists() else []
    if args.publish_all:
        for f in sorted(RUNS_DIR.glob("*/*/metrics.json")):
            if not args.exp or f.parent.parent.name == args.exp:
                publish(f.parent)
    for e in exps:
        if (RESULTS_DIR / e).exists():
            print(write_summary(e))


if __name__ == "__main__":
    main()
