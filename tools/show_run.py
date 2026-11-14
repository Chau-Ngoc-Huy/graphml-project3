"""Xem nhanh một lượt: log Kaggle (lọc), lỗi, metric, môi trường.

Dùng: python tools/show_run.py runs/<exp>/<run_id>   (không truyền: lượt mới nhất; thư mục _failed/<stamp> cũng được)
"""

import json
import sys
from pathlib import Path

d = Path(sys.argv[1]) if len(sys.argv) > 1 else max((p.parent for p in Path("runs").glob("*/*/metrics.json")), key=lambda p: p.stat().st_mtime)
print(d)
for log in sorted(d.glob("kaggle*.log")):
    raw = log.read_text(encoding="utf-8")
    try:
        text = "".join(x["data"] for x in json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        text = raw
    keep = [l for l in text.splitlines() if l.startswith(("[", "$ ", "Traceback", "  File")) or "Error" in l or "error" in l.lower()[:40]]
    print(f"--- {log.name} ({len(text.splitlines())} dòng, lọc còn {len(keep)}):")
    print("\n".join(keep[-60:]))
if (d / "metrics.json").exists():
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    print("metrics:", json.dumps(m["metrics"], ensure_ascii=False))
    run = m["run"]
    print({k: run.get(k) for k in ("run_id", "wall_time_s", "sec_per_sample", "peak_vram_gb", "tag")})
    if run.get("error"):
        print("LỖI:\n" + run["error"][-2000:])
    print("env:", json.dumps({k: v for k, v in m["environment"].items() if k != "packages"}, ensure_ascii=False))
