"""Ghi chú tay vào results/<exp>/<run_id>.json (giữ qua các lần publish lại): nộp ở đâu, điểm ngoài, nhận xét ngắn.

Dùng: python tools/note.py --run runs/<exp>/<run_id> --text "khớp paper ±0.3" [--kv leaderboard=0.91 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from make_kernel import PKG  # noqa: E402

results = __import__(f"{PKG}.core.results", fromlist=["write_summary"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="runs/<exp>/<run_id> hoặc results/<exp>/<run_id>.json")
    p.add_argument("--text", default="")
    p.add_argument("--kv", action="append", default=[], metavar="KEY=VAL")
    args = p.parse_args()
    run = Path(args.run)
    exp, run_id = run.parent.name, run.stem if run.suffix == ".json" else run.name
    path = ROOT / "results" / exp / f"{run_id}.json"
    if not path.exists():
        sys.exit(f"Không thấy {path.relative_to(ROOT)} (đã make pull / publish chưa?)")
    m = json.loads(path.read_text(encoding="utf-8"))
    notes = m.setdefault("notes", {})
    if args.text:
        notes["text"] = args.text
    for item in args.kv:
        k, _, v = item.partition("=")
        try:
            notes[k] = float(v)
        except ValueError:
            notes[k] = v
    notes["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    results.write_summary(exp)
    print(f"Đã ghi {notes} vào {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
