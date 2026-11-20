"""Mỗi thí nghiệm = một module ở đây có hàm run(ctx) -> {"metrics": {...}, "n": int, "extra": {...}}.

Tên module = tên thí nghiệm = tên file cấu hình configs/experiments/<tên>.yaml. Xem _template.py.
"""

from pathlib import Path


def available() -> list[str]:
    return sorted(p.stem for p in Path(__file__).parent.glob("*.py") if not p.stem.startswith("_"))
