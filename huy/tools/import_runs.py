"""Nhập các lượt chạy ở nơi khác (Colab/Drive, máy khác) vào runs/ rồi chép metric sang results/.

Dùng: python tools/import_runs.py <thư mục chứa runs/ hoặc runs.zip>
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from make_kernel import PKG  # noqa: E402

results = __import__(f"{PKG}.core.results", fromlist=["publish"])


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)
    tmp = None
    if src.suffix == ".zip":
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(src).extractall(tmp)
        src = Path(tmp)
    n = 0
    for m in sorted(src.rglob("metrics.json")):
        d = m.parent
        dst = ROOT / "runs" / d.parent.name / d.name
        shutil.copytree(d, dst, dirs_exist_ok=True)
        print(f"{d.parent.name}/{d.name} → {results.publish(dst).relative_to(ROOT)}")
        n += 1
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"Đã nhập {n} lượt")


if __name__ == "__main__":
    main()
