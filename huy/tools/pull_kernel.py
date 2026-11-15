"""Tải output kernel của một thí nghiệm về runs/<exp>/<run_id>/ (kèm kaggle.log) rồi chép metrics sang results/.

Dùng: python tools/pull_kernel.py --exp <tên>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from make_kernel import PKG, kaggle_bin, kernel_id  # noqa: E402

results = __import__(f"{PKG}.core.results", fromlist=["publish"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True)
    args = p.parse_args()
    kernel = kernel_id(args.exp)
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
        subprocess.run([kaggle_bin(), "kernels", "output", kernel, "-p", tmp, "-o", "-q"], check=True)
        tmp = Path(tmp)
        logs = list(tmp.glob("*.log"))
        run_dirs = sorted(d for d in (tmp / "runs").glob("*/*") if d.is_dir()) if (tmp / "runs").exists() else []
        if not run_dirs:
            dst = ROOT / "runs" / args.exp / "_failed" / f"{datetime.now():%Y%m%d-%H%M%S}"
            dst.mkdir(parents=True, exist_ok=True)
            for log in logs:
                shutil.copy(log, dst / "kaggle.log")
            sys.exit(f"Kernel {kernel} không có kết quả trong runs/; log ở {dst.relative_to(ROOT)} (xem bằng tools/show_run.py)")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for d in run_dirs:
            dst = ROOT / "runs" / d.relative_to(tmp / "runs")
            existed = dst.exists()
            dst.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_dir():
                    shutil.copytree(f, dst / f.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(f, dst / f.name)
            for log in logs:
                shutil.copy(log, dst / (f"kaggle-{stamp}.log" if existed and (dst / "kaggle.log").exists() else "kaggle.log"))
            print(f"Đã tải về {dst.relative_to(ROOT)}")
            if (dst / "metrics.json").exists():
                print(f"Đã chép metric sang {results.publish(dst)}")


if __name__ == "__main__":
    main()
