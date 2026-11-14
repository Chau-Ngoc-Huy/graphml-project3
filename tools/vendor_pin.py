"""Đổi commit ghim của repo gốc: checkout REF trong vendor/<name>, cập nhật configs/vendor.yaml: commit.

Dùng: python tools/vendor_pin.py <sha|tag|branch>   (không truyền: in commit hiện tại)
"""

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "configs" / "vendor.yaml"
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
vdir = ROOT / "vendor" / cfg["name"]
if len(sys.argv) > 1:
    subprocess.run(["git", "fetch", "-q", "--all", "--tags"], cwd=vdir, check=False)
    subprocess.run(["git", "checkout", "-q", sys.argv[1]], cwd=vdir, check=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=vdir, capture_output=True, text=True, check=True).stdout.strip()
text = re.sub(r"^commit:.*$", f"commit: {sha}", cfg_path.read_text(encoding="utf-8"), count=1, flags=re.M)
cfg_path.write_text(text, encoding="utf-8")
print(f"vendor/{cfg['name']} @ {sha} (đã ghi vào configs/vendor.yaml; nhớ `git add vendor configs/vendor.yaml` nếu là submodule)")
