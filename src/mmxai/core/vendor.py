"""Gọi code gốc của paper (vendor/<tên>) từ adapter: tìm thư mục, chạy lệnh có log, đọc kết quả."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from .config import load_yaml
from .env import ROOT


def vendor_config() -> dict:
    return load_yaml(ROOT / "configs" / "vendor.yaml")


def vendor_dir(name: str | None = None) -> Path:
    """Thư mục repo gốc: LAB_VENDOR_DIR (Kaggle/Colab đặt) > vendor/<name> trong project."""
    if os.environ.get("LAB_VENDOR_DIR"):
        return Path(os.environ["LAB_VENDOR_DIR"])
    name = name or vendor_config().get("name")
    if not name:
        raise FileNotFoundError("configs/vendor.yaml chưa có `name` và LAB_VENDOR_DIR chưa đặt")
    d = ROOT / "vendor" / name
    if not d.exists() or not any(d.iterdir()):
        raise FileNotFoundError(f"{d} trống: chạy `make vendor` (git submodule update --init) hoặc đặt LAB_VENDOR_DIR")
    return d


def add_to_path(sub: str = "") -> Path:
    """Cho phép `import` code gốc trực tiếp (khi repo paper là package Python)."""
    d = vendor_dir() / sub if sub else vendor_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d


def run_cmd(cmd: str | list[str], cwd: Path | str | None = None, log_file: Path | None = None,
            env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Chạy CLI của repo gốc, in ra màn hình đồng thời ghi vào log_file (runs/<run_id>/vendor.log)."""
    args = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    cwd = Path(cwd) if cwd else vendor_dir()
    print(f"[vendor] cd {cwd} && {' '.join(shlex.quote(a) for a in args)}", flush=True)
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(args, cwd=cwd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    log = open(log_file, "a", encoding="utf-8") if log_file else None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lines.append(line)
            if log:
                log.write(line)
    finally:
        if log:
            log.close()
    rc = proc.wait()
    result = subprocess.CompletedProcess(args, rc, "".join(lines), "")
    if check and rc != 0:
        raise RuntimeError(f"Lệnh vendor trả mã {rc}: {' '.join(args)}")
    return result


def last_json(text: str) -> dict | None:
    """JSON cuối cùng trong stdout (nhiều repo in dict kết quả ở cuối)."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def grep_metrics(text: str, patterns: dict[str, str]) -> dict:
    r"""Bắt số trong log theo regex: {"acc": r"accuracy[:=]\s*([\d.]+)"} → {"acc": 0.87} (lấy lần cuối)."""
    out = {}
    for key, pat in patterns.items():
        hits = re.findall(pat, text, flags=re.IGNORECASE)
        if hits:
            val = hits[-1] if not isinstance(hits[-1], tuple) else hits[-1][0]
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
