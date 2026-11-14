"""Seed, thông tin môi trường (thư viện, git commit của harness và của repo paper), tên người chạy, secret."""

from __future__ import annotations

import getpass
import os
import platform
import random
import re
import subprocess
import sys
import unicodedata
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Thư viện ghi phiên bản vào metrics.json; thêm thư viện riêng của paper vào configs/vendor.yaml: track_packages
PACKAGES = ["torch", "numpy", "pyyaml"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git(*args: str, cwd: Path | str = ROOT) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def user_slug() -> str:
    """Tên ngắn của người chạy trong run_id: LAB_USER (local.mk) > git user.name > tài khoản máy."""
    name = os.environ.get("LAB_USER") or git("config", "user.name") or getpass.getuser()
    name = unicodedata.normalize("NFKD", name.replace("đ", "d").replace("Đ", "D"))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"


def compute_kind() -> str:
    if Path("/kaggle/working").exists():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists():
        return "colab"
    return "local"


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def environment_info(extra_packages: list[str] | None = None, vendor_dir: Path | None = None) -> dict:
    versions = {}
    for pkg in PACKAGES + list(extra_packages or []):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "compute": compute_kind(),
        "gpu": gpu_name(),
        "packages": versions,
        "git_commit": os.environ.get("LAB_GIT_COMMIT") or git("rev-parse", "HEAD"),
        "vendor_commit": os.environ.get("LAB_VENDOR_COMMIT") or (git("rev-parse", "HEAD", cwd=vendor_dir) if vendor_dir and vendor_dir.exists() else None),
    }


def load_secrets(names: list[str]) -> dict[str, str | None]:
    """Đọc key API: biến môi trường > .env ở gốc project > Kaggle Secrets (Add-ons → Secrets, cùng tên)."""
    out: dict[str, str | None] = {}
    env_file = ROOT / ".env"
    dotenv = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                dotenv[k.strip()] = v.strip().strip('"').strip("'")
    for name in names:
        val = os.environ.get(name) or dotenv.get(name)
        if not val and compute_kind() == "kaggle":
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore

                val = UserSecretsClient().get_secret(name)
            except Exception:
                val = None
        if val:
            os.environ[name] = val
        out[name] = val or None
    return out
