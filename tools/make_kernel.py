"""Sinh kernel Kaggle tự chứa: build/kaggle/<exp>/{run.py, kernel-metadata.json}.

run.py nhúng src/ + configs/ (zip, base64). Trên Kaggle nó: giải nén vào /kaggle/working/_code, clone repo paper ở
commit đã ghim (configs/vendor.yaml) hoặc giải nén bản đóng gói kèm (--bundle-vendor), cài pip, đổi `--set data=<tên>`
thành đường dẫn thật trong /kaggle/input, rồi chạy `python -m mmxai.run --exp ... --runs-dir /kaggle/working/runs`.
Kernel id = <KAGGLE_USER>/<PROJECT>-<exp>: mỗi thí nghiệm một kernel, mỗi người một tài khoản.

Dùng: python tools/make_kernel.py --exp <tên> [--bundle-vendor] [--dataset owner/slug,...] -- --limit 4 --tag thu
Đối số sau `--` chuyển nguyên cho mmxai.run (không truyền --runs-dir, --config-dir, --exp).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = next(p.name for p in (ROOT / "src").iterdir() if (p / "run.py").exists())
def _project_name() -> str:
    """Tên project cố định từ paper-lab.config.json (cố định để tên kernel Kaggle không đổi)."""
    try:
        return json.loads((ROOT / "paper-lab.config.json").read_text())["project"]
    except (OSError, KeyError, ValueError):
        return ROOT.name.lower()


PROJECT = os.environ.get("PROJECT") or _project_name()
INCLUDE = [f"src/{PKG}", "src/mmxai_eval.py", "configs"]   # mmxai_eval: bộ đánh giá chung (một bản duy nhất)
TEMPLATE_META = ROOT / "kaggle" / "kernel-metadata.template.json"

TEMPLATE = '''# File sinh tự động bởi tools/make_kernel.py, không sửa tay.
# git: {git}  exp: {exp}
import base64, io, os, re, runpy, subprocess, sys, zipfile
from pathlib import Path

PAYLOAD = "{payload}"
PKG = {pkg!r}
ARGS = {args!r}
GIT_COMMIT = {git!r}
LAB_USER = {user!r}
VENDOR = {vendor!r}          # name, url, commit, install, pip, mode
BUNDLED_VENDOR = {bundled_vendor!r}

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
CODE = WORK / "_code"
VENDOR_DIR = WORK / "vendor" / (VENDOR.get("name") or "_none")


def sh(cmd, cwd=None, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)


def find_input(name):
    """Thư mục trong /kaggle/input có tên chứa `name` (tên đúng bằng được ưu tiên)."""
    if Path(name).exists():
        return name
    # ưu tiên thư mục TÊN ĐÚNG BẰNG `name` ở độ sâu 1–3 (/kaggle/input/<slug> hoặc /kaggle/input/datasets/<owner>/<slug>),
    # không lấy nhầm thư mục con như <slug>/data chỉ vì đường dẫn chứa tên
    cands = [p for pat in ("*", "*/*", "*/*/*") for p in INPUT.glob(pat) if p.is_dir()]
    hits = sorted([p for p in cands if p.name == name] or [p for p in cands if name in p.name],
                  key=lambda d: len(str(d)))
    if not hits:
        sys.exit(f"Không tìm thấy dataset chứa '{{name}}' trong {{INPUT}}: " + ", ".join(str(p) for p in INPUT.glob("*/*")))
    return str(hits[0])


def resolve_data_args(args):
    """--set data=<tên> / --set data.path=<tên> → đường dẫn thật trong /kaggle/input."""
    out = []
    for a in args:
        m = re.match(r"^(data(?:\\.[a-z_]+)?)=(.+)$", a)
        if m and not m.group(2).startswith("/"):
            a = f"{{m.group(1)}}={{find_input(m.group(2))}}"
        out.append(a)
    return out


def main():
    zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD)))
    zf.extractall(CODE)
    sys.path.insert(0, str(CODE / "src"))
    os.environ.update(LAB_GIT_COMMIT=GIT_COMMIT, LAB_USER=LAB_USER, LAB_VENDOR_DIR=str(VENDOR_DIR))
    os.chdir(WORK)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("Không có GPU (nvidia-smi không tồn tại)")

    if VENDOR.get("pip"):  # thư viện project cần (vd. torch_geometric), cài cả khi không có repo vendor
        sh(f"{{sys.executable}} -m pip install -q " + " ".join(repr(p) for p in VENDOR["pip"]))
    if VENDOR.get("name"):
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        if BUNDLED_VENDOR:
            for name in zf.namelist():
                if name.startswith("vendor/"):
                    zf.extract(name, WORK)
        elif VENDOR.get("url"):
            sh(f"git clone -q {{VENDOR['url']}} {{VENDOR_DIR}}")
            if VENDOR.get("commit"):
                sh(f"git checkout -q {{VENDOR['commit']}}", cwd=VENDOR_DIR)
                os.environ["LAB_VENDOR_COMMIT"] = VENDOR["commit"]
        for cmd in VENDOR.get("install") or []:
            sh(cmd, cwd=VENDOR_DIR)

    args = resolve_data_args(list(ARGS)) + ["--config-dir", str(CODE / "configs"), "--runs-dir", str(WORK / "runs"), "--no-publish"]
    print(f"python -m {{PKG}}.run", " ".join(args), flush=True)
    sys.argv = [f"{{PKG}}.run"] + args
    try:
        runpy.run_module(f"{{PKG}}.run", run_name="__main__")
    finally:
        import shutil

        shutil.rmtree(CODE, ignore_errors=True)       # output của kernel chỉ còn runs/
        shutil.rmtree(WORK / "vendor", ignore_errors=True)


main()
'''


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *INCLUDE], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def kaggle_bin() -> str:
    k = ROOT / ".venv" / "bin" / "kaggle"
    return str(k) if k.exists() else "kaggle"


def kaggle_user() -> str:
    if os.environ.get("KAGGLE_USER"):
        return os.environ["KAGGLE_USER"]
    out = subprocess.run([kaggle_bin(), "config", "view"], capture_output=True, text=True).stdout
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        sys.exit("Không xác định được username Kaggle: đặt KAGGLE_USER trong local.mk")
    return m.group(1)


def kernel_id(exp: str) -> str:
    """Kaggle chuẩn hóa slug ('_' → '-'); dùng sẵn dạng đó để status/wait/pull tìm đúng kernel."""
    return f"{kaggle_user()}/{PROJECT}-{exp}".lower().replace("_", "-")


def build_payload(bundle_vendor: bool, vendor_name: str | None) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for inc in INCLUDE:
            for path in sorted((ROOT / inc).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    zf.write(path, path.relative_to(ROOT).as_posix())
        if bundle_vendor and vendor_name:
            vdir = ROOT / "vendor" / vendor_name
            for path in sorted(vdir.rglob("*")):
                if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                    zf.write(path, path.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def main(argv: list[str]) -> None:
    if "--" in argv:
        i = argv.index("--")
        own, args = argv[:i], argv[i + 1 :]
    else:
        own, args = argv, []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True)
    p.add_argument("--dataset", default=os.environ.get("DATASET", ""), help="owner/slug Kaggle, nhiều cái ngăn bằng dấu phẩy")
    p.add_argument("--bundle-vendor", action="store_true", help="đóng gói vendor/ vào run.py thay vì git clone trên Kaggle (repo private / nhỏ)")
    p.add_argument("--out", default=None)
    opts = p.parse_args(own)
    for forbidden in ("--runs-dir", "--config-dir", "--exp"):
        if forbidden in args:
            sys.exit(f"{forbidden} do make_kernel tự đặt, đừng truyền sau `--`")

    vendor = yaml.safe_load((ROOT / "configs" / "vendor.yaml").read_text(encoding="utf-8")) or {}
    vendor = {k: vendor.get(k) for k in ("name", "url", "commit", "install", "pip", "mode")}
    if vendor["name"] in (None, "VENDOR_NAME", ""):
        vendor = {"name": None, "pip": vendor.get("pip") or []}
    if vendor.get("name") and not opts.bundle_vendor and not vendor.get("url"):
        print("CẢNH BÁO: vendor không có url → dùng --bundle-vendor để đóng gói kèm")
    vcfg_full = yaml.safe_load((ROOT / "configs" / "vendor.yaml").read_text(encoding="utf-8")) or {}

    meta = json.loads(TEMPLATE_META.read_text(encoding="utf-8"))
    datasets = [d.strip() for d in opts.dataset.split(",") if d.strip()] + list(vcfg_full.get("kaggle_datasets") or [])
    meta.update(id=kernel_id(opts.exp), title=f"{PROJECT}-{opts.exp}", dataset_sources=sorted(set(datasets)))
    meta["model_sources"] = sorted(set(meta.get("model_sources", [])) | set(vcfg_full.get("kaggle_models") or []))

    out_dir = Path(opts.out) if opts.out else ROOT / "build" / "kaggle" / opts.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    run_py = out_dir / meta["code_file"]
    run_py.write_text(TEMPLATE.format(
        payload=build_payload(opts.bundle_vendor, vendor.get("name")), pkg=PKG, args=["--exp", opts.exp, *args],
        git=git_commit(), exp=opts.exp, user=os.environ.get("LAB_USER", ""), vendor=vendor, bundled_vendor=opts.bundle_vendor,
    ), encoding="utf-8")
    size_kb = run_py.stat().st_size // 1024
    print(f"Đã sinh {run_py.relative_to(ROOT)} ({size_kb} KB) cho kernel {meta['id']}: --exp {opts.exp} {' '.join(args)}")
    if size_kb > 900:
        print("CẢNH BÁO: run.py lớn (>900 KB) — Kaggle có thể từ chối; bỏ --bundle-vendor hoặc đẩy vendor lên dataset Kaggle")


if __name__ == "__main__":
    main(sys.argv[1:])
