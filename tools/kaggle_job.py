"""Chạy các notebook Kaggle của pipeline bằng kaggle CLI (thay cho bấm tay trên web): push → chờ → tải kết quả.

    python tools/kaggle_job.py run task1                    # Nhiệm vụ 1 (Thắng): VG∩COCO, split, CLIP, FAISS, baseline 3 seed
    python tools/kaggle_job.py run adamf --dataset psg      # AdaMF-MAT (Khánh) + ablation trên bundle PSG (hoặc vg)
    python tools/kaggle_job.py run task1 --smoke            # lượt thử nhanh (vài trăm ảnh / vài epoch)
    python tools/kaggle_job.py push|status|wait|pull <job> [--dataset ..] [--smoke]
    python tools/kaggle_job.py pull adamf --dataset psg --install   # chép kết quả vào results/ (task1: giải nén vào data/)

Huấn luyện QA-Graph trên Kaggle đã có lệnh riêng (`make push EXP=...`, tools/make_kernel.py) — không qua file này.
Notebook gốc KHÔNG bị sửa: bản đẩy lên được sinh trong build/kaggle_jobs/<kernel>/ (vá cấu hình ở ô đầu nếu cần).
Kernel: <KAGGLE_USER>/<slug> (KAGGLE_USER lấy từ local.mk hoặc `kaggle config view`), riêng tư, GPU T4, Internet bật.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle") if (ROOT / ".venv" / "bin" / "kaggle").exists() else "kaggle"
BUNDLE = {"vg": "huychau/mmxai-task1-bundle", "psg": "huychau/mmxai-psg-bundle"}

JOBS = {
    "task1": dict(
        notebook="notebooks/kaggle/vg_data/vg_data_clip_baseline.ipynb",
        slug="mmxai-vg-data", datasets=lambda ds: ["yenphamnhu/visual-genome"], datasets_ok=("vg",),
        patches=lambda ds: [],
        smoke=[('SMOKE = bool(int(os.environ.get("MMXAI_SMOKE", "0")))', "SMOKE = True")],
        outputs=r"^(vg_bundle\.zip|task1/results/.+)$"),  # không tải raw/ (dữ liệu thô ~200 MB)
    "adamf": dict(
        notebook="notebooks/kaggle/adamf/protocol_adamf_kaggle.ipynb",
        slug="mmxai-adamf-{ds}", datasets=lambda ds: [BUNDLE[ds]], datasets_ok=("vg", "psg"),
        # ablation (xáo / bỏ quan hệ) bật cho bản nộp; tên MMKG theo bộ dữ liệu
        patches=lambda ds: [("RUN_ABLATIONS = False", "RUN_ABLATIONS = True ")] + (
            [('NAMES = {"full": "COCO_full", "shuffle": "COCO_shuf", "drop": "COCO_drop"}',
              'NAMES = {"full": "PSG_full", "shuffle": "PSG_shuf", "drop": "PSG_drop"}')] if ds == "psg" else []),
        smoke=[("BUDGET_H = 2.5 ", "BUDGET_H = 0.01"), ("EPOCH_CAP = 300", "EPOCH_CAP = 2  ")],
        outputs=r"^(report_protocol\.txt|results_[A-Za-z_]+\.json|[A-Za-z]+_[a-z]+-(full|smoke)\.log)$"),  # không tải repo AdaMF-MAT
}


def user():
    mk = ROOT / "local.mk"
    if mk.exists():
        m = re.search(r"^KAGGLE_USER\s*[?:]?=\s*(\S+)", mk.read_text(), re.M)
        if m:
            return m.group(1)
    out = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True).stdout
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        sys.exit("Không biết KAGGLE_USER: ghi `KAGGLE_USER = <tên>` vào local.mk hoặc `kaggle config set -n username -v <tên>`")
    return m.group(1)


def kernel(job, ds, smoke):
    name = JOBS[job]["slug"].format(ds=ds) + ("-smoke" if smoke else "")
    return name, f"{user()}/{name}"


def build(job, ds, smoke) -> Path:
    J = JOBS[job]
    if ds not in J["datasets_ok"]:
        sys.exit(f"job {job} không hỗ trợ --dataset {ds} (chỉ {J['datasets_ok']})")
    name, kid = kernel(job, ds, smoke)
    nb = json.loads((ROOT / J["notebook"]).read_text(encoding="utf-8"))
    cfg = next(c for c in nb["cells"] if c["cell_type"] == "code")
    src = "".join(cfg["source"])
    for a, b in J["patches"](ds) + (J["smoke"] if smoke else []):
        if src.count(a) != 1:
            sys.exit(f"Ô cấu hình của {J['notebook']} không còn dòng `{a}` — cập nhật JOBS['{job}'] trong tools/kaggle_job.py")
        src = src.replace(a, b)
    cfg["source"] = src.splitlines(keepends=True)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    out = ROOT / "build" / "kaggle_jobs" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    code = Path(J["notebook"]).name
    (out / code).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"id": kid, "title": name, "code_file": code, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": J["datasets"](ds), "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def status(kid):
    out = subprocess.run([KAGGLE, "kernels", "status", kid], capture_output=True, text=True)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out.stdout + out.stderr)
    return (m.group(1) if m else "UNKNOWN").upper(), (out.stdout + out.stderr).strip()


def push(job, ds, smoke):
    d = build(job, ds, smoke)
    for k in range(60):  # Kaggle cho tối đa 2 phiên GPU: bị từ chối thì chờ rồi thử lại
        r = subprocess.run([KAGGLE, "kernels", "push", "-p", str(d)], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
        if "successfully pushed" in (r.stdout + r.stderr):
            return
        print(f"push chưa được, thử lại sau 120 s ({k + 1}/60)")
        time.sleep(120)
    sys.exit("push thất bại")


def wait(kid, poll=120):
    time.sleep(30)
    while True:
        st, raw = status(kid)
        print(f"{time.strftime('%H:%M:%S')} {kid}: {st}", flush=True)
        if st in ("COMPLETE", "ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED"):
            return st
        time.sleep(poll)


def pull(job, ds, smoke, install):
    name, kid = kernel(job, ds, smoke)
    dst = ROOT / "runs" / "kaggle" / name / time.strftime("%Y%m%d-%H%M%S")
    dst.mkdir(parents=True)
    subprocess.run([KAGGLE, "kernels", "output", kid, "-p", str(dst), "--file-pattern", JOBS[job]["outputs"]], check=True)
    print(f"→ {dst.relative_to(ROOT)}")
    if not install:
        return
    if job == "task1":
        z = next(dst.rglob("vg_bundle.zip"), None)
        if not z:
            sys.exit("Không thấy vg_bundle.zip trong output")
        tgt = ROOT / "data" / ("vg_bundle_smoke" if smoke else "vg_bundle")
        if tgt.exists() and any(tgt.iterdir()):
            sys.exit(f"{tgt.relative_to(ROOT)} đã có — xóa/đổi tên trước khi --install (tránh ghi đè bundle đang dùng)")
        zipfile.ZipFile(z).extractall(tgt)
        print(f"→ giải nén vào {tgt.relative_to(ROOT)} (tiếp: make extras DS=vg)")
    elif job == "adamf":
        tgt = ROOT / "results" / f"adamf_{ds}{'_smoke' if smoke else ''}"
        tgt.mkdir(parents=True, exist_ok=True)
        for f in list(dst.rglob("report_protocol.txt")) + list(dst.rglob("results_*.json")):
            shutil.copy2(f, tgt / f.name)
        print(f"→ chép báo cáo + results_*.json vào {tgt.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "push", "status", "wait", "pull", "run"])
    ap.add_argument("job", choices=list(JOBS))
    ap.add_argument("--dataset", default="vg", choices=list(BUNDLE))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--install", action="store_true", help="pull/run: đưa kết quả vào data/ (task1) hoặc results/ (adamf)")
    a = ap.parse_args()
    name, kid = kernel(a.job, a.dataset, a.smoke)
    if a.action == "build":
        print(build(a.job, a.dataset, a.smoke))
    elif a.action == "push":
        push(a.job, a.dataset, a.smoke)
    elif a.action == "status":
        print(status(kid)[1])
    elif a.action == "wait":
        wait(kid)
    elif a.action == "pull":
        pull(a.job, a.dataset, a.smoke, a.install)
    else:
        push(a.job, a.dataset, a.smoke)
        st = wait(kid)
        pull(a.job, a.dataset, a.smoke, a.install and st == "COMPLETE")
        if st != "COMPLETE":
            sys.exit(f"kernel {kid} kết thúc với trạng thái {st} — xem log trong runs/kaggle/{name}/")


if __name__ == "__main__":
    main()
