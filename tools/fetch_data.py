"""Tải dữ liệu thô công khai của pipeline (bước 0). Bỏ qua file đã có; tải lại khi lỗi mạng (thử 5 lần).

    python tools/fetch_data.py captions conceptnet psg      # dữ liệu thô cho mọi bước chạy LOCAL
    python tools/fetch_data.py bundles models               # (đường tắt) bundle + checkpoint đã dựng sẵn trên Kaggle (riêng tư)

Nguồn (đều công khai trừ bundles/models):
  captions    Karpathy split MS-COCO (5 caption/ảnh)          cs.stanford.edu/people/karpathy/deepimagesent → data/raw/dataset_coco.json
  conceptnet  ConceptNet 5 tiếng Anh, bản HF (CC BY-SA 4.0)   peandrew/conceptnet_en_simple → data/conceptnet/hf_en_simple/train.parquet
  psg         PSG (ECCV 2022): psg.json (có coco_image_id) từ HarborYuan/OpenPSG; ảnh + split từ maelic/PSG-coco-format
              (test, val, train shard 0–5 — đúng 6 shard dùng cho bundle PSG; psg_to_bundle lấy mẫu 15 000 ảnh train từ đó)
  bundles     Kaggle datasets huychau/mmxai-vg-bundle (VG), huychau/mmxai-psg-bundle (PSG) → data/vg_bundle, data/psg_bundle
  models      Kaggle model huychau/mmxai-qa-graph (4 checkpoint seed 0) → models/<biến thể>/
Visual Genome (scene_graphs.json) KHÔNG tải ở đây: notebook giao thức đánh giá đọc trực tiếp dataset Kaggle yenphamnhu/visual-genome
(`make data DS=vg` — xem docs/PIPELINE.md).
"""

import argparse
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
PSG_FILES = ["data/test-00000-of-00001.parquet", "data/val-00000-of-00001.parquet"] + \
            [f"data/train-{i:05d}-of-00016.parquet" for i in range(6)] + ["README.md", "categories.json"]
MODELS = ["vg-qa-seed0:vg_t1_qa_seed0", "vg-qa-cn-seed0:vg_t1_qa_cn_seed0", "psg-qa-seed0:psg_qa_seed0",
          "psg-qa-cn-seed0:psg_qa_cn_seed0"]


def get(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  có rồi  {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(5):
        try:
            print(f"  tải     {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "mmxai-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dst)
            return
        except Exception as e:  # noqa: BLE001 — mạng chập chờn: thử lại
            print(f"  lỗi ({e}); thử lại {k + 1}/5")
            time.sleep(5 * (k + 1))
    sys.exit(f"Không tải được {url}")


def kaggle(*args):
    exe = ROOT / ".venv" / "bin" / "kaggle"
    subprocess.run([str(exe) if exe.exists() else "kaggle", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["captions", "conceptnet", "psg", "bundles", "models"])
    for w in ap.parse_args().what:
        print(f"[{w}]")
        if w == "captions":
            z = DATA / "raw" / "caption_datasets.zip"
            if (DATA / "raw" / "dataset_coco.json").exists():
                print("  có rồi  data/raw/dataset_coco.json")
            else:
                get("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip", z)
                with zipfile.ZipFile(z) as f:
                    f.extract("dataset_coco.json", DATA / "raw")
        elif w == "conceptnet":
            get(HF.format(repo="peandrew/conceptnet_en_simple", path="data/train-00000-of-00001.parquet"),
                DATA / "conceptnet" / "hf_en_simple" / "train.parquet")
        elif w == "psg":
            get(HF.format(repo="HarborYuan/OpenPSG", path="psg.json"), DATA / "psg_raw" / "psg.json")
            for f in PSG_FILES:
                get(HF.format(repo="maelic/PSG-coco-format", path=f), DATA / "psg_raw" / f)
        elif w == "bundles":
            for slug, d in (("huychau/mmxai-vg-bundle", "vg_bundle"), ("huychau/mmxai-psg-bundle", "psg_bundle")):
                if (DATA / d / "data" / "images.jsonl").exists():
                    print(f"  có rồi  data/{d}")
                else:
                    kaggle("datasets", "download", slug, "-p", str(DATA / d), "--unzip")
        elif w == "models":
            for m in MODELS:
                inst, d = m.split(":")
                if (ROOT / "models" / d / "model.pt").exists():
                    print(f"  có rồi  models/{d}")
                else:
                    kaggle("models", "instances", "versions", "download", f"huychau/mmxai-qa-graph/pyTorch/{inst}/1",
                           "-p", str(ROOT / "models" / d), "--untar")


if __name__ == "__main__":
    main()
