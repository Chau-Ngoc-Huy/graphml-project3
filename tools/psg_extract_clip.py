"""Đặc trưng CLIP ĐÓNG BĂNG cho bundle PSG (tools/psg_to_bundle.py), cùng định dạng features/ của bundle VG.

Ảnh đọc thẳng từ bytes trong parquet HF (maelic/PSG-coco-format), caption từ data/captions.jsonl.
Ghi features/clip_<model>_<tag>.npz {img (N,D) f16, cap (Nc,D) f16, cap_img (Nc,) i32, ok (N,) bool} + meta.json.
Có điểm lưu tạm (features/_img_part.npy) → chạy lại sẽ tiếp tục chỗ dừng.

Dùng: OMP_NUM_THREADS=2 nice -n 19 python -I tools/psg_extract_clip.py --bundle data/psg_bundle --raw data/psg_raw --threads 2
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--bundle", required=True)
p.add_argument("--raw", required=True)
p.add_argument("--model", default="ViT-B-32")
p.add_argument("--pretrained", default="laion2b_s34b_b79k")
p.add_argument("--tag", default="laion2b")
p.add_argument("--threads", type=int, default=2)
p.add_argument("--batch", type=int, default=64)
a = p.parse_args()

import open_clip  # noqa: E402
from PIL import Image  # noqa: E402

torch.set_num_threads(a.threads)
root, raw = Path(a.bundle), Path(a.raw)
feat = root / "features"
feat.mkdir(exist_ok=True)
images = [json.loads(l) for l in open(root / "data" / "images.jsonl", encoding="utf-8")]
caps = [json.loads(l) for l in open(root / "data" / "captions.jsonl", encoding="utf-8")]
row_of = {im["psg_id"]: im["idx"] for im in images}
model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
tok = open_clip.get_tokenizer(a.model)
model.eval()
D = model.visual.output_dim
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- ảnh
part = feat / "_img_part.npy"
okp = feat / "_ok_part.npy"
img = np.load(part) if part.exists() else np.zeros((len(images), D), np.float32)
ok = np.load(okp) if okp.exists() else np.zeros(len(images), bool)
todo = len(images) - int(ok.sum())
log(f"{len(images)} ảnh, còn {todo} chưa mã hóa")
done_since_save = 0
for f in sorted((raw / "data").glob("*.parquet")):
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        ids = pf.read_row_group(rg, columns=["image_id"]).column(0).to_pylist()
        need = [k for k, i in enumerate(ids) if i in row_of and not ok[row_of[i]]]
        if not need:
            continue
        tab = pf.read_row_group(rg, columns=["image_id", "image"])
        ims = tab.column("image").to_pylist()
        for s in range(0, len(need), a.batch):
            ks = need[s:s + a.batch]
            x, rows = [], []
            for k in ks:
                try:
                    x.append(preprocess(Image.open(io.BytesIO(ims[k]["bytes"])).convert("RGB")))
                    rows.append(row_of[ids[k]])
                except Exception as e:  # noqa: BLE001 — ảnh hỏng: ok=False, Protocol tự bỏ
                    log(f"ảnh lỗi {ids[k]}: {e}")
            if not x:
                continue
            with torch.no_grad():
                e = torch.nn.functional.normalize(model.encode_image(torch.stack(x)).float(), dim=-1).numpy()
            img[rows], ok[rows] = e, True
            done_since_save += len(rows)
            if done_since_save >= 1000:
                np.save(part, img); np.save(okp, ok); done_since_save = 0
                log(f"ảnh {int(ok.sum())}/{len(images)}")
np.save(part, img); np.save(okp, ok)
log(f"xong ảnh: {int(ok.sum())}/{len(images)} ok")

# ---------------------------------------------------------------- caption
cap = np.zeros((len(caps), D), np.float32)
with torch.no_grad():
    for s in range(0, len(caps), 256):
        e = model.encode_text(tok([c["text"] for c in caps[s:s + 256]])).float()
        cap[s:s + 256] = torch.nn.functional.normalize(e, dim=-1).numpy()
        if (s // 256) % 40 == 0:
            log(f"caption {s}/{len(caps)}")
cap_img = np.array([c["img"] for c in caps], np.int32)

out = feat / f"clip_{a.model}_{a.tag}.npz"
np.savez(out, img=img.astype(np.float16), cap=cap.astype(np.float16), cap_img=cap_img, ok=ok)
(feat / "meta.json").write_text(json.dumps({"model": a.model, "pretrained": a.pretrained, "dim": D,
                                            "n_images": len(images), "n_images_ok": int(ok.sum()),
                                            "n_captions": len(caps), "normalized": True, "dtype": "float16"}, indent=1))
part.unlink(); okp.unlink()
log(f"→ {out}")
