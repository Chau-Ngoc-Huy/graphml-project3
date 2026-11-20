"""Trích đặc trưng OpenCLIP ĐÓNG BĂNG theo hợp đồng dữ liệu (docs/data-contract.md). Bản tham chiếu: phần chính
thức là — chỉ cần file đầu ra cùng tên/định dạng. Chạy trên Kaggle GPU (ảnh) hoặc CPU (chỉ --labels, --captions).

  --images    features/image_emb.npy + image_ids.json      ảnh <images_dir>/<image_id>.jpg
  --captions  features/caption_emb.npy + caption_ids.json  từ captions.json
  --labels    features/label_emb.npy + labels.json         tên nhãn VG (sau chuẩn hóa) + khái niệm ConceptNet

Dùng: python tools/extract_clip.py --data <data_dir> --images-dir <dir> --images --captions --labels \
        [--model ViT-B-32 --pretrained laion2b_s34b_b79k] [--prompt "a photo of a {}"]
Cần: pip install open_clip_torch pillow
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", required=True)
p.add_argument("--images-dir")
p.add_argument("--images", action="store_true")
p.add_argument("--captions", action="store_true")
p.add_argument("--labels", action="store_true")
p.add_argument("--model", default="ViT-B-32")
p.add_argument("--pretrained", default="laion2b_s34b_b79k")
p.add_argument("--prompt", default="a photo of a {}")
p.add_argument("--batch", type=int, default=256)
a = p.parse_args()

import open_clip  # noqa: E402
from PIL import Image  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained, device=dev)
tok = open_clip.get_tokenizer(a.model)
model.eval()
root = Path(a.data)
out = root / "features"
out.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def enc_text(texts):
    embs = []
    for s in range(0, len(texts), a.batch):
        e = model.encode_text(tok(texts[s:s + a.batch]).to(dev)).float()
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    return torch.cat(embs).numpy().astype(np.float32)


if a.captions:
    caps = json.loads((root / "captions.json").read_text())
    np.save(out / "caption_emb.npy", enc_text([c["text"] for c in caps]))
    (out / "caption_ids.json").write_text(json.dumps([c["caption_id"] for c in caps]))
    print("captions:", len(caps))

if a.labels:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mmxai.data.contract import normalize_label

    syn = json.loads((root / "synonyms.json").read_text()) if (root / "synonyms.json").exists() else {}
    labs = {normalize_label(o["names"][0], syn) for r in json.loads((root / "vg" / "objects.json").read_text())
            for o in r.get("objects", []) if o.get("names")}
    if (root / "concepts_added.json").exists():
        labs |= set(json.loads((root / "concepts_added.json").read_text()))
    labs = sorted(labs)
    np.save(out / "label_emb.npy", enc_text([a.prompt.format(l) for l in labs]))
    (out / "labels.json").write_text(json.dumps(labs, ensure_ascii=False))
    print("labels:", len(labs))

if a.images:
    ids = sorted({int(r["image_id"]) for r in json.loads((root / "vg" / "objects.json").read_text())})
    ids = [i for i in ids if (Path(a.images_dir) / f"{i}.jpg").exists()]
    embs = []
    with torch.no_grad():
        for s in range(0, len(ids), a.batch):
            x = torch.stack([preprocess(Image.open(Path(a.images_dir) / f"{i}.jpg").convert("RGB")) for i in ids[s:s + a.batch]])
            e = model.encode_image(x.to(dev)).float()
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
            print(f"  ảnh {s + len(x)}/{len(ids)}", flush=True)
    np.save(out / "image_emb.npy", torch.cat(embs).numpy().astype(np.float32))
    (out / "image_ids.json").write_text(json.dumps(ids))
    print("images:", len(ids))
(out / "clip_model.json").write_text(json.dumps({"model": a.model, "pretrained": a.pretrained, "prompt": a.prompt}))
