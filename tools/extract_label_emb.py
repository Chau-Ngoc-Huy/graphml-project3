"""Trích CLIP text cho nhãn đối tượng (+ khái niệm ConceptNet) bằng ĐÚNG model trong features/meta.json của bundle Thắng.

Nhãn = data/object_vocab_train.json (2000 nhãn, đếm trên ảnh TRAIN) ∪ <extra>/concepts_added.json (nếu đã lọc ConceptNet).
Ghi: <extra>/label_emb.npy (float32, L2-normalize), labels.json, label_emb_meta.json. Chạy CPU vài phút.

Dùng: python tools/extract_label_emb.py --bundle data/task1_bundle [--extra data/task1_bundle/huy_extra]
        [--prompt "a photo of a {}"]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmxai.data.contract import normalize_label  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--bundle", required=True)
p.add_argument("--extra")
p.add_argument("--prompt", default="a photo of a {}")
p.add_argument("--batch", type=int, default=256)
a = p.parse_args()

root = Path(a.bundle)
extra = Path(a.extra) if a.extra else root / "huy_extra"
extra.mkdir(parents=True, exist_ok=True)
meta = json.loads((root / "features" / "meta.json").read_text())
syn = json.loads((extra / "synonyms.json").read_text()) if (extra / "synonyms.json").exists() else {}

labs = {normalize_label(l, syn) for l in json.loads((root / "data" / "object_vocab_train.json").read_text())}
if (extra / "concepts_added.json").exists():
    labs |= {normalize_label(l, syn) for l in json.loads((extra / "concepts_added.json").read_text())}
labs = sorted(labs)

import open_clip  # noqa: E402

torch.set_num_threads(4)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms(meta["model"], pretrained=meta["pretrained"], device=dev)
tok = open_clip.get_tokenizer(meta["model"])
model.eval()
embs = []
with torch.no_grad():
    for s in range(0, len(labs), a.batch):
        e = model.encode_text(tok([a.prompt.format(l) for l in labs[s:s + a.batch]]).to(dev)).float()
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
E = torch.cat(embs).numpy().astype(np.float32)
np.save(extra / "label_emb.npy", E)
(extra / "labels.json").write_text(json.dumps(labs, ensure_ascii=False))
(extra / "label_emb_meta.json").write_text(json.dumps({"model": meta["model"], "pretrained": meta["pretrained"],
                                                       "prompt": a.prompt, "n": len(labs)}))
print(f"{len(labs)} nhãn → {extra / 'label_emb.npy'} {E.shape} ({meta['model']}/{meta['pretrained']})")
