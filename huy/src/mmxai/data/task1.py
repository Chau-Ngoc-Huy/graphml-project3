"""Đọc bundle Nhiệm vụ 1 của Thắng (task1_bundle/) thành Bundle cho pipeline đồ thị.

Bundle của Thắng (KHÔNG sửa, chỉ đọc):
    data/images.jsonl      {idx, id, coco_id, vg_id, split, objects: [tên], relations: [[subj, pred, obj]]}
    data/captions.jsonl    {idx, img, text, group, concepts}
    data/object_vocab_train.json   2000 nhãn phổ biến nhất trên ảnh TRAIN
    features/clip_<model>.npz      img (N,512) f16, cap (Nc,512) f16, cap_img (Nc,), ok (N,)
    features/meta.json             model/pretrained CLIP — label_emb PHẢI cùng model

File của Huy, sinh thêm vào thư mục `extra` (mặc định <bundle>/huy_extra):
    label_emb.npy + labels.json    CLIP text của nhãn + khái niệm (tools/extract_label_emb.py)
    conceptnet.tsv                 cạnh khái niệm (tools/fetch_conceptnet.py)
    synonyms.json                  (tùy chọn) chuẩn hóa nhãn — khi Yến xong

image_id trong Bundle = idx của Thắng (= số dòng images.jsonl = chỉ số trong Protocol), caption_id = idx caption,
nên điểm đồ thị cắm thẳng vào Protocol.evaluate() mà không cần ánh xạ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import Bundle, normalize_label


def is_task1(path: Path | str) -> bool:
    return (Path(path) / "data" / "images.jsonl").exists()


def _jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_task1(bundle: Path | str, extra: Path | str | None = None, max_images: int | None = None) -> Bundle:
    root = Path(bundle)
    extra = Path(extra) if extra else root / "huy_extra"
    if not extra.exists() and (root / extra.name).exists():  # `extra: huy_extra_cn` = thư mục trong bundle (Kaggle)
        extra = root / extra.name
    syn_p = extra / "synonyms.json"
    syn = {k.lower(): v.lower() for k, v in json.loads(syn_p.read_text()).items()} if syn_p.exists() else {}

    images = _jsonl(root / "data" / "images.jsonl")
    caps = _jsonl(root / "data" / "captions.jsonl")
    feat_file = sorted((root / "features").glob("clip_*.npz"))[0]
    z = np.load(feat_file, allow_pickle=False)
    img_emb, cap_emb, cap_img, ok = (np.asarray(z["img"], np.float32), np.asarray(z["cap"], np.float32),
                                     np.asarray(z["cap_img"]), np.asarray(z["ok"], bool))
    assert len(images) == len(img_emb) and len(caps) == len(cap_emb), "images/captions không khớp file features"

    keep = [im["idx"] for im in images if ok[im["idx"]]]
    if max_images:  # lượt thử: giữ tỷ lệ split
        by = {}
        for i in keep:
            by.setdefault(images[i]["split"], []).append(i)
        frac = max_images / len(keep)
        keep = sorted(i for v in by.values() for i in v[: max(5, int(len(v) * frac))])
    kept = set(keep)

    splits = {s: [i for i in keep if images[i]["split"] == s] for s in ("train", "val", "test")}
    image_labels = {i: [normalize_label(o, syn) for o in images[i]["objects"]] for i in keep}
    image_rels = {i: [(normalize_label(s, syn), str(p).strip().lower(), normalize_label(o, syn))
                      for s, p, o in images[i]["relations"]] for i in keep}

    rows = [c["idx"] for c in caps if int(c["img"]) in kept]
    captions = [{"caption_id": caps[r]["idx"], "image_id": int(caps[r]["img"]), "text": caps[r]["text"],
                 "group": caps[r].get("group"),
                 "concepts": [normalize_label(x, syn) for x in caps[r].get("concepts") or []]} for r in rows]
    assert all(int(cap_img[r]) == int(caps[r]["img"]) for r in rows[:1000]), "cap_img lệch captions.jsonl"

    lab_p = extra / "labels.json"
    if not lab_p.exists():
        raise FileNotFoundError(f"Chưa có {lab_p}: chạy tools/extract_label_emb.py --bundle {root}")
    labels = [normalize_label(l, syn) for l in json.loads(lab_p.read_text())]
    label_emb = np.load(extra / "label_emb.npy").astype(np.float32)
    meta = json.loads((root / "features" / "meta.json").read_text())
    lmeta_p = extra / "label_emb_meta.json"
    if lmeta_p.exists():
        lm = json.loads(lmeta_p.read_text())
        if (lm.get("model"), lm.get("pretrained")) != (meta.get("model"), meta.get("pretrained")):
            raise ValueError(f"label_emb dùng {lm} nhưng features của Thắng dùng {meta} — phải cùng model CLIP")

    concept_edges = []
    cn = extra / "conceptnet.tsv"
    if cn.exists():
        for line in cn.read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 3:
                concept_edges.append((normalize_label(p[0], syn), p[1], normalize_label(p[2], syn)))

    return Bundle(image_ids=keep, image_emb=img_emb[keep], image_labels=image_labels, image_rels=image_rels,
                  captions=captions, caption_emb=cap_emb[rows], splits=splits, labels=labels, label_emb=label_emb,
                  concept_edges=concept_edges, root=root)
