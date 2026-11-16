"""Dữ liệu giả đúng định dạng đầu ra của Thắng (data/images.jsonl, captions.jsonl, split/split.json, features/clip_*.npz)."""
import hashlib
import json

import numpy as np
import pytest

WORDS = ["man", "woman", "horse", "dog", "ball", "field", "tree", "grass", "sky", "bench", "bus", "street", "child",
         "kite", "beach", "boat", "water", "table", "cup", "pizza"]
RELS = [("man", "riding", "horse"), ("child", "playing", "ball"), ("dog", "on", "grass"), ("man", "on", "bench"),
        ("boat", "on", "water"), ("child", "flying", "kite")]
D = 64


def wv(w):
    return np.random.RandomState(int(hashlib.md5(w.encode()).hexdigest()[:8], 16)).randn(D)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture()
def work_dir(tmp_path):
    rng = np.random.RandomState(0)
    n = 400
    split = {"train": [], "val": [], "test": []}
    images, captions = [], []
    for i in range(n):
        s = "train" if i < 280 else ("val" if i < 340 else "test")
        base = RELS[rng.randint(len(RELS))]
        objs = list(dict.fromkeys([base[0], base[2]] + list(rng.choice(WORDS, 3, replace=False))))
        images.append(dict(idx=i, id=f"coco{i}", coco_id=i, vg_id=i, split=s, url="", coco_url="", objects=objs,
                           relations=[list(base)]))
        split[s].append(i)
        for _ in range(5):
            cs = [o for o in objs if rng.rand() < 0.7] or objs[:1]
            captions.append(dict(idx=len(captions), img=i, text="a " + " and a ".join(cs), group="direct", concepts=cs))
    for sub in ("data", "split", "features"):
        (tmp_path / sub).mkdir()
    (tmp_path / "split" / "split.json").write_text(json.dumps(dict(split, meta=dict(gallery_size=50, top_k=20, seed_data=0))))
    (tmp_path / "data" / "images.jsonl").write_text("".join(json.dumps(x) + "\n" for x in images))
    (tmp_path / "data" / "captions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in captions))
    img = unit(np.stack([np.mean([wv(o) for o in im["objects"]], 0) + 0.7 * rng.randn(D) for im in images]))
    cap = unit(np.stack([np.mean([wv(w) for w in c["concepts"]], 0) + 0.7 * rng.randn(D) for c in captions]))
    np.savez(tmp_path / "features" / "clip_fake.npz", img=img.astype("float16"), cap=cap.astype("float16"),
             cap_img=np.array([c["img"] for c in captions]), ok=np.ones(n, bool))
    return tmp_path


def fake_text_encode(texts):
    """CLIP giả: trung bình vector của các từ trong câu (cùng chiều D với ảnh)."""
    return unit(np.stack([np.mean([wv(w) for w in t.split()], 0) for t in texts]).astype("float32"))
