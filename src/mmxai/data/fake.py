"""Sinh dữ liệu GIẢ đúng định dạng hợp đồng dữ liệu (docs/data-contract.md) để chạy pipeline khi chưa có dữ liệu thật.

Dữ liệu giả được thiết kế để có cả truy vấn trực tiếp và gián tiếp:
- Mỗi nhãn cụ thể (boy, ball...) có một vectơ "CLIP" ngẫu nhiên; khái niệm trừu tượng (child, sport...) có vectơ riêng,
  KHÔNG gần vectơ con của nó → CLIP thuần không nối được "child" với ảnh chỉ có "boy".
- Ảnh = trung bình vectơ các đối tượng + nhiễu (CLIP nhìn thấy đối tượng nhưng không hoàn hảo).
- Caption trực tiếp nhắc tên đối tượng; caption gián tiếp thay một đối tượng bằng khái niệm cha (boy → child).
- ConceptNet giả chứa cạnh IsA (boy IsA child) → đồ thị là nơi duy nhất có tri thức để giải truy vấn gián tiếp.
Nếu pipeline đúng, CLIP+Graph phải tốt hơn CLIP rõ rệt ở nhóm gián tiếp, và ablation (xáo/ngẫu nhiên cạnh) phải mất lợi thế đó.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PREDICATES = ["on", "near", "holding", "wearing", "behind", "playing with", "next to", "under"]


def generate(out_dir: Path | str, n_images: int = 2000, n_concepts: int = 12, children_per_concept: int = 4,
             n_extra: int = 20, dim: int = 64, img_noise: float = 0.15, cap_noise: float = 0.1,
             caps_per_image: int = 2, p_indirect: float = 0.4, split=(0.7, 0.1, 0.2), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "vg").mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(parents=True, exist_ok=True)

    concepts = [f"concept{c}" for c in range(n_concepts)]
    children = {c: [f"{c}_obj{j}" for j in range(children_per_concept)] for c in concepts}
    parent = {o: c for c, objs in children.items() for o in objs}
    extras = [f"thing{j}" for j in range(n_extra)]          # đối tượng không có khái niệm cha
    concrete = list(parent) + extras
    labels = concepts + concrete

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    label_emb = unit(rng.normal(size=(len(labels), dim))).astype(np.float32)
    lid = {l: i for i, l in enumerate(labels)}

    objects_json, rel_json, captions, image_emb = [], [], [], []
    caption_emb, cap_ids = [], []
    obj_counter = 0
    for i in range(n_images):
        image_id = i + 1
        k = int(rng.integers(3, 6))
        objs = list(rng.choice(concrete, size=k, replace=False))
        recs = []
        for name in objs:
            obj_counter += 1
            recs.append({"object_id": obj_counter, "names": [name], "x": 0, "y": 0, "w": 10, "h": 10})
        objects_json.append({"image_id": image_id, "objects": recs})
        rels = []
        for _ in range(int(rng.integers(1, 4))):
            a, b = rng.choice(len(recs), size=2, replace=False)
            rels.append({"predicate": str(rng.choice(PREDICATES)),
                         "subject": {"object_id": recs[a]["object_id"], "names": recs[a]["names"]},
                         "object": {"object_id": recs[b]["object_id"], "names": recs[b]["names"]}})
        rel_json.append({"image_id": image_id, "relationships": rels})
        v = label_emb[[lid[o] for o in objs]].mean(0) + img_noise * rng.normal(size=dim) / np.sqrt(dim) * 4
        image_emb.append(unit(v))

        for c in range(caps_per_image):
            pick = list(rng.choice(objs, size=2, replace=False))
            words, indirect = list(pick), False
            if rng.random() < p_indirect:
                cand = [j for j, w in enumerate(words) if w in parent]
                if cand:
                    j = cand[0]
                    words[j], indirect = parent[words[j]], True
            pred = str(rng.choice(PREDICATES))
            cid = f"{image_id}_{c}"
            captions.append({"caption_id": cid, "image_id": image_id, "text": f"a {words[0]} {pred} a {words[1]}",
                             "group": "indirect" if indirect else "direct"})
            qv = label_emb[[lid[w] for w in words]].mean(0) + cap_noise * rng.normal(size=dim) / np.sqrt(dim) * 4
            caption_emb.append(unit(qv))
            cap_ids.append(cid)

    ids = rng.permutation(n_images) + 1
    n_tr, n_va = int(split[0] * n_images), int(split[1] * n_images)
    splits = {"seed": seed, "train": sorted(map(int, ids[:n_tr])), "val": sorted(map(int, ids[n_tr:n_tr + n_va])),
              "test": sorted(map(int, ids[n_tr + n_va:]))}

    conceptnet = [(o, "IsA", parent[o]) for o in parent]
    # thêm vài cạnh nhiễu RelatedTo giữa khái niệm để giống dữ liệu thật
    for _ in range(n_concepts):
        a, b = rng.choice(concepts, size=2, replace=False)
        conceptnet.append((str(a), "RelatedTo", str(b)))

    (out / "vg" / "objects.json").write_text(json.dumps(objects_json))
    (out / "vg" / "relationships.json").write_text(json.dumps(rel_json))
    (out / "captions.json").write_text(json.dumps(captions, ensure_ascii=False))
    (out / "splits.json").write_text(json.dumps(splits))
    (out / "conceptnet.tsv").write_text("".join(f"{h}\t{r}\t{t}\n" for h, r, t in conceptnet))
    np.save(out / "features" / "image_emb.npy", np.stack(image_emb).astype(np.float32))
    (out / "features" / "image_ids.json").write_text(json.dumps(list(range(1, n_images + 1))))
    np.save(out / "features" / "caption_emb.npy", np.stack(caption_emb).astype(np.float32))
    (out / "features" / "caption_ids.json").write_text(json.dumps(cap_ids))
    np.save(out / "features" / "label_emb.npy", label_emb)
    (out / "features" / "labels.json").write_text(json.dumps(labels))
    (out / "DATA_CARD.md").write_text("Dữ liệu GIẢ do mmxai.data.fake sinh ra (seed=%d). Chỉ dùng để kiểm tra pipeline.\n" % seed)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sinh dữ liệu giả định dạng VG")
    p.add_argument("--out", default="data/fake_vg")
    p.add_argument("--n-images", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(generate(a.out, n_images=a.n_images, seed=a.seed))
