"""PSG (Panoptic Scene Graph, ECCV 2022) + caption COCO → bundle cùng định dạng bundle Nhiệm vụ 1 của Thắng.

Đầu ra dùng thẳng được bởi mmxai_eval.Protocol, mmxai.data.bundle.load_bundle, các thí nghiệm vg_*/psg_* và UI.

Nguồn (thư mục --raw):
    psg.json                         chú thích gốc OpenPSG (HF HarborYuan/OpenPSG): đối tượng, quan hệ, coco_image_id
    data/{train-*,val-*,test-*}.parquet   bản HF maelic/PSG-coco-format: chỉ dùng để biết ảnh thuộc split nào (+ bytes ảnh
                                     cho tools/psg_extract_clip.py)
    --coco dataset_coco.json         caption COCO (Karpathy) — 5 câu người viết / ảnh, ghép theo coco_image_id
    --conceptnet *.parquet           ConceptNet tiếng Anh (chọn khái niệm mở rộng cho caption)

Ghi (--out):
    data/images.jsonl     {idx, id, coco_id, psg_id, split, objects, relations}   objects xếp theo số lần xuất hiện
    data/captions.jsonl   {idx, img, text, group, concepts}
    data/object_vocab_train.json   TỪ VỰNG KHÁI NIỆM = nhãn ảnh PSG (đếm trên ảnh train) + khái niệm mở rộng
    data/image_labels.json         chỉ các nhãn ảnh PSG;   data/concepts_ext.json   khái niệm mở rộng (+ số caption train)
    split/split.json, mmxai_eval.py (chép từ bundle VG), psg_stats.json

Khái niệm mở rộng: PSG chỉ có 133 lớp (vd. "person"), caption viết "woman", "boy", "puppy"… Ta lấy các cụm 1–3 từ
xuất hiện trong ≥ --min-caption-count caption TRAIN, là đầu kia của một cạnh ConceptNet danh từ–danh từ
(IsA/Synonym/PartOf/HasA/AtLocation, weight ≥ 1) với một nhãn PSG. Chỉ đếm caption train → không rò rỉ test.
Các khái niệm này là đỉnh đồ thị (có vector CLIP); việc nối chúng với nhãn ảnh là việc của cạnh ConceptNet (graph_extra_cn).

Dùng: python -I tools/psg_to_bundle.py --raw data/psg_raw --coco ~/Downloads/raw/dataset_coco.json \
        --conceptnet data/conceptnet/hf_en_simple/train.parquet --out data/psg_bundle
"""

import argparse
import importlib.util
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--raw", default=str(ROOT / "data" / "psg_raw"))
p.add_argument("--coco", required=True)
p.add_argument("--conceptnet", required=True)
p.add_argument("--out", default=str(ROOT / "data" / "psg_bundle"))
p.add_argument("--eval-module", default=str(ROOT / "data" / "vg_bundle" / "mmxai_eval.py"))
p.add_argument("--n-train", type=int, default=15000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--min-images", type=int, default=5, help="nhãn ảnh phải có trong ≥ N ảnh train")
p.add_argument("--min-caption-count", type=int, default=20, help="khái niệm mở rộng phải có trong ≥ N caption train")
p.add_argument("--max-concepts", type=int, default=1000)
a = p.parse_args()

spec = importlib.util.spec_from_file_location("mmxai_eval", a.eval_module)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

raw, out = Path(a.raw), Path(a.out)
for d in ("data", "split", "features"):
    (out / d).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- tên lớp
RENAME = {"sports ball": "ball", "hair drier": "hair dryer", "playingfield": "playing field"}
DROP = {"other", "merged", "stuff"}


def clean_class(c: str) -> str:
    c = RENAME.get(c, c)
    toks = [t for t in c.split("-") if t not in DROP]
    if len(toks) == 2:                       # wall-brick → brick wall, floor-wood → wood floor, window-blind → blind window
        toks = toks[::-1] if toks[0] in {"wall", "floor"} else [" ".join(toks)]
    name = " ".join(toks)
    return ev.norm_phrase(name) or name


psg = json.loads((raw / "psg.json").read_text())
classes = psg["thing_classes"] + psg["stuff_classes"]
cls_name = [clean_class(c) for c in classes]
preds = psg["predicate_classes"]
print("tên lớp đã đổi:", {c: n for c, n in zip(classes, cls_name) if c != n})

# ---------------------------------------------------------------- split theo file parquet
def ids_of(pattern):
    ids = []
    for f in sorted((raw / "data").glob(pattern)):
        ids += [int(x) for x in pq.read_table(f, columns=["image_id"]).column(0).to_pylist()]
    return ids


test_ids, val_ids, train_pool = ids_of("test-*.parquet"), ids_of("val-*.parquet"), ids_of("train-*.parquet")
assert set(test_ids) == {int(x) for x in psg["test_image_ids"]}, "test parquet ≠ test_image_ids của psg.json"
assert not (set(test_ids) & set(val_ids)), "ảnh nằm cả ở val và test"
leak = set(train_pool) & (set(test_ids) | set(val_ids))
if leak:  # bản HF có 1 ảnh vừa ở train vừa ở val → bỏ khỏi train
    print(f"bỏ {len(leak)} ảnh train trùng val/test: {sorted(leak)}")
    train_pool = [i for i in train_pool if i not in leak]
rng = np.random.RandomState(a.seed)
train_ids = sorted(rng.choice(sorted(train_pool), min(a.n_train, len(train_pool)), replace=False).tolist())
split_of = {**{i: "train" for i in train_ids}, **{i: "val" for i in val_ids}, **{i: "test" for i in test_ids}}
print(f"split: train {len(train_ids)} (pool {len(train_pool)}), val {len(val_ids)}, test {len(test_ids)}")

# ---------------------------------------------------------------- ảnh: đối tượng + quan hệ
kp = {im["cocoid"]: im for im in json.loads(Path(a.coco).expanduser().read_text())["images"]}
recs = {int(r["image_id"]): r for r in psg["data"] if int(r["image_id"]) in split_of}
images, captions = [], []
for pid in sorted(split_of, key=lambda i: int(recs[i]["coco_image_id"])):
    r = recs[pid]
    names = [cls_name[o["category_id"]] for o in r["annotations"]]
    cnt = Counter(names)
    rels = {(names[s], preds[k], names[o]) for s, o, k in r["relations"] if names[s] != names[o]}
    cid = int(r["coco_image_id"])
    images.append(dict(idx=len(images), id=f"psg{pid}", coco_id=cid, psg_id=pid, split=split_of[pid],
                       coco_file=r["file_name"], objects=[n for n, _ in cnt.most_common()],
                       relations=[list(t) for t in sorted(rels)]))
    for s in kp[cid]["sentences"]:
        captions.append(dict(idx=len(captions), img=images[-1]["idx"], text=s["raw"].strip()))

# ---------------------------------------------------------------- nhãn ảnh (train)
img_cnt = Counter()
for im in images:
    if im["split"] == "train":
        img_cnt.update(set(im["objects"]))
labels = [n for n, c in img_cnt.most_common() if c >= a.min_images]
lab_set = set(labels)

# ---------------------------------------------------------------- khái niệm mở rộng (ConceptNet ∩ caption train)
NOUN_RELS = {"IsA", "Synonym", "PartOf", "HasA", "AtLocation"}
t0 = time.time()
cand: dict[str, set] = {}
pos: dict[str, Counter] = {}       # loại từ ConceptNet ghi trong URI (/c/en/sit/v) → lọc động/tính từ
for batch in pq.ParquetFile(a.conceptnet).iter_batches(columns=["rel", "arg1", "arg2", "weight"]):
    for rel, a1, a2, w in zip(*(batch.column(i).to_pylist() for i in range(4))):
        for uri in (a1, a2):
            mp = re.match(r"^/c/en/([^/]+)/([nvar])\b", uri)
            if mp:
                pos.setdefault(mp.group(1).replace("_", " "), Counter())[mp.group(2)] += 1
        rel = rel.strip("/").split("/")[-1]
        if rel not in NOUN_RELS or (w is not None and float(w) < 1.0):
            continue
        m1, m2 = re.match(r"^/c/en/([^/]+)", a1), re.match(r"^/c/en/([^/]+)", a2)
        if not (m1 and m2):
            continue
        h, t = ev.norm_phrase(m1.group(1).replace("_", " ")), ev.norm_phrase(m2.group(1).replace("_", " "))
        if not h or not t or h == t:
            continue
        if h in lab_set and t not in lab_set:
            cand.setdefault(t, set()).add(h)
        elif t in lab_set and h not in lab_set:
            cand.setdefault(h, set()).add(t)
print(f"ConceptNet: {len(cand)} khái niệm nối với nhãn PSG ({time.time() - t0:.0f}s)")

train_imgs = {im["idx"] for im in images if im["split"] == "train"}
cap_cnt = Counter()
for c in captions:
    if c["img"] not in train_imgs:
        continue
    seq = [ev.rule_lemma(t) for t in ev.tokenize(c["text"])]
    grams = {" ".join(seq[i:i + n]) for n in (1, 2, 3) for i in range(len(seq) - n + 1)}
    cap_cnt.update(g for g in grams if g in cand)
bad = ev.STOPWORDS | ev.GENERIC_CONCEPTS
NOUN_ING = {"clothing", "painting", "wedding", "frosting", "topping", "bedding", "railing", "icing", "pudding",
            "dumpling", "stuffing", "evening", "morning", "parking lot"}


def nounish(g: str) -> bool:
    """Khái niệm phải là danh từ (đỉnh 'vật' trong đồ thị): bỏ động/tính/trạng từ theo nhãn loại từ của ConceptNet."""
    if g in NOUN_ING:
        return True
    last = g.split()[-1]
    if len(g.split()) == 1 and (last.endswith("ing") or last.endswith("ed")):
        return False
    c = pos.get(g)
    if not c:                 # ConceptNet không ghi loại từ → giữ
        return True
    return c["n"] > c["v"] + c["a"] + c["r"]


dropped_pos = []
ext = []
for g, n in cap_cnt.most_common():
    if n < a.min_caption_count or g in bad or any(t in ev.STOPWORDS for t in g.split()[:1] + g.split()[-1:]):
        continue
    if not nounish(g):
        dropped_pos.append(g)
        continue
    ext.append((g, n))
ext = ext[: a.max_concepts]
print(f"bỏ {len(dropped_pos)} khái niệm không phải danh từ, vd. {dropped_pos[:40]}")
concept_vocab = labels + [g for g, _ in ext]

# ---------------------------------------------------------------- nhóm truy vấn + khái niệm caption
groups, concepts = ev.label_query_groups(images, captions, concept_vocab)
for c, g, cs in zip(captions, groups, concepts):
    c["group"], c["concepts"] = str(g), cs


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


dump_jsonl(out / "data" / "images.jsonl", images)
dump_jsonl(out / "data" / "captions.jsonl", captions)
(out / "data" / "object_vocab_train.json").write_text(json.dumps(concept_vocab, ensure_ascii=False))
(out / "data" / "image_labels.json").write_text(json.dumps(labels, ensure_ascii=False))
(out / "data" / "concepts_ext.json").write_text(json.dumps(
    [{"concept": g, "train_captions": n, "linked_labels": sorted(cand[g])[:10]} for g, n in ext], ensure_ascii=False, indent=0))
split = {s: [im["coco_id"] for im in images if im["split"] == s] for s in ("train", "val", "test")}
split["meta"] = dict(source="PSG (OpenPSG psg.json; HF maelic/PSG-coco-format splits) + COCO captions (Karpathy json)",
                     pool_sizes={"train": len(train_pool), "val": len(val_ids), "test": len(test_ids)},
                     seed_data=a.seed, n_train=len(train_ids), n_val=None, n_test=None,
                     gallery_size=1000, top_k=100, seeds=[0, 1, 2], created=time.strftime("%Y-%m-%d %H:%M:%S"),
                     smoke=False)
(out / "split" / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1))
shutil.copy(a.eval_module, out / "mmxai_eval.py")

grp = Counter(c["group"] for c in captions if images[c["img"]]["split"] == "test")
n_rel = Counter()
for im in images:
    if im["split"] == "train":
        n_rel.update(tuple(r) for r in im["relations"])
stats = dict(images=len(images), captions=len(captions), splits={s: len(v) for s, v in split.items() if s != "meta"},
             image_labels=len(labels), concepts_ext=len(ext), concept_vocab=len(concept_vocab),
             mean_objects=float(np.mean([len(set(im["objects"])) for im in images])),
             mean_relations=float(np.mean([len(im["relations"]) for im in images])),
             train_relation_types=len(n_rel), test_groups=dict(grp),
             top_ext=[g for g, _ in ext[:40]], dropped_labels=sorted(set(cls_name) - lab_set))
(out / "psg_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
print(json.dumps(stats, ensure_ascii=False, indent=1))
