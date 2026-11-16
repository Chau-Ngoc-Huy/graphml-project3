"""Export a Visual Genome subset as an MMKG in AdaMF-MAT (OpenKE) format  (Level 2, plan A).

Entities  : concepts (object names, vocab from TRAIN images) + images.        names: c:<name>, img:<id>
Relations : "contains" (image -> concept) + the top-P predicates (concept -> concept).
train2id  : (image, contains, concept) for ALL images (object list = input, not label)
            + concept relation triples seen in >= min_triple TRAIN images only.
valid/test: concept relation triples that appear only in val/test images and not in train (new facts).
Embeddings (CLIP): visual = image feature (concept = mean of TRAIN images containing it);
                   textual = text feature (concept name; image = mean of its concepts' text features).
Leak rules: eval-query positive images are forced into TEST, so their relations never reach training.

--hide-eval-objects : for every INDIRECT eval query (s p o) and each of its positive images, the object o is removed from the
                      image's object list (contains edges + textual embedding), simulating a detector that missed o. The CLIP
                      pixel feature is untouched. Presence-only then fails; only relational context can infer o.
--variants          : full | shuffle | drop  (comma separated), all built from ONE CLIP pass:
                      full    = concept relation triples as they are
                      shuffle = same triples but tails permuted (relation structure destroyed, sizes kept)
                      drop    = no concept relation triples (only image--concept edges)
                      dataset names: <name>, <name>_shuf, <name>_drop

Usage (Kaggle, repo cloned to /kaggle/working/AdaMF-MAT):
  python src/export_vg_mmkg.py --scene .../scene_graphs.json --img-dirs D1 D2 --n 8000 --repo /kaggle/working/AdaMF-MAT \
      --eval-queries src/eval_queries.json --hide-eval-objects --variants full,shuffle,drop --name VGH
Local structure test:  python src/export_vg_mmkg.py --scene data/scene_graphs.json --n 2000 --fake --repo /tmp/x --skip-nn ...
"""
import argparse
import json
import os
import random
import shutil
import subprocess
from collections import Counter

import numpy as np

from common import load_images, split_ids
from features import Encoder

SUFFIX = {"full": "", "shuffle": "_shuf", "drop": "_drop"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--img-dirs", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--repo", required=True, help="AdaMF-MAT repo dir; files go to <repo>/benchmarks/<name>, <repo>/embeddings")
    ap.add_argument("--name", default="VG")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-triple", type=int, default=2)
    ap.add_argument("--top-pred", type=int, default=30)
    ap.add_argument("--eval-queries", default=None)
    ap.add_argument("--hide-eval-objects", action="store_true")
    ap.add_argument("--variants", default="full")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--skip-nn", action="store_true", help="skip running the repo's n-n.py (needed on Kaggle)")
    a = ap.parse_args()
    variants = a.variants.split(",")
    assert all(v in SUFFIX for v in variants), variants
    assert not a.hide_eval_objects or a.eval_queries, "--hide-eval-objects needs --eval-queries"

    eq = json.load(open(a.eval_queries)) if a.eval_queries else []
    eval_ids = {i for q in eq for i in q["positives"]}
    images = load_images(a.scene, a.n, a.split_seed)
    if eval_ids:  # make sure the eval images are present even if the random subset missed them
        images.update(load_images(a.scene, None, a.split_seed, keep_ids=eval_ids - set(images)))
    enc = Encoder("fake", images=images) if a.fake else Encoder("clip", img_dirs=a.img_dirs)
    kept, IF = enc.images(sorted(images))  # CLIP image features: computed once, shared by all variants
    images = {i: images[i] for i in kept}
    pos_of = {i: k for k, i in enumerate(kept)}

    sp = split_ids(kept, a.split_seed)
    ev = eval_ids & set(kept)  # force eval positives into test
    sp["train"] = [i for i in sp["train"] if i not in ev]
    sp["val"] = [i for i in sp["val"] if i not in ev]
    sp["test"] = sorted(set(sp["test"]) | ev)
    print({k: len(v) for k, v in sp.items()}, "images;", len(ev), "eval images forced to test")

    # vocab + predicates from TRAIN only (eval images are test, so hiding below cannot change them)
    cnt = Counter(o for i in sp["train"] for o in images[i]["objs"])
    vocab = sorted(o for o, c in cnt.items() if c >= a.min_count)
    cidx = {c: k for k, c in enumerate(vocab)}
    pc = Counter(p for i in sp["train"] for _, p, _ in images[i]["rels"])
    preds = [p for p, _ in pc.most_common(a.top_pred)]
    print(f"{len(vocab)} concepts, {len(preds)} predicates")

    def triples(ids):
        c = Counter()
        for i in ids:
            for s, p, o in images[i]["rels"]:
                if s in cidx and o in cidx and s != o and p in preds:
                    c[(s, p, o)] += 1
        return c

    tr = triples(sp["train"])
    train_rel = sorted(t for t, c in tr.items() if c >= a.min_triple)
    heldout = sorted(t for t in triples(sp["val"] + sp["test"]) if t not in tr)  # computed BEFORE hiding: facts, not inputs
    random.Random(a.split_seed).shuffle(heldout)
    half = len(heldout) // 2
    valid_rel, test_rel = heldout[:half], heldout[half:]
    if not test_rel:  # tiny subsets: the repo's tester needs >= 1 test triple
        test_rel = train_rel[:1]
    print(f"concept triples: train {len(train_rel)}, valid {len(valid_rel)}, test {len(test_rel)}")

    # hidden-object test: remove o from every indirect eval positive's object list
    hidden = {}
    if a.hide_eval_objects:
        for q in eq:
            if q["group"] != "indirect":
                continue
            o = q["triple"][2]
            for i in q["positives"]:
                if i in images and o in images[i]["objs"]:
                    hidden.setdefault(i, set()).add(o)
        for i, cs in hidden.items():
            images[i]["objs"] = images[i]["objs"] - cs
        print(f"hidden objects: {sum(len(v) for v in hidden.values())} (image, concept) pairs in {len(hidden)} images")

    C = len(vocab)
    ents = [f"c:{c.replace(' ', '_')}" for c in vocab] + [f"img:{i}" for i in kept]
    rels = ["contains"] + [p.replace(" ", "_") for p in preds]
    ridx = {p: k + 1 for k, p in enumerate(preds)}
    contains = [(C + pos_of[i], cidx[o], 0) for i in kept for o in sorted(images[i]["objs"]) if o in cidx]
    rel_rows = [(cidx[s], cidx[o], ridx[p]) for s, p, o in train_rel]

    # embeddings (identical for all variants; images that lost an object also lose it in their textual embedding)
    import torch
    T = enc.texts([f"a photo of a {c}" for c in vocab])
    d = T.shape[1]
    vis = np.zeros((len(ents), d), "float32")
    txt = np.zeros((len(ents), d), "float32")
    txt[:C] = T
    sums, ns = np.zeros((C, d), "float32"), np.zeros(C)
    for i in sp["train"]:
        for o in images[i]["objs"]:
            if o in cidx:
                sums[cidx[o]] += IF[pos_of[i]]
                ns[cidx[o]] += 1
    vis[:C] = sums / np.maximum(ns, 1)[:, None]
    for i in kept:
        k = C + pos_of[i]
        vis[k] = IF[pos_of[i]]
        cs = [cidx[o] for o in images[i]["objs"] if o in cidx]
        if cs:
            m = T[cs].mean(0)
            txt[k] = m / (np.linalg.norm(m) + 1e-9)

    def write(bdir, fname, rows):  # OpenKE: first line = count, then "h t r"
        with open(os.path.join(bdir, fname), "w") as f:
            f.write(f"{len(rows)}\n")
            for h, t, r in rows:
                f.write(f"{h} {t} {r}\n")

    for v in variants:
        name = a.name + SUFFIX[v]
        bdir = os.path.join(a.repo, "benchmarks", name)
        os.makedirs(bdir, exist_ok=True)
        os.makedirs(os.path.join(a.repo, "embeddings"), exist_ok=True)
        rows = list(rel_rows)
        if v == "drop":
            rows = []
        elif v == "shuffle":  # keep heads and relation labels, permute tails
            tails = np.random.RandomState(a.split_seed).permutation([t for _, t, _ in rows])
            rows = [(h, int(t2), r) for (h, _, r), t2 in zip(rows, tails)]
        write(bdir, "train2id.txt", contains + rows)
        write(bdir, "valid2id.txt", [(cidx[s], cidx[o], ridx[p]) for s, p, o in valid_rel])
        write(bdir, "test2id.txt", [(cidx[s], cidx[o], ridx[p]) for s, p, o in test_rel])
        for fname, names in (("entity2id.txt", ents), ("relation2id.txt", rels)):
            with open(os.path.join(bdir, fname), "w") as f:
                f.write(f"{len(names)}\n")
                for k, n in enumerate(names):
                    f.write(f"{n} {k}\n")
        emb = os.path.join(a.repo, "embeddings")
        torch.save(torch.tensor(vis), f"{emb}/{name}-visual.pth")
        torch.save(torch.tensor(txt), f"{emb}/{name}-textual.pth")
        meta = {"name": name, "variant": v, "vocab": vocab, "preds": preds, "n_concepts": C, "kept": kept, "split": sp,
                "eval_images": sorted(ev), "dim": int(d), "hidden": {str(i): sorted(cs) for i, cs in hidden.items()}}
        json.dump(meta, open(os.path.join(a.repo, f"{name}_meta.json"), "w"))
        print(f"[{name}] {len(ents)} entities, {len(rels)} relations, {len(contains) + len(rows)} train triples "
              f"({len(contains)} contains, {len(rows)} concept relations)")
        if not a.skip_nn:  # repo's own script writes type_constrain.txt, 1-1.txt ... test2id_all.txt
            shutil.copy(os.path.join(a.repo, "benchmarks", "MKG-W", "n-n.py"), os.path.join(bdir, "n-n.py"))
            subprocess.run(["python", "n-n.py"], cwd=bdir, check=True)
    print("export done ->", a.repo)


if __name__ == "__main__":
    main()
