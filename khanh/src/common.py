"""Shared pieces: data loading, fixed split, query building, metrics.

Everything is seeded and image-level so all methods see the same split (spec section 6).
"""
import random
import re
from collections import defaultdict

import ijson
import numpy as np

# ponytail: hand-picked action list; extend after reading data/queries/predicates.txt
ACTIONS = {"holding", "riding", "playing", "playing with", "eating", "carrying", "flying",
           "pulling", "throwing", "hitting", "catching", "kicking", "watching", "reading",
           "driving", "pushing", "feeding", "petting", "sitting on", "standing on", "walking on"}


def norm(s):
    return re.sub(r"^(a|an|the)\s+|\s+(a|an|the)$", "", s.strip().lower())


def obj_name(o):
    n = o.get("names") or [o.get("name", "")]
    return norm(n[0]) if n and n[0] else ""


def load_images(path, n=None, seed=0, keep_ids=None):
    """Stream scene_graphs.json -> {image_id: {"objs": set(names), "rels": set((s,p,o))}}.

    n: random subset size (seeded, hash-based so no full load needed). keep_ids: restrict to these ids.
    """
    p = None if n is None else n / 108077.0  # ponytail: 108077 = size of full VG file
    out = {}
    with open(path, "rb") as f:
        for img in ijson.items(f, "item"):
            iid = img["image_id"]
            if keep_ids is not None and iid not in keep_ids:
                continue
            if p is not None and random.Random(f"{seed}-{iid}").random() >= p:
                continue
            id2n = {o["object_id"]: obj_name(o) for o in img["objects"]}
            objs = {v for v in id2n.values() if v}
            rels = set()
            for r in img["relationships"]:
                s, o, pr = id2n.get(r["subject_id"]), id2n.get(r["object_id"]), norm(r["predicate"])
                if s and o:
                    rels.add((s, pr, o))
            out[iid] = {"objs": objs, "rels": rels}
    return out


def split_ids(ids, seed=0, frac=(0.8, 0.1, 0.1)):
    """Fixed image-level split. Same ids + seed -> same split. Returns dict name -> list."""
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    a, b = int(len(ids) * frac[0]), int(len(ids) * (frac[0] + frac[1]))
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def build_queries(images, ids, min_pos=2, max_pos=50, max_q=None, seed=0):
    """Triple queries restricted to `ids` (one split's gallery).

    Query text "s p o"; positives = images in `ids` whose annotation has that action triple.
    """
    idset = set(ids)
    pos = defaultdict(list)
    for i in ids:
        for t in images[i]["rels"]:
            if t[1] in ACTIONS and t[0] != t[2]:
                pos[t].append(i)
    qs = [{"text": f"{s} {p} {o}", "triple": (s, p, o), "pos": sorted(set(v))}
          for (s, p, o), v in pos.items() if min_pos <= len(set(v)) <= max_pos]
    qs.sort(key=lambda q: q["text"])
    random.Random(seed).shuffle(qs)
    return qs[:max_q] if max_q else qs


def metrics(rank_lists, positives, ks=(1, 5, 10)):
    """rank_lists: list of ranked image-id lists; positives: list of sets. Hit-rate Recall@k, MRR, nDCG@10."""
    r = {f"R@{k}": 0.0 for k in ks}
    mrr = ndcg = 0.0
    for ranked, pos in zip(rank_lists, positives):
        first = next((j for j, i in enumerate(ranked) if i in pos), None)
        if first is not None:
            mrr += 1.0 / (first + 1)
            for k in ks:
                r[f"R@{k}"] += first < k
        dcg = sum(1.0 / np.log2(j + 2) for j, i in enumerate(ranked[:10]) if i in pos)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(pos), 10)))
        ndcg += dcg / idcg if idcg else 0.0
    n = max(len(rank_lists), 1)
    out = {k: v / n for k, v in r.items()}
    out["MRR"], out["nDCG@10"] = mrr / n, ndcg / n
    return out


if __name__ == "__main__":  # smoke check
    m = metrics([[3, 1, 2], [9, 8, 7]], [{1}, {5}])
    assert m["R@1"] == 0 and m["R@5"] == 0.5 and abs(m["MRR"] - 0.25) < 1e-9, m
    s1, s2 = split_ids(range(100), 0), split_ids(range(100), 0)
    assert s1 == s2 and len(s1["train"]) == 80 and not set(s1["train"]) & set(s1["test"])
    print("common ok")
