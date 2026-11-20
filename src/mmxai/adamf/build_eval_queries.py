"""Level 2 evaluation query set (Task 2, item 2): >=20 queries, direct vs indirect.

Needs only scene_graphs.json (no images, no GPU, no teammates).
Usage: python src/build_eval_queries.py data/scene_graphs.json data [per_group=12] [seed=0]

direct   query (s,p,o): positives = images whose annotation contains the triple s-p-o.
indirect query (s,p,o): positives = images where s and o are linked ONLY through a 2-hop path
         s -r1-> x -r2-> o (x a third object) and have NO annotated relation between s and o.
         The relation is not stated in the image graph; it must be inferred (the 'child - plays -
         football' case). Indirect positives are CANDIDATES until checked by eye (check column).
Evaluation only: never tune on this set; keep these image ids out of any training you do.
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict

try:  # trong gói mmxai.adamf
    from .common import ACTIONS, load_images
except ImportError:  # chạy phẳng (notebook Kaggle ghi file cạnh nhau, hoặc `python <file>.py`)
    from common import ACTIONS, load_images

VAGUE = {"hand", "hands", "game", "food", "thing", "item", "area", "side", "part", "top", "front", "back"}


def main(path, out, per_group=12, seed=0, min_obj=200, per_pred=3):
    images = load_images(path)
    objc = Counter(o for im in images.values() for o in im["objs"])
    by_name = defaultdict(set)
    for i, im in images.items():
        for o in im["objs"]:
            by_name[o].add(i)

    direct = defaultdict(set)
    for i, im in images.items():
        for s, p, o in im["rels"]:
            if p in ACTIONS and s != o and objc[s] >= min_obj and objc[o] >= min_obj:
                direct[(s, p, o)].add(i)

    def related(im, a, b):  # any annotated relation a<->b
        return any((x == a and z == b) or (x == b and z == a) for x, _, z in im["rels"])

    def two_hop(im, s, o):
        out = {}
        for x, _, z in im["rels"]:
            if x == s and z not in (s, o):
                out.setdefault(z, []).append(z)
        for z in out:
            if any(a == z and c == o for a, _, c in im["rels"]):
                return z
        return None

    rng = random.Random(seed)
    cands = [t for t, v in direct.items() if 5 <= len(v) <= 60]
    rng.shuffle(cands)

    # indirect candidates: triples with a normal action reading but where the pair is often un-annotated
    ind_pool = []
    for (s, p, o) in cands[:600]:
        pos, ex = [], []
        for i in by_name[s] & by_name[o]:
            im = images[i]
            if related(im, s, o):
                continue
            x = two_hop(im, s, o)
            if x:
                pos.append(i)
                ex.append((i, x))
        # ponytail: tiny stoplist of vague objects; a ceiling of 60 keeps queries specific. Hand-check still required.
        if 5 <= len(pos) <= 60 and not {s, o} & VAGUE:
            ind_pool.append(((s, p, o), sorted(pos), ex[:5]))

    used_pred, used_so, queries = Counter(), set(), []

    def take(group, pool, n):
        k = 0
        for item in pool:
            t = item[0]
            if used_pred[t[1]] >= per_pred or (t[0], t[2]) in used_so:
                continue
            used_pred[t[1]] += 1
            used_so.add((t[0], t[2]))
            s, p, o = t
            if group == "direct":
                queries.append({"id": f"d{k}", "group": group, "text": f"{s} {p} {o}", "triple": t,
                                "positives": sorted(direct[t]), "examples": [], "check": ""})
            else:
                queries.append({"id": f"i{k}", "group": group, "text": f"{s} {p} {o}", "triple": t,
                                "positives": item[1], "examples": [{"image": i, "via": x} for i, x in item[2]], "check": ""})
            k += 1
            if k >= n:
                break

    rng.shuffle(ind_pool)
    take("indirect", ind_pool, per_group)
    dpool = [(t, sorted(direct[t])) for t in cands if (t[0], t[2]) not in used_so]
    take("direct", dpool, per_group)

    json.dump(queries, open(f"{out}/eval_queries.json", "w"), indent=1)
    with open(f"{out}/eval_queries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "query", "n_positives", "example_image:via_object", "keep(y/n)", "note"])
        for q in queries:
            ex = "; ".join(f"{e['image']}:{e['via']}" for e in q["examples"])
            w.writerow([q["id"], q["group"], q["text"], len(q["positives"]), ex, "", ""])
    print(Counter(q["group"] for q in queries), "queries saved")
    for q in queries:
        print(f"  {q['id']:>3} {q['group']:8} {q['text']!r:40} pos={len(q['positives'])} {q['examples'][:2]}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 12, int(a[4]) if len(a) > 4 else 0)
