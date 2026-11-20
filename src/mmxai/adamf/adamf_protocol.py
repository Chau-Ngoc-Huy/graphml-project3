"""AdaMF-MAT as a score_fn for the team's shared protocol (Thang's mmxai_eval.Protocol).   [Khanh, Level 2]

Pipeline (same split, same 1K-gallery folds, same CLIP top-K as every other method):
  caption -> concepts (Thang's ConceptLinker, already in captions.jsonl["concepts"])
  graph_score(caption, candidate image) = sum over the caption's concepts c of AdaMF-MAT plausibility of (image, contains, c)
  final score = alpha*z(cosine) + (1-alpha)*z(graph_score), alpha chosen on VAL  (mmxai_eval.fuse / tune_alpha)

MMKG that AdaMF-MAT is trained on (built from the Protocol, no leakage):
  entities  : concepts = P.object_vocab("train")  +  images (sampled train images + ALL val/test images)
  relations : "contains" + top predicates
  train2id  : (image, contains, concept) for every included image (object list = input, not label)
              + concept->concept relation triples seen in >= min_triple TRAIN images only
  valid/test: concept relation triples seen only in val/test images (new facts), used by the repo's own KGC test
  embeddings: visual = CLIP image feature (concept = mean over TRAIN images containing it)
              textual = CLIP text feature of the concept name (image = mean of its concepts' text features)
  variants  : full | shuffle (tails permuted) | drop (no concept relations) for the relation-structure ablation.

Usage (Kaggle, after Thang's notebook produced WORK=/kaggle/working/task1 and the repo is cloned):
  python adamf_protocol.py build --work WORK --repo REPO --name COCO_full --variant full
  (train with the repo: python run_adamf_mat.py -dataset=COCO_full ... -save=./checkpoint/COCO_full-full)
  python adamf_protocol.py eval  --work WORK --repo REPO --name COCO_full --ckpt checkpoint/COCO_full-full --out results.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

try:
    import mmxai_eval  # noqa: F401  (Kaggle notebook writes it next to this file)
except ImportError:  # trong repo thống nhất, module chung nằm ở src/mmxai_eval.py (src/mmxai/adamf → ../..)
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SUFFIX = {"full": "", "shuffle": "_shuf", "drop": "_drop"}


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# ----------------------------------------------------------------------------------------- build the MMKG
def build_mmkg(P, repo, name, text_encode, variant="full", n_train_images=6000, min_triple=2, top_pred=30,
               seed=0, run_nn=True):
    """Write benchmarks/<name>/*.txt, embeddings/<name>-{visual,textual}.pth and <repo>/<name>_meta.json."""
    import torch
    vocab = P.object_vocab("train")
    cidx = {c: k for k, c in enumerate(vocab)}
    C = len(vocab)

    pool_tr, pool_va, pool_te = P.pool("train"), P.pool("val"), P.pool("test")
    sub = sorted(np.random.RandomState(seed).choice(pool_tr, min(n_train_images, len(pool_tr)), replace=False).tolist())
    kg_imgs = sub + sorted(pool_va) + sorted(pool_te)  # entity order after the concepts
    row_of = {ii: C + k for k, ii in enumerate(kg_imgs)}

    from collections import Counter

    def rel_counts(idxs):
        cnt = Counter()
        for ii in idxs:
            seen = {(s, p, o) for s, p, o in P.images[ii]["relations"] if s in cidx and o in cidx and s != o}
            cnt.update(seen)
        return cnt

    tr = rel_counts(pool_tr)  # ALL train images feed the concept relations (cheap), only `sub` get contains edges
    preds = [p for p, _ in Counter(p for (_, p, _), c in tr.items() for _ in range(c)).most_common(top_pred)]
    pidx = {p: k + 1 for k, p in enumerate(preds)}
    train_rel = sorted(t for t, c in tr.items() if c >= min_triple and t[1] in pidx)
    held = sorted(t for t in rel_counts(pool_va + pool_te) if t not in tr and t[1] in pidx)
    np.random.RandomState(seed).shuffle(held)
    valid_rel, test_rel = held[: len(held) // 2], held[len(held) // 2:]
    if not test_rel:
        test_rel = train_rel[:1]

    contains = [(row_of[ii], cidx[o], 0) for ii in kg_imgs for o in P.images[ii]["objects"] if o in cidx]
    rows = [(cidx[s], cidx[o], pidx[p]) for s, p, o in train_rel]
    if variant == "drop":
        rows = []
    elif variant == "shuffle":
        tails = np.random.RandomState(seed).permutation([t for _, t, _ in rows])
        rows = [(h, int(t2), r) for (h, _, r), t2 in zip(rows, tails)]

    vname = name
    bdir = os.path.join(repo, "benchmarks", vname)
    os.makedirs(bdir, exist_ok=True)
    os.makedirs(os.path.join(repo, "embeddings"), exist_ok=True)

    def write(fname, rr):
        with open(os.path.join(bdir, fname), "w") as f:
            f.write(f"{len(rr)}\n")
            for h, t, r in rr:
                f.write(f"{h} {t} {r}\n")

    write("train2id.txt", contains + rows)
    write("valid2id.txt", [(cidx[s], cidx[o], pidx[p]) for s, p, o in valid_rel])
    write("test2id.txt", [(cidx[s], cidx[o], pidx[p]) for s, p, o in test_rel])
    ents = [f"c:{c.replace(' ', '_')}" for c in vocab] + [f"img:{ii}" for ii in kg_imgs]
    rels = ["contains"] + [p.replace(" ", "_") for p in preds]
    for fname, names in (("entity2id.txt", ents), ("relation2id.txt", rels)):
        with open(os.path.join(bdir, fname), "w") as f:
            f.write(f"{len(names)}\n")
            for k, n in enumerate(names):
                f.write(f"{n.replace(' ', '_')} {k}\n")

    # embeddings
    T = _unit(np.asarray(text_encode([f"a photo of a {c}" for c in vocab]), np.float32))
    d = T.shape[1]
    assert d == P.img.shape[1], f"text dim {d} != image dim {P.img.shape[1]}"
    vis = np.zeros((len(ents), d), "float32")
    txt = np.zeros((len(ents), d), "float32")
    txt[:C] = T
    sums, ns = np.zeros((C, d), "float32"), np.zeros(C)
    for ii in pool_tr:
        for o in P.images[ii]["objects"]:
            if o in cidx:
                sums[cidx[o]] += P.img[ii]
                ns[cidx[o]] += 1
    vis[:C] = sums / np.maximum(ns, 1)[:, None]
    for ii in kg_imgs:
        k = row_of[ii]
        vis[k] = P.img[ii]
        cs = [cidx[o] for o in P.images[ii]["objects"] if o in cidx]
        if cs:
            txt[k] = _unit(T[cs].mean(0))
    emb = os.path.join(repo, "embeddings")
    torch.save(torch.tensor(vis), f"{emb}/{vname}-visual.pth")
    torch.save(torch.tensor(txt), f"{emb}/{vname}-textual.pth")
    meta = dict(name=vname, variant=variant, vocab=vocab, preds=preds, n_concepts=C, kg_images=kg_imgs, dim=int(d))
    json.dump(meta, open(os.path.join(repo, f"{vname}_meta.json"), "w"))
    print(f"[{vname}] {len(ents)} entities ({C} concepts, {len(kg_imgs)} images), {len(rels)} relations, "
          f"{len(contains) + len(rows)} train triples ({len(contains)} contains, {len(rows)} concept relations), "
          f"heldout valid/test {len(valid_rel)}/{len(test_rel)}")
    if run_nn:  # the repo's own script writes type_constrain.txt, 1-1.txt ... test2id_all.txt
        shutil.copy(os.path.join(repo, "benchmarks", "MKG-W", "n-n.py"), os.path.join(bdir, "n-n.py"))
        subprocess.run(["python", "n-n.py"], cwd=bdir, check=True)
    return meta


# ----------------------------------------------------------------------------------------- the scorer
class AdamfScorer:
    """graph_score(c, P) -> (Nq, K): AdaMF-MAT plausibility of (candidate image, contains, concept) summed over the caption's concepts."""

    def __init__(self, P, repo, name, ckpt, dim=200, margin=12.0, device=None):
        import torch
        sys.path.insert(0, repo)
        from mmkgc.module.model import AdvMixRotatE  # repo code (license not checked: do not redistribute)
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = json.load(open(os.path.join(repo, f"{name}_meta.json")))
        self.cidx = {c: k for k, c in enumerate(meta["vocab"])}
        self.C = meta["n_concepts"]
        self.row = np.full(len(P.images), -1, np.int64)  # protocol image idx -> entity id
        for k, ii in enumerate(meta["kg_images"]):
            self.row[ii] = self.C + k
        img_emb = torch.load(os.path.join(repo, "embeddings", f"{name}-visual.pth"))
        txt_emb = torch.load(os.path.join(repo, "embeddings", f"{name}-textual.pth"))
        ent_tot = int(open(os.path.join(repo, "benchmarks", name, "entity2id.txt")).readline())
        rel_tot = int(open(os.path.join(repo, "benchmarks", name, "relation2id.txt")).readline())
        self.model = AdvMixRotatE(ent_tot=ent_tot, rel_tot=rel_tot, dim=dim, margin=margin, epsilon=2.0,
                                  img_emb=img_emb, text_emb=txt_emb)
        if ckpt:  # tests may pass None to keep random weights
            self.model.load_state_dict(torch.load(os.path.join(repo, ckpt), map_location="cpu"))
        self.model = self.model.to(self.dev).eval()
        self.imgs_ent = torch.arange(self.C, ent_tot, device=self.dev)
        self._vec = {}

    def _concept_vec(self, c):
        """plausibility of (img, contains, c) for EVERY image entity, cached per concept."""
        if c not in self._vec:
            t, h = self.torch.full_like(self.imgs_ent, self.cidx[c]), self.imgs_ent
            with self.torch.no_grad():
                s = self.model({"batch_h": h, "batch_t": t, "batch_r": self.torch.zeros_like(h), "mode": "normal"})
            self._vec[c] = s.float().cpu().numpy()
        return self._vec[c]

    def __call__(self, c, P):
        out = np.zeros(c["sim"].shape, np.float64)
        rows = self.row[c["cand"]] - self.C  # (Nq,K) position in the image-entity block, negative = not in the MMKG
        for r, q in enumerate(c["q_cap"]):
            cs = [x for x in P.captions[int(q)].get("concepts", []) if x in self.cidx]
            for x in cs:
                v = self._concept_vec(x)
                ok = rows[r] >= 0
                out[r, ok] += v[rows[r][ok]]
        return out


def presence_score(c, P):
    """Control with NO relational knowledge: share of the caption's concepts in the candidate's object list (Thang's lexical rerank)."""
    obj = [set(im["objects"]) for im in P.images]
    out = np.zeros(c["sim"].shape, np.float64)
    for r, q in enumerate(c["q_cap"]):
        cs = set(P.captions[int(q)].get("concepts", []))
        if cs:
            out[r] = [len(cs & obj[ii]) / len(cs) for ii in c["cand"][r]]
    return out


# ----------------------------------------------------------------------------------------- evaluate on the shared protocol
def evaluate_all(P, scorer, seeds=(0, 1, 2), name="CLIP+AdaMF-MAT", adapter=None):
    """adapter (tùy chọn, `adapter.Adapter` đã huấn luyện): thêm CLIP+Adapter, +khớp tên, +AdaMF-MAT. Mốc đối chứng đúng cho AdaMF-MAT
    khi có adapter là CLIP+Adapter (không phải CLIP thô), nên bootstrap của các phương pháp có adapter lấy mốc đó."""
    from mmxai_eval import fuse, paired_bootstrap, summary_table, tune_alpha
    base = P.evaluate(lambda c, p: c["sim"], "CLIP", "test", seeds)
    c_val = P.candidates("val", None)
    res, rows = {"CLIP": base}, [base]
    for label, fn in (("CLIP+presence (no relations)", presence_score), (name, scorer)):
        alpha, curve = tune_alpha(c_val, fn(c_val, P))  # alpha on VAL only
        r = P.evaluate(lambda c, p, fn=fn, a=alpha: fuse(c["sim"], fn(c, p), a), f"{label} (a={alpha})", "test", seeds)
        r["alpha"], r["alpha_curve"] = alpha, curve
        r["bootstrap_vs_clip"] = paired_bootstrap(base["full"]["ranks"], r["full"]["ranks"])
        res[label] = r
        rows.append(r)
    if adapter is not None:
        try:
            from .adapter import adapted_sim
        except ImportError:
            from adapter import adapted_sim
        asim = lambda c: adapted_sim(adapter, P, c)
        ad = P.evaluate(lambda c, p: asim(c), "CLIP+Adapter", "test", seeds)
        ad["bootstrap_vs_clip"] = paired_bootstrap(base["full"]["ranks"], ad["full"]["ranks"])
        res["CLIP+Adapter"] = ad
        rows.append(ad)
        c_val_a = dict(c_val, sim=asim(c_val))  # val với cosine của adapter: alpha chọn trên VAL
        for label, fn in (("CLIP+Adapter+presence (no relations)", presence_score), (f"CLIP+Adapter+{name.replace('CLIP+', '')}", scorer)):
            alpha, curve = tune_alpha(c_val_a, fn(c_val, P))
            r = P.evaluate(lambda c, p, fn=fn, a=alpha: fuse(asim(c), fn(c, p), a), f"{label} (a={alpha})", "test", seeds)
            r["alpha"], r["alpha_curve"] = alpha, curve
            r["bootstrap_vs_clip"] = paired_bootstrap(base["full"]["ranks"], r["full"]["ranks"])
            r["bootstrap_vs_adapter"] = paired_bootstrap(ad["full"]["ranks"], r["full"]["ranks"])
            res[label] = r
            rows.append(r)
    print(summary_table(rows, title="text->image, test, 1K galleries, mean +- std over folds"))
    for g in base["by_group"]:
        print(summary_table([dict(name=f"{r['name']} - {g}", split="test", mean_std=r["by_group"][g]) for r in rows],
                            title=f"query group: {g}"))
    for k, r in res.items():
        if k != "CLIP":
            print(k, "bootstrap vs CLIP (full test pool):", json.dumps(r["bootstrap_vs_clip"]))
        if "bootstrap_vs_adapter" in r:
            print(k, "bootstrap vs CLIP+Adapter (full test pool):", json.dumps(r["bootstrap_vs_adapter"]))
    return res


def _load_protocol(work):
    from mmxai_eval import Protocol
    return Protocol.load(work)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "eval"])
    ap.add_argument("--work", required=True, help="Thang's WORK dir (data/, split/, features/)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--name", default="COCO_full")
    ap.add_argument("--variant", default="full", choices=list(SUFFIX))
    ap.add_argument("--n-train-images", type=int, default=6000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--margin", type=float, default=12.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results_protocol.json")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--adapter", action="store_true", help="eval: huấn luyện adapter (train/val) và thêm CLIP+Adapter(+AdaMF-MAT)")
    a = ap.parse_args()
    P = _load_protocol(a.work)
    if a.mode == "build":
        try:
            from .features import Encoder
        except ImportError:
            from features import Encoder
        enc = Encoder("clip")  # same OpenCLIP ViT-B-32 laion2b as the shared features
        build_mmkg(P, a.repo, a.name, enc.texts, a.variant, a.n_train_images, run_nn=not a.skip_nn)
    else:
        sc = AdamfScorer(P, a.repo, a.name, a.ckpt, a.dim, a.margin)
        adapter = None
        if a.adapter:
            try:
                from .adapter import train_adapter
            except ImportError:
                from adapter import train_adapter
            adapter = train_adapter(P, seed=0)
        res = evaluate_all(P, sc, a.seeds, adapter=adapter)
        from mmxai_eval import save_json
        save_json(res, a.out)
        print("saved", a.out)


if __name__ == "__main__":
    main()
