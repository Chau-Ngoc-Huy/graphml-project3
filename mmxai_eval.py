"""mmxai_eval — module đánh giá dùng chung cho Project 3 (Nhiệm vụ 1: truy vấn & đánh giá chuẩn — Thắng).

Mọi phương pháp (CLIP thuần, CLIP + Graph của Huy, phương pháp mới của Khánh) đều đi qua CÙNG MỘT
giao thức: cùng file split, cùng gallery / fold theo seed, cùng top-K ứng viên CLIP, cùng hàm metric.
Một phương pháp chỉ cần cung cấp hàm  ``score_fn(cands, protocol) -> np.ndarray (Nq, K)``:
điểm cuối cùng của từng ứng viên trong top-K CLIP (điểm cao = xếp trước).

Ví dụ:
    P = Protocol.load("work_task1")                          # split + đặc trưng CLIP + nhóm truy vấn
    clip = lambda c, p: c["sim"]                              # baseline CLIP: điểm = cosine
    res = P.evaluate(clip, name="CLIP", split="test", seeds=[0, 1, 2])
    print(summary_table([res]))

    def clip_graph(c, p):                                     # CLIP + Graph (Huy)
        sg = my_graph_score(c["q_cap"], c["cand"])            # (Nq, K) điểm đồ thị
        return fuse(c["sim"], sg, alpha=0.7)                  # α·z(cos) + (1−α)·z(s_G)
    res2 = P.evaluate(clip_graph, name="CLIP+GAT", split="test", seeds=[0, 1, 2])
    print(paired_bootstrap(res["per_seed"][0]["ranks"], res2["per_seed"][0]["ranks"]))

Thành phần:
    * chuẩn hoá văn bản không phụ thuộc thư viện ngoài (``norm_phrase``, ``ConceptLinker``)
    * metric xếp hạng: Recall@k, MRR, MedR, nDCG@10 (1 ảnh đúng) và bản nhiều ảnh liên quan
    * ``Protocol``: gallery / fold theo seed, top-K ứng viên (FAISS nếu có), đánh giá text→image và
      image→text, tách nhóm truy vấn (trực tiếp / gián tiếp)
    * ``fuse`` / ``tune_alpha``: kết hợp điểm α·z(cos) + (1−α)·z(s_G), chọn α trên validation
    * ``aggregate`` / ``summary_table`` / ``paired_bootstrap``: mean ± std theo seed, bảng Markdown, kiểm định
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

__all__ = [
    "rank_metrics", "multi_relevant_metrics", "ranks_from_scores", "rerank_ranks", "zscore", "fuse",
    "tune_alpha", "aggregate", "summary_table", "paired_bootstrap", "Protocol", "ConceptLinker",
    "norm_phrase", "label_query_groups", "topk", "log", "save_json", "load_json",
]

METRICS = ["R@1", "R@5", "R@10", "MRR", "nDCG@10"]
_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=default)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================== text normalisation
STOPWORDS = set("""
a an the and or but if then than so of at by for with about against between into through during before
after above below to from up down in out on off over under again further once here there when where why how
all any both each few more most other some such no nor not only own same too very s t can will just don
should now is are was were be been being am have has had having do does did doing would could ought i me
my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself
it its itself they them their theirs themselves what which who whom this that these those while as until
one two three four five six seven eight nine ten several many lot lots something someone somebody thing
things another next near around along front top side back also very image picture photo view shot
showing shows show close closeup
""".split())
_TOKEN_RE = re.compile(r"[a-z]+")
_IRREG = {"men": "man", "women": "woman", "children": "child", "people": "people", "feet": "foot",
          "teeth": "tooth", "mice": "mouse", "geese": "goose", "knives": "knife", "leaves": "leaf",
          "shelves": "shelf", "wolves": "wolf", "glasses": "glasses", "jeans": "jeans", "pants": "pants",
          "shorts": "shorts", "scissors": "scissors", "bus": "bus", "grass": "grass", "dress": "dress",
          "skis": "ski", "fries": "fries", "sunglasses": "sunglasses", "clothes": "clothes", "tennis": "tennis",
          "news": "news", "lens": "lens", "gas": "gas", "canvas": "canvas", "atlas": "atlas", "chess": "chess"}


def rule_lemma(w):
    """Lemmatiser luật đơn giản (không cần NLTK): 'dogs'->'dog', 'buses'->'bus', 'riding'->'rid' (chấp nhận)."""
    if w in _IRREG or w.endswith(("us", "is", "ss")):
        return _IRREG.get(w, w)
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxzh":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def norm_phrase(text, max_words=3):
    """'Tennis Rackets' -> 'tennis racket'. Trả về None nếu rỗng hoặc quá dài."""
    toks = [rule_lemma(t) for t in tokenize(str(text).replace("_", " "))]
    toks = [t for t in toks if t and t not in STOPWORDS] or toks
    if not toks or len(toks) > max_words:
        return None
    return " ".join(toks)


class ConceptLinker:
    """Khớp tham lam n-gram (dài nhất trước) của truy vấn với từ vựng khái niệm (tên đối tượng VG)."""

    def __init__(self, vocab, max_ngram=3):
        self.vocab = set(vocab)
        self.max_ngram = max_ngram

    def link(self, text):
        seq = [rule_lemma(t) for t in tokenize(text)]
        out, i = [], 0
        while i < len(seq):
            hit = None
            for n in range(min(self.max_ngram, len(seq) - i), 0, -1):
                gram = seq[i:i + n]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                name = " ".join(gram)
                if name in self.vocab:
                    hit = (name, n)
                    break
            if hit is None:
                i += 1
                continue
            if hit[0] not in out:
                out.append(hit[0])
            i += hit[1]
        return out


# Màu sắc / thuộc tính / danh từ chung chung xuất hiện như "đối tượng" trong VG nhưng không phải thực thể thị giác —
# bỏ khỏi việc GÁN NHÓM truy vấn (đồ thị của Huy vẫn có thể dùng chúng).
GENERIC_CONCEPTS = set("""
white black blue green brown yellow orange red pink gray grey purple gold silver tan dark light color colour
group bunch area piece background foreground day night air body row part side top front middle number couple pair
lot variety kind type set picture image photo scene view stand spot line edge corner shape pattern design
""".split())


def label_query_groups(images, captions, vocab, generic=GENERIC_CONCEPTS):
    """Nhóm truy vấn theo scene graph của ảnh đúng (chỉ để BÁO CÁO, mô hình không dùng).

    direct     : mọi khái niệm liên kết được từ caption đều có trong objects của ảnh đúng
    indirect   : có ít nhất 1 khái niệm KHÔNG có trong ảnh (chỉ nối được qua quan hệ / đồng nghĩa)
    no_concept : caption không liên kết được khái niệm nào trong từ vựng
    """
    linker = ConceptLinker(vocab)
    groups, concepts = [], []
    for c in captions:
        cs = [x for x in linker.link(c["text"]) if x not in generic]
        objs = set(images[c["img"]]["objects"])
        concepts.append(cs)
        if not cs:
            groups.append("no_concept")
        elif all(x in objs for x in cs):
            groups.append("direct")
        else:
            groups.append("indirect")
    return np.array(groups), concepts


# =============================================================================== metrics
def rank_metrics(ranks, ks=(1, 5, 10)):
    """Metric khi mỗi truy vấn có đúng 1 ảnh đúng. ``ranks``: hạng (1-based) của ảnh đúng."""
    r = np.asarray(ranks, dtype=np.float64)
    m = {f"R@{k}": float((r <= k).mean() * 100) for k in ks}
    m["MRR"] = float((1.0 / r).mean() * 100)
    m["nDCG@10"] = float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0).mean() * 100)
    m["MedR"] = float(np.median(r))
    m["n"] = int(len(r))
    return m


def multi_relevant_metrics(order, relevant, ks=(1, 5, 10)):
    """Metric khi một truy vấn có NHIỀU ảnh liên quan (bộ ≥20 truy vấn của Khánh).

    ``order``: danh sách id ảnh đã xếp hạng; ``relevant``: tập id ảnh liên quan.
    """
    hits = np.array([i in relevant for i in order], dtype=float)
    first = int(np.argmax(hits)) + 1 if hits.any() else len(order) + 1
    m = {f"R@{k}": float(hits[:k].any() * 100) for k in ks}
    m["MRR"] = 100.0 / first
    m["P@10"] = float(hits[:10].mean() * 100)
    disc = 1.0 / np.log2(np.arange(2, 12))
    ideal = disc[: min(len(relevant), 10)].sum()
    m["nDCG@10"] = float((hits[:10] * disc[: len(hits[:10])]).sum() / ideal * 100) if ideal else 0.0
    return m


def ranks_from_scores(scores, gt_col):
    """Hạng (1-based) của cột gt_col trong từng hàng điểm. Hoà điểm: xếp ảnh đúng trước (đếm 'lớn hơn hẳn')."""
    gt = scores[np.arange(len(scores)), gt_col][:, None]
    return (scores > gt).sum(1) + 1


def rerank_ranks(cands, scores):
    """Hạng ảnh đúng sau khi xếp lại top-K bằng ``scores`` (Nq, K); ảnh đúng ngoài top-K giữ hạng CLIP."""
    pos = cands["pos"]
    order = np.argsort(-scores, axis=1, kind="stable")
    new_pos = np.argmax(order == pos[:, None], axis=1)
    return np.where(pos >= 0, new_pos + 1, cands["gt_rank"])


def zscore(x):
    mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
    return np.where(sd > 1e-8, (x - mu) / np.maximum(sd, 1e-8), 0.0)


def fuse(sim, graph_score, alpha):
    """score = α·z(cos) + (1−α)·z(s_G), chuẩn hoá z-score trong top-K của từng truy vấn."""
    return alpha * zscore(np.asarray(sim, np.float64)) + (1 - alpha) * zscore(np.asarray(graph_score, np.float64))


def tune_alpha(cands, graph_score, step=0.05, key="MRR"):
    """Quét α ∈ [0,1] trên một split (PHẢI là validation). Trả về (α tốt nhất, đường cong)."""
    curve = []
    for a in np.round(np.arange(0, 1 + 1e-9, step), 4):
        m = rank_metrics(rerank_ranks(cands, fuse(cands["sim"], graph_score, a)))
        curve.append(dict(alpha=float(a), **m))
    best = max(curve, key=lambda d: (d[key], d["alpha"]))
    return best["alpha"], curve


def aggregate(list_of_metrics, keys=None):
    keys = keys or [k for k in list_of_metrics[0] if isinstance(list_of_metrics[0][k], (int, float)) and k != "n"]
    out = {}
    for k in keys:
        v = np.array([m[k] for m in list_of_metrics], dtype=float)
        out[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=0)), values=v.tolist())
    out["n"] = int(np.mean([m.get("n", 0) for m in list_of_metrics]))
    return out


def _cell(v, digits=2):
    if isinstance(v, dict):
        return f"{v['mean']:.{digits}f} ± {v['std']:.{digits}f}"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def summary_table(results, metrics=METRICS, field="mean_std", title=None):
    """Bảng Markdown: một dòng / kết quả của ``Protocol.evaluate``."""
    lines = ([f"**{title}**", ""] if title else []) + [
        "| phương pháp | split | n truy vấn | " + " | ".join(metrics) + " |",
        "|---|---|---|" + "---|" * len(metrics)]
    for r in results:
        src = r[field]
        lines.append(f"| {r['name']} | {r['split']} | {src.get('n', '')} | " + " | ".join(_cell(src[m]) for m in metrics) + " |")
    return "\n".join(lines)


def paired_bootstrap(ranks_a, ranks_b, n_boot=2000, seed=0):
    """So sánh 2 phương pháp trên CÙNG các truy vấn (ghép cặp) bằng bootstrap trên hiệu MRR và R@1.

    Trả về hiệu (b − a), khoảng tin cậy 95 % và p-value hai phía ≈ 2·min(P(Δ≤0), P(Δ≥0)).
    """
    a, b = np.asarray(ranks_a, float), np.asarray(ranks_b, float)
    assert a.shape == b.shape, "hai phương pháp phải chạy trên cùng danh sách truy vấn"
    rng = np.random.RandomState(seed)
    out = {}
    for name, fa in (("MRR", lambda r: 100 / r), ("R@1", lambda r: 100.0 * (r <= 1)), ("R@10", lambda r: 100.0 * (r <= 10))):
        d = fa(b) - fa(a)
        idx = rng.randint(0, len(d), size=(n_boot, len(d)))
        boots = d[idx].mean(1)
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        out[name] = dict(delta=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         p_value=float(min(1.0, p)))
    return out


# =============================================================================== similarity / top-K
def _device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:  # noqa: BLE001
        return None


def topk(q, gal, K, use_faiss=True):
    """Top-K theo tích vô hướng (cosine nếu đã chuẩn hoá). Trả về (sim (Nq,K), idx (Nq,K), sims đầy đủ hoặc None)."""
    q = np.ascontiguousarray(q, dtype=np.float32)
    gal = np.ascontiguousarray(gal, dtype=np.float32)
    K = min(K, len(gal))
    if use_faiss:
        try:
            import faiss
            index = faiss.IndexFlatIP(gal.shape[1])
            index.add(gal)
            s, i = index.search(q, K)
            return s, i, index
        except Exception:  # noqa: BLE001
            pass
    sims = q @ gal.T
    i = np.argpartition(-sims, K - 1, axis=1)[:, :K]
    s = np.take_along_axis(sims, i, 1)
    o = np.argsort(-s, axis=1, kind="stable")
    return np.take_along_axis(s, o, 1), np.take_along_axis(i, o, 1), None


def full_sims(q, gal, batch=4096):
    """Ma trận cosine đầy đủ (Nq, Ng), dùng GPU nếu có."""
    dev = _device()
    q = np.asarray(q, np.float32)
    gal = np.asarray(gal, np.float32)
    if dev is not None and dev.type == "cuda":
        import torch
        g = torch.tensor(gal, device=dev)
        out = [(torch.tensor(q[b:b + batch], device=dev) @ g.T).cpu().numpy() for b in range(0, len(q), batch)]
        return np.concatenate(out)
    return q @ gal.T


# =============================================================================== protocol
class Protocol:
    """Giao thức đánh giá chung.

    Dữ liệu (do notebook Nhiệm vụ 1 tạo ra trong ``work_dir``):
        data/images.jsonl   : mỗi dòng 1 ảnh {idx, id, vg_id, coco_id, url, split, objects, relations}
        data/captions.jsonl : mỗi dòng 1 caption {idx, img, text, group}
        split/split.json    : {"train": [...coco_id], "val": [...], "test": [...], "meta": {...}}
        features/clip_<model>.npz : img (Ni,D) float16, cap (Nc,D) float16, cap_img (Nc,), ok (Ni,) bool

    Seed:  CLIP đóng băng là tất định, nên ``seed`` điều khiển (i) fold gallery ``gallery_size`` ảnh lấy
    mẫu từ pool test/val, (ii) thứ tự gallery train. Mô hình học (GNN) dùng cùng seed để khởi tạo.
    ``seed=None`` → dùng toàn bộ pool (kết quả tất định, dùng làm số chính).
    """

    def __init__(self, images, captions, split, feats, gallery_size=1000, K=100, seed_data=0):
        self.images, self.captions, self.split = images, captions, split
        self.img = np.asarray(feats["img"], np.float32)
        self.cap = np.asarray(feats["cap"], np.float32)
        self.cap_img = np.asarray(feats["cap_img"])
        self.ok = np.asarray(feats["ok"], bool)
        self.G, self.K, self.seed_data = gallery_size, K, seed_data
        self.caps_of = {}
        for ci, ii in enumerate(self.cap_img):
            self.caps_of.setdefault(int(ii), []).append(ci)
        self.groups = np.array([c.get("group", "all") for c in captions])
        self._cache = {}

    # ---- loading
    @classmethod
    def load(cls, work_dir, feature_file=None, **kw):
        work = Path(work_dir)
        images = [json.loads(l) for l in open(work / "data" / "images.jsonl", encoding="utf-8")]
        captions = [json.loads(l) for l in open(work / "data" / "captions.jsonl", encoding="utf-8")]
        split = load_json(work / "split" / "split.json")
        feature_file = feature_file or sorted((work / "features").glob("clip_*.npz"))[0]
        feats = np.load(feature_file, allow_pickle=False)
        meta = split.get("meta", {})
        kw.setdefault("gallery_size", meta.get("gallery_size", 1000))
        kw.setdefault("K", meta.get("top_k", 100))
        kw.setdefault("seed_data", meta.get("seed_data", 0))
        log(f"Protocol: {len(images)} ảnh, {len(captions)} caption, features {Path(feature_file).name} "
            f"{feats['img'].shape}, gallery={kw['gallery_size']}, K={kw['K']}")
        return cls(images, captions, split, feats, **kw)

    # ---- galleries
    def pool(self, split):
        return [im["idx"] for im in self.images if im["split"] == split and self.ok[im["idx"]]]

    def gallery(self, split, seed=None):
        pool = self.pool(split)
        if seed is None or len(pool) <= self.G:
            return sorted(pool)
        rng = np.random.RandomState(seed)
        return sorted(rng.choice(pool, self.G, replace=False).tolist())

    def train_galleries(self, seed=None):
        """Train chia thành các gallery ``gallery_size`` ảnh (giống test) để lấy hard negative."""
        pool = self.pool("train")
        rng = np.random.RandomState(self.seed_data if seed is None else seed)
        rng.shuffle(pool)
        return [pool[i:i + self.G] for i in range(0, len(pool), self.G)]

    # ---- candidates
    def candidates(self, split="test", seed=None, galleries=None, ensure_positive=False, captions_per_image=None):
        """Top-K ứng viên CLIP cho mọi caption của ảnh trong gallery (định dạng chung cho mọi phương pháp).

        Trả về dict: q_cap (Nq,) idx caption, q_img (Nq,) idx ảnh đúng, cand (Nq,K) idx ảnh ứng viên,
        sim (Nq,K) cosine, pos (Nq,) cột của ảnh đúng trong cand (−1 nếu ngoài top-K),
        gt_rank (Nq,) hạng CLIP của ảnh đúng trong toàn gallery, gallery (list idx), group (Nq,).
        """
        key = (split, seed, galleries is None, ensure_positive, captions_per_image)
        if key in self._cache:
            return self._cache[key]
        if galleries is None:
            galleries = self.train_galleries(seed) if split == "train" else [self.gallery(split, seed)]
        rng = np.random.RandomState(0 if seed is None else seed)
        out = dict(q_cap=[], q_img=[], cand=[], sim=[], pos=[], gt_rank=[])
        for gal in galleries:
            gal = list(gal)
            col = {g: j for j, g in enumerate(gal)}
            q_cap = []
            for g in gal:
                cs = self.caps_of.get(g, [])
                if captions_per_image and len(cs) > captions_per_image:
                    cs = sorted(rng.choice(cs, captions_per_image, replace=False).tolist())
                q_cap += cs
            q_cap = np.array(q_cap, dtype=np.int64)
            q_img = self.cap_img[q_cap].astype(np.int64)
            sims = full_sims(self.cap[q_cap], self.img[gal])
            gt_col = np.array([col[int(i)] for i in q_img])
            gt_rank = ranks_from_scores(sims, gt_col)
            K = min(self.K, len(gal))
            idx = np.argsort(-sims, axis=1, kind="stable")[:, :K]
            sim = np.take_along_axis(sims, idx, 1).astype(np.float32)
            cand = np.array(gal)[idx]
            hit = cand == q_img[:, None]
            pos = np.where(hit.any(1), hit.argmax(1), -1)
            if ensure_positive:  # train: ảnh đúng luôn nằm trong ứng viên (thay ứng viên cuối)
                miss = pos < 0
                cand[miss, -1] = q_img[miss]
                sim[miss, -1] = sims[miss, gt_col[miss]]
                pos[miss] = K - 1
            for k, v in zip(out, (q_cap, q_img, cand, sim, pos, gt_rank)):
                out[k].append(v)
        res = {k: np.concatenate(v) for k, v in out.items()}
        res["gallery"] = [int(x) for g in galleries for x in g]
        res["gallery_size"] = len(galleries[0])
        res["group"] = self.groups[res["q_cap"]]
        res["split"], res["seed"] = split, seed
        self._cache[key] = res
        return res

    # ---- evaluation text -> image
    def evaluate(self, score_fn, name, split="test", seeds=(0, 1, 2), full=True, by_group=True, verbose=True):
        """Chạy ``score_fn`` trên từng fold theo seed (+ toàn bộ pool nếu ``full``) và gộp mean ± std."""
        per_seed, groups = [], {}
        for seed in list(seeds) + ([None] if full else []):
            c = self.candidates(split, seed)
            scores = np.asarray(score_fn(c, self), dtype=np.float64)
            assert scores.shape == c["sim"].shape, f"score_fn phải trả về (Nq,K)={c['sim'].shape}, nhận {scores.shape}"
            ranks = rerank_ranks(c, scores)
            m = rank_metrics(ranks)
            rec = dict(seed=seed, gallery_size=c["gallery_size"], ranks=ranks.tolist(), **m)
            if by_group:
                rec["by_group"] = {g: rank_metrics(ranks[c["group"] == g]) for g in np.unique(c["group"])}
            per_seed.append(rec)
            if verbose:
                tag = "full" if seed is None else f"seed {seed}"
                log(f"{name:>24s} | {split} {tag:>6s} | gallery {c['gallery_size']:5d} | " +
                    " ".join(f"{k}={m[k]:.2f}" for k in METRICS) + f" MedR={m['MedR']:.0f}")
        seeded = [r for r in per_seed if r["seed"] is not None]
        res = dict(name=name, split=split, seeds=list(seeds), gallery_size=self.G, per_seed=per_seed,
                   mean_std=aggregate(seeded) if seeded else None,
                   full=next((r for r in per_seed if r["seed"] is None), None))
        if by_group and seeded:
            for g in np.unique(self.groups):
                vals = [r["by_group"][g] for r in seeded if g in r["by_group"]]
                if vals:
                    groups[g] = aggregate(vals)
            res["by_group"] = groups
        return res

    # ---- evaluation image -> text (nhánh B)
    def evaluate_i2t(self, name="CLIP", split="test", seeds=(0, 1, 2), full=True, score_fn=None, verbose=True):
        """Truy vấn = ảnh, gallery = mọi caption của ảnh trong gallery; đúng nếu lấy được 1 trong 5 caption.

        ``score_fn(sims, protocol, gallery, cap_idx) -> (Ni, Nc)`` để cắm phương pháp khác (mặc định: cosine).
        """
        per_seed = []
        for seed in list(seeds) + ([None] if full else []):
            gal = self.gallery(split, seed)
            cap_idx = np.array([ci for g in gal for ci in self.caps_of.get(g, [])])
            sims = full_sims(self.img[gal], self.cap[cap_idx])
            if score_fn is not None:
                sims = np.asarray(score_fn(sims, self, gal, cap_idx), np.float64)
            owner = self.cap_img[cap_idx]
            ranks = []
            for r, g in enumerate(gal):
                mine = np.where(owner == g)[0]
                ranks.append(int(ranks_from_scores(sims[r][None, :].repeat(len(mine), 0), mine).min()))
            m = rank_metrics(ranks)
            per_seed.append(dict(seed=seed, gallery_size=len(gal), ranks=ranks, **m))
            if verbose:
                tag = "full" if seed is None else f"seed {seed}"
                log(f"{name + ' (i2t)':>24s} | {split} {tag:>6s} | gallery {len(gal):5d} | " +
                    " ".join(f"{k}={m[k]:.2f}" for k in METRICS) + f" MedR={m['MedR']:.0f}")
        seeded = [r for r in per_seed if r["seed"] is not None]
        return dict(name=name + " (image→text)", split=split, seeds=list(seeds), per_seed=per_seed,
                    mean_std=aggregate(seeded) if seeded else None,
                    full=next((r for r in per_seed if r["seed"] is None), None))

    # ---- helpers for other methods
    def image_objects(self, idx):
        return self.images[int(idx)]["objects"]

    def caption_text(self, idx):
        return self.captions[int(idx)]["text"]

    def object_vocab(self, split="train", min_images=5, max_size=2000):
        """Từ vựng tên đối tượng đếm trên ảnh TRAIN (không rò rỉ từ val/test)."""
        cnt = Counter()
        for im in self.images:
            if im["split"] == split:
                cnt.update(set(im["objects"]))
        return [n for n, c in cnt.most_common(max_size) if c >= min_images]
