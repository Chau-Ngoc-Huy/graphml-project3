"""MMXAI Lab UI — thử mô hình, xem dữ liệu từng bước, xem ablation trên dữ liệu THẬT.

Chạy:   make ui            (hoặc: .venv/bin/python -m uvicorn ui.server:app --port 8765)
Mở:     http://127.0.0.1:8765

Biến môi trường (tùy chọn):
  MMXAI_BUNDLE   bundle Nhiệm vụ 1 (mặc định data/task1_bundle)
  MMXAI_RUN      thư mục lượt t1_qa có model.pt (mặc định: lượt `t1_qa-seed0` mới nhất)
  MMXAI_THREADS  số luồng torch (mặc định 2 — máy không bị lag)

Chỉ đọc dữ liệu và model đã có; không huấn luyện. CLIP text encoder chỉ nạp ở lần gõ câu tự do đầu tiên (~10 s).
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mmxai.data.contract import normalize_label
from mmxai.data.task1 import load_task1
from mmxai.experiments.t1_graph import load_eval_module
from mmxai.graph.build import CONCEPT, CONTAINS, SG_REL, TYPE_NAMES, build_graph
from mmxai.graph.qa import QAData, QAScorer

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
torch.set_num_threads(int(os.environ.get("MMXAI_THREADS", "2")))
K = 100  # số ứng viên CLIP được xếp hạng lại (giống giao thức của Thắng)


def _pick_run() -> Path:
    if os.environ.get("MMXAI_RUN"):
        return Path(os.environ["MMXAI_RUN"])
    cands = sorted(p for p in (ROOT / "runs" / "t1_qa").glob("*_t1_qa-seed0") if (p / "model.pt").exists())
    if not cands:
        raise SystemExit("Không thấy runs/t1_qa/*_t1_qa-seed0/model.pt — chạy `make local EXP=t1_qa ARGS='--seed 0 --tag seed0'`")
    return cands[-1]


class State:
    """Mọi thứ nạp một lần khi khởi động."""

    def __init__(self):
        self.bundle = Path(os.environ.get("MMXAI_BUNDLE", ROOT / "data" / "task1_bundle"))
        if not self.bundle.is_absolute():
            self.bundle = ROOT / self.bundle
        self.run = _pick_run()
        cfg = json.loads((self.run / "config.json").read_text())
        self.cfg = cfg
        extra = cfg.get("extra")
        self.extra = (ROOT / extra) if extra and not Path(extra).is_absolute() else extra
        self.ev = load_eval_module(self.bundle)
        b = load_task1(self.bundle, self.extra)
        self.b = b
        gcfg = cfg.get("graph", {})
        self.g = build_graph(b, gcfg.get("variant", "full"), int(gcfg.get("min_rel_count", 1)))
        self.cap_by_id = {c["caption_id"]: c for c in b.captions}
        self.cap_row = {c["caption_id"]: i for i, c in enumerate(b.captions)}
        self._clip_baseline = None
        self.qd = QAData(self.g, {cid: c["concepts"] for cid, c in self.cap_by_id.items()})
        self.model = QAScorer(b.image_emb.shape[1], **cfg.get("gnn", {}))
        self.model.load_state_dict(torch.load(self.run / "model.pt", map_location="cpu"))
        self.model.eval()
        info = json.loads((self.run / "train_info.json").read_text())[0]
        self.alpha = float(info["alpha"])
        self.train_info = info
        with torch.no_grad():
            self.H = self.model.label_h(self.qd.x_lab, self.qd.ei_lab)
            self.x_img = torch.from_numpy(b.image_emb)
            self.img_ad = F.normalize(self.model.skip(self.x_img), dim=-1)
        self.label_idx = {l: i for i, l in enumerate(self.g.labels)}
        self.n_vocab = len(json.loads((self.bundle / "data" / "object_vocab_train.json").read_text()))
        # meta ảnh (url, đối tượng gốc, quan hệ) theo idx của Thắng
        self.meta = {}
        with open(self.bundle / "data" / "images.jsonl", encoding="utf-8") as f:
            for line in f:
                im = json.loads(line)
                self.meta[im["idx"]] = {"url": im.get("url"), "coco_url": im.get("coco_url"), "split": im["split"],
                                        "objects": im["objects"], "relations": im["relations"]}
        self.row_of = {iid: r for r, iid in enumerate(b.image_ids)}
        self.pool = {s: np.array([self.row_of[i] for i in b.splits[s]]) for s in ("val", "test")}
        # cạnh nhãn–nhãn theo đỉnh (chỉ số trong không gian nhãn), để vẽ và tìm đường 2 bước
        ei, et, n = self.g.data.edge_index, self.g.data.edge_type, self.g.n_img
        lab = (ei[0] >= n) & (ei[1] >= n)
        self.lab_nbrs: dict[int, list[tuple[int, int]]] = {}
        for s, d, t in zip((ei[0][lab] - n).tolist(), (ei[1][lab] - n).tolist(), et[lab].tolist()):
            self.lab_nbrs.setdefault(d, []).append((s, t))
        self.img_count = Counter((ei[0][et == CONTAINS] - n).tolist())
        self._clip = None
        self._lock = threading.Lock()

    # ---------- CLIP text (lười)
    def encode_text(self, text: str) -> np.ndarray:
        with self._lock:
            if self._clip is None:
                import open_clip

                meta = json.loads((self.bundle / "features" / "meta.json").read_text())
                m, _, _ = open_clip.create_model_and_transforms(meta["model"], pretrained=meta["pretrained"])
                self._clip = (m.eval(), open_clip.get_tokenizer(meta["model"]))
            m, tok = self._clip
            with torch.no_grad():
                e = m.encode_text(tok([text])).float()
            return F.normalize(e, dim=-1)[0].numpy()

    def extract_concepts(self, text: str) -> list[str]:
        """Xấp xỉ trường `concepts` của Thắng cho câu tự do: khớp 1–3 từ liên tiếp với bộ nhãn gốc (bỏ s/es)."""
        words = re.findall(r"[a-z]+", text.lower())
        vocab = set(self.g.labels[: self.n_vocab])
        out, i = [], 0
        while i < len(words):
            hit = None
            for n in (3, 2, 1):
                ph = " ".join(words[i:i + n])
                for cand in (ph, ph[:-1] if ph.endswith("s") else None, ph[:-2] if ph.endswith("es") else None):
                    if cand and normalize_label(cand, {}) in vocab:
                        hit = (normalize_label(cand, {}), n)
                        break
                if hit:
                    break
            if hit and hit[0] not in out and hit[0] not in {"a", "the", "of", "on", "in"}:
                out.append(hit[0])
            i += hit[1] if hit else 1
        return out

    def img_card(self, row: int, **extra) -> dict:
        iid = self.b.image_ids[row]
        m = self.meta[iid]
        return {"row": row, "idx": iid, "url": m["url"], "coco_url": m["coco_url"], "split": m["split"],
                "labels": [self.g.labels[l] for l in self.qd.img_labels[row].tolist() if l >= 0], **extra}


S: State | None = None
app = FastAPI(title="MMXAI Lab")


@app.on_event("startup")
def _startup():
    global S
    S = State()
    print(f"[ui] model: {S.run.name} | α={S.alpha} | {S.g.stats['n_img']} ảnh, {len(S.g.labels)} nhãn/khái niệm")


# ---------------------------------------------------------------- tổng quan dữ liệu theo bước
@app.get("/api/overview")
def overview():
    b, g = S.b, S.g
    caps_by = Counter((c.get("group") or "?") for c in b.captions)
    split_caps = Counter()
    for c in b.captions:
        split_caps[S.meta[c["image_id"]]["split"]] += 1
    st = g.stats
    lpi = [(S.qd.img_labels[r] >= 0).sum().item() for r in range(g.n_img)]
    return {
        "model_run": S.run.name, "alpha": S.alpha, "config": {"graph": S.cfg.get("graph"), "gnn": S.cfg.get("gnn"),
                                                              "extra": S.cfg.get("extra")},
        "learned": {"w": S.model.w.item(), "tau_match": S.model.log_tau.exp().item(), "gate": S.model.gate.item()},
        "steps": [
            {"id": "raw", "title": "1 · Dữ liệu gốc", "who": "Yến / Thắng",
             "items": [["Ảnh VG ∩ COCO", len(b.image_ids)], ["Caption COCO", len(b.captions)],
                       ["Ảnh train / val / test", f"{len(b.splits['train'])} / {len(b.splits['val'])} / {len(b.splits['test'])}"],
                       ["Caption train / val / test", f"{split_caps['train']} / {split_caps['val']} / {split_caps['test']}"]]},
            {"id": "clip", "title": "2 · Đặc trưng CLIP", "who": "Thắng",
             "items": [["Mô hình", "ViT-B-32 / laion2b (đóng băng)"], ["Chiều vector", b.image_emb.shape[1]],
                       ["Ứng viên mỗi truy vấn", f"top-{K}"]]},
            {"id": "graph", "title": "3 · Đồ thị", "who": "Huy",
             "items": [["Đỉnh ảnh", st["n_img"]], ["Đỉnh nhãn VG", S.n_vocab],
                       ["Đỉnh khái niệm ConceptNet", len(g.labels) - S.n_vocab],
                       ["Cạnh contains (nhãn → ảnh)", st.get("E_contains", 0)],
                       ["Cạnh sg_rel (nhãn ↔ nhãn, từ ảnh train)", st.get("E_sg_rel", 0)],
                       ["Cạnh ConceptNet", st.get("E_concept", 0)],
                       ["Nhãn / ảnh (trung bình · tối đa)", f"{np.mean(lpi):.1f} · {max(lpi)}"]]},
            {"id": "query", "title": "4 · Truy vấn → khái niệm", "who": "Thắng (concepts) / Huy",
             "items": [["Khái niệm / caption (trung bình)",
                        f"{np.mean([len(c['concepts']) for c in b.captions]):.2f}"],
                       ["Nhóm direct / indirect / no_concept",
                        f"{caps_by.get('direct', 0)} / {caps_by.get('indirect', 0)} / {caps_by.get('no_concept', 0)}"]]},
            {"id": "score", "title": "5 · Chấm điểm & xếp hạng lại", "who": "Huy",
             "items": [["Điểm mô hình", "cos(Pq, Sx) + w · s_G(q, ảnh)"],
                       ["s_G", "trung bình theo khái niệm của max mềm cos(u_c, H_nhãn)"],
                       ["Điểm cuối", f"α·z(cos CLIP) + (1−α)·z(điểm mô hình), α = {S.alpha}"]]},
        ],
        "label_degree_top": [[S.g.labels[l], c] for l, c in S.img_count.most_common(15)],
        "rel_hist": _rel_hist(),
    }


def _rel_hist():
    c = Counter(r.split("⁻")[0] for r in S.g.rel_name.values())
    return c.most_common(12)


# ---------------------------------------------------------------- caption mẫu
@app.get("/api/captions")
def captions(split: str = "test", group: str = "", n: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    pool = [c for c in S.b.captions if S.meta[c["image_id"]]["split"] == split and (not group or c.get("group") == group)]
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [{"caption_id": pool[i]["caption_id"], "text": pool[i]["text"], "group": pool[i].get("group"),
             "concepts": pool[i]["concepts"], "gold": S.row_of[pool[i]["image_id"]]} for i in pick]


# ---------------------------------------------------------------- tìm kiếm
class SearchReq(BaseModel):
    text: str | None = None
    caption_id: int | None = None
    concepts: list[str] | None = None   # sửa tay khái niệm (tùy chọn)
    split: str = "test"
    k: int = 12


def _concept_ids(names):
    ids = [S.label_idx[n] for n in names if n in S.label_idx]
    return list(dict.fromkeys(ids))


@app.post("/api/search")
def search(req: SearchReq):
    gold = None
    if req.caption_id is not None:
        c = S.cap_by_id.get(req.caption_id)
        if c is None:
            raise HTTPException(404, "caption_id không có")
        text, q = c["text"], S.b.caption_emb[S.cap_row[req.caption_id]]
        concepts = req.concepts if req.concepts is not None else c["concepts"]
        gold = S.row_of[c["image_id"]]
        group = c.get("group")
    else:
        if not req.text or not req.text.strip():
            raise HTTPException(400, "Nhập câu truy vấn")
        text = req.text.strip()
        q = S.encode_text(text)
        concepts = req.concepts if req.concepts is not None else S.extract_concepts(text)
        group = None
    pool = S.pool[req.split]
    sims = S.b.image_emb[pool] @ q
    order = np.argsort(-sims)[:K]
    cand, sim = pool[order], sims[order]
    cids = _concept_ids(concepts)
    with torch.no_grad():
        qa = F.normalize(S.model.q_proj(torch.from_numpy(q)[None]), dim=-1)[0]
        ad = (S.img_ad[torch.from_numpy(cand)] @ qa).numpy()
        if cids:
            M = S.model.concept_image_match(S.H, torch.tensor(cids), S.qd.img_labels[torch.from_numpy(cand)])
            gs = M.mean(0).numpy()
            per_concept = M.numpy()
        else:
            gs, per_concept = np.zeros(len(cand)), np.zeros((0, len(cand)))
    total = ad + S.model.w.item() * gs
    fused = S.ev.fuse(sim[None], total[None], S.alpha)[0]
    ranks = {}
    for name, sc in (("clip", sim), ("graph", gs), ("final", fused)):
        o = np.argsort(-sc, kind="stable")
        r = np.empty(len(o), int)
        r[o] = np.arange(1, len(o) + 1)
        ranks[name] = r

    def rows(name):
        o = np.argsort(ranks[name])[: req.k]
        return [S.img_card(int(cand[j]), rank=int(ranks[name][j]), clip_rank=int(ranks["clip"][j]),
                           final_rank=int(ranks["final"][j]), clip=float(sim[j]), adapter=float(ad[j]),
                           graph=float(gs[j]), fused=float(fused[j]), is_gold=bool(gold is not None and cand[j] == gold),
                           per_concept=[float(x) for x in per_concept[:, j]]) for j in o]

    gold_info = None
    if gold is not None:
        hit = np.where(cand == gold)[0]
        gold_info = {"row": int(gold), "in_top_k": bool(len(hit)),
                     "clip_rank": int(ranks["clip"][hit[0]]) if len(hit) else None,
                     "graph_rank": int(ranks["graph"][hit[0]]) if len(hit) else None,
                     "final_rank": int(ranks["final"][hit[0]]) if len(hit) else None,
                     "card": S.img_card(int(gold))}
    return {"text": text, "group": group, "concepts": [S.g.labels[i] for i in cids],
            "concepts_unknown": [n for n in concepts if n not in S.label_idx],
            "alpha": S.alpha, "w": S.model.w.item(), "pool": req.split, "pool_size": int(len(pool)), "K": K,
            "results": {"clip": rows("clip"), "graph": rows("graph"), "final": rows("final")}, "gold": gold_info,
            "moved": _moved(cand, ranks)}


def _moved(cand, ranks):
    d = ranks["clip"] - ranks["final"]
    o = np.argsort(-np.abs(d))[:8]
    return [{"row": int(cand[j]), "clip_rank": int(ranks["clip"][j]), "final_rank": int(ranks["final"][j])} for j in o]


# ---------------------------------------------------------------- giải thích + đồ thị con của một ảnh
class ExplainReq(BaseModel):
    row: int
    concepts: list[str] = []
    max_other_labels: int = 40


@app.post("/api/explain")
def explain(req: ExplainReq):
    row = req.row
    labs = [l for l in S.qd.img_labels[row].tolist() if l >= 0]
    cids = _concept_ids(req.concepts)
    n_img = S.g.n_img
    nodes, edges, seen = {}, [], set()

    def add_node(nid, label, kind, **kw):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": label, "kind": kind, **kw}

    def add_edge(a, b, kind, **kw):
        key = (a, b, kind)
        if key not in seen:
            seen.add(key)
            edges.append({"source": a, "target": b, "kind": kind, **kw})

    img_id = f"img:{row}"
    add_node(img_id, f"ảnh #{S.b.image_ids[row]}", "image")
    matches = []
    tau = S.model.log_tau.exp().item()
    with torch.no_grad():
        for c in cids:
            u = F.normalize(S.model.c_proj(S.H[c]), dim=-1)
            cos = (S.H[torch.tensor(labs)] @ u).numpy() if labs else np.zeros(0)
            if len(cos):
                wsm = np.exp((cos - cos.max()) / tau)
                wsm /= wsm.sum()
                m = float(tau * (np.log(np.exp((cos - cos.max()) / tau).sum())) + cos.max())
            else:
                wsm, m = cos, 0.0
            top = np.argsort(-cos)[:3]
            path = None
            if len(top):
                best = labs[top[0]]
                if best != c:
                    path = _two_hop(c, best)
            matches.append({"concept": S.g.labels[c], "in_image": c in labs, "m": m,
                            "top": [{"label": S.g.labels[labs[j]], "cos": float(cos[j]), "weight": float(wsm[j])}
                                    for j in top], "path": path})
    hot = {labs[j] for mt in matches for j in range(len(labs)) if S.g.labels[labs[j]] in {t["label"] for t in mt["top"][:1]}}
    show = list(dict.fromkeys([*hot, *labs]))[: len(hot) + req.max_other_labels]
    for l in show:
        add_node(f"lab:{l}", S.g.labels[l], "label" if l < S.n_vocab else "concept", hot=l in hot)
        add_edge(f"lab:{l}", img_id, "contains", hot=l in hot)
    for mt, c in zip(matches, cids):
        qid = f"q:{c}"
        add_node(qid, S.g.labels[c], "query")
        if mt["top"]:
            best = S.label_idx[mt["top"][0]["label"]]
            if mt["path"]:
                prev = qid
                for step in mt["path"]["nodes"][1:]:
                    nid = f"lab:{step['idx']}"
                    add_node(nid, step["name"], "label" if step["idx"] < S.n_vocab else "concept")
                    add_edge(prev, nid, step["edge"], hot=True)
                    prev = nid
            else:
                add_edge(qid, f"lab:{best}", "match", hot=True, cos=mt["top"][0]["cos"])
    shown = {int(k.split(":")[1]) for k in nodes if k.startswith("lab:")}
    extra = 0
    for l in list(shown):
        for s, t in S.lab_nbrs.get(l, []):
            if s in shown and s < l and extra < 80:
                add_edge(f"lab:{s}", f"lab:{l}", TYPE_NAMES[t], rel=S.g.rel_name.get((s + n_img, l + n_img), ""))
                extra += 1
    m = S.meta[S.b.image_ids[row]]
    return {"card": S.img_card(row), "relations": m["relations"][:30], "matches": matches,
            "graph_score": float(np.mean([mt["m"] for mt in matches])) if matches else 0.0,
            "elements": {"nodes": list(nodes.values()), "edges": edges},
            "counts": {"labels_total": len(labs), "labels_shown": len(show), "label_label_edges": extra}}


def _two_hop(c: int, target: int):
    """Đường ngắn nhất ≤ 2 bước c → x → target trên đồ thị nhãn (sg_rel / ConceptNet), để minh họa câu gián tiếp."""
    def name(i):
        return S.g.labels[i]
    direct = [t for s, t in S.lab_nbrs.get(target, []) if s == c]
    if direct:
        return {"nodes": [{"idx": c, "name": name(c)}, {"idx": target, "name": name(target), "edge": TYPE_NAMES[direct[0]]}]}
    into_t = {s: t for s, t in S.lab_nbrs.get(target, [])}
    for s, t in S.lab_nbrs.get(c, []):  # cạnh s→c tồn tại thì c→s cũng tồn tại (đồ thị nhãn hai chiều)
        if s in into_t:
            return {"nodes": [{"idx": c, "name": name(c)}, {"idx": s, "name": name(s), "edge": TYPE_NAMES[t]},
                              {"idx": target, "name": name(target), "edge": TYPE_NAMES[into_t[s]]}]}
    return None


# ---------------------------------------------------------------- graph builder: đầu vào → đầu ra của MỘT ảnh
@app.get("/api/images")
def images(split: str = "test", label: str = "", n: int = 24, seed: int = 0):
    """Chọn ảnh để xem: lọc theo split và (tùy chọn) nhãn có trong ảnh."""
    rows = [S.row_of[i] for i in S.b.splits.get(split, [])]
    if label:
        l = S.label_idx.get(normalize_label(label, {}))
        if l is None:
            raise HTTPException(404, f"Không có nhãn '{label}'")
        rows = [r for r in rows if bool((S.qd.img_labels[r] == l).any())]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False) if rows else []
    return {"total": len(rows), "items": [S.img_card(int(rows[i])) for i in pick]}


@app.get("/api/build/{row}")
def build(row: int, two_hop: int = 0):
    """Tái hiện build_graph cho một ảnh: objects/relations gốc → nhãn giữ/bỏ → cạnh contains, sg_rel, ConceptNet."""
    if not 0 <= row < S.g.n_img:
        raise HTTPException(404, "row ngoài phạm vi")
    iid = S.b.image_ids[row]
    m = S.meta[iid]
    vocab = S.label_idx
    n_img = S.g.n_img
    is_train = m["split"] == "train"
    # 1) đối tượng gốc → nhãn
    objs = Counter(m["objects"])
    objects = []
    for o, c in objs.most_common():
        nl = normalize_label(o, {})
        objects.append({"raw": o, "norm": nl, "count": c, "kept": nl in vocab and S.label_idx[nl] < S.n_vocab,
                        "images_with_label": S.img_count.get(vocab[nl], 0) if nl in vocab else 0})
    labs = [l for l in S.qd.img_labels[row].tolist() if l >= 0]
    labset = set(labs)
    # 2) quan hệ gốc → có thành cạnh sg_rel không
    rels = []
    for s, p, o in m["relations"]:
        ns, no = normalize_label(s, {}), normalize_label(o, {})
        both = ns in vocab and no in vocab and ns != no
        in_graph = both and (vocab[ns] + n_img, vocab[no] + n_img) in S.g.rel_name
        why = ("dùng (ảnh train) — gộp với các ảnh train khác" if is_train and in_graph else
               "ảnh không thuộc train → không đóng góp cạnh (chống rò rỉ)" if not is_train and both else
               "một đầu không thuộc bộ nhãn → bỏ" if not both else "bị lọc bởi min_rel_count")
        rels.append({"s": s, "p": p, "o": o, "used": bool(is_train and in_graph),
                     "edge_exists": bool(in_graph), "graph_rel": S.g.rel_name.get((vocab[ns] + n_img, vocab[no] + n_img), "") if both else "",
                     "why": why})
    # 3) cạnh đầu ra quanh ảnh
    nodes, edges, seen = {}, [], set()
    img_id = f"img:{row}"
    nodes[img_id] = {"id": img_id, "label": f"ảnh #{iid}", "kind": "image"}
    contains = []
    for l in labs:
        nodes[f"lab:{l}"] = {"id": f"lab:{l}", "label": S.g.labels[l], "kind": "label"}
        edges.append({"source": f"lab:{l}", "target": img_id, "kind": "contains"})
        contains.append({"label": S.g.labels[l], "images_with_label": S.img_count.get(l, 0)})
    among, concept, outside, seen_out = [], [], Counter(), set()
    for l in labs:
        for s, t in S.lab_nbrs.get(l, []):
            key = (min(s, l), max(s, l), t)
            if s in labset:
                if key in seen:
                    continue
                seen.add(key)
                rel = S.g.rel_name.get((s + n_img, l + n_img), "")
                among.append({"a": S.g.labels[s], "b": S.g.labels[l], "type": TYPE_NAMES[t], "rel": rel})
                edges.append({"source": f"lab:{s}", "target": f"lab:{l}", "kind": TYPE_NAMES[t], "rel": rel})
            else:
                if (s, t, l) not in seen_out:  # đếm số NHÃN của ảnh mà s nối tới (cạnh sg_rel có thể bị trùng)
                    seen_out.add((s, t, l))
                    outside[(s, t)] += 1
                if t == CONCEPT and key not in seen:
                    seen.add(key)
                    concept.append({"label": S.g.labels[l], "rel": S.g.rel_name.get((s + n_img, l + n_img), ""),
                                    "concept": S.g.labels[s]})
    hop2 = [{"name": S.g.labels[s], "type": TYPE_NAMES[t], "links": c} for (s, t), c in outside.most_common(30)]
    if two_hop:
        for h in hop2[: int(two_hop)]:
            s = S.label_idx[h["name"]]
            nid = f"lab:{s}"
            nodes[nid] = {"id": nid, "label": h["name"], "kind": "concept" if s >= S.n_vocab else "hop2"}
            for l in labs:
                if any(x == s for x, _ in S.lab_nbrs.get(l, [])):
                    edges.append({"source": nid, "target": f"lab:{l}", "kind": h["type"]})
    kept = sum(1 for o in objects if o["kept"])
    return {"card": S.img_card(row), "is_train": is_train,
            "stats": {"objects_raw": sum(objs.values()), "objects_unique": len(objs), "labels_kept": kept,
                      "objects_dropped": len(objs) - kept, "relations_raw": len(rels),
                      "relations_used": sum(r["used"] for r in rels), "contains": len(contains),
                      "label_label_edges": len(among), "concept_edges": len(concept), "hop2_labels": len(outside)},
            "objects": objects, "relations": rels, "contains": contains, "among": among, "concept": concept,
            "hop2": hop2, "elements": {"nodes": list(nodes.values()), "edges": edges},
            "min_rel_count": int(S.cfg.get("graph", {}).get("min_rel_count", 1))}


# ---------------------------------------------------------------- khám phá một nhãn
@app.get("/api/label/{name}")
def label(name: str):
    l = S.label_idx.get(normalize_label(name, {}))
    if l is None:
        raise HTTPException(404, f"Không có nhãn '{name}'")
    n = S.g.n_img
    nb, seen = [], set()
    for s, t in S.lab_nbrs.get(l, []):
        if (s, t) in seen:  # build_graph thêm cạnh sg_rel 2 lần khi có cả (a,b) và (b,a) — hiển thị một lần
            continue
        seen.add((s, t))
        nb.append({"name": S.g.labels[s], "type": TYPE_NAMES[t], "rel": S.g.rel_name.get((s + n, l + n), ""),
                   "images": S.img_count.get(s, 0)})
    nb.sort(key=lambda x: -x["images"])
    with torch.no_grad():
        sim = (S.H @ S.H[l]).numpy()
    close = [(S.g.labels[i], float(sim[i])) for i in np.argsort(-sim)[1:11]]
    return {"name": S.g.labels[l], "kind": "label" if l < S.n_vocab else "concept", "images": S.img_count.get(l, 0),
            "neighbors": nb[:60], "n_neighbors": len(nb), "nearest_after_gnn": close}


@app.get("/api/labels")
def labels(q: str = ""):
    q = q.lower().strip()
    return [l for l in S.g.labels if q in l][:20]


# ---------------------------------------------------------------- ablation
STORY = [
    ("t1_clip", "", "CLIP (baseline của Thắng)", "Không học gì; cos(câu, ảnh)."),
    ("t1_graph", "seed0-dau-tien", "Legacy GNN (#2)", "Gộp nhãn + lớp chiếu; tưởng đồ thị giúp +2.8 MRR."),
    ("t1_ablation_no_edge", "seed0", "Legacy · no_edge (#3)", "Bỏ hết cạnh vẫn +2.8 ⇒ mức tăng là của adapter."),
    ("t1_ablation_shuffle", "seed0", "Legacy · shuffle (#3)", "Xáo cạnh vẫn như cũ ⇒ đồ thị không được dùng."),
    ("t1_ablation_no_edge", "split-seed0", "CLIP + Adapter (#4)", "Kiến trúc split, không cạnh = chỉ adapter."),
    ("t1_graph", "split-seed0", "Split: Adapter + Graph gộp (#4)", "Nhánh đồ thị tách rời: không thêm gì."),
    ("t1_ablation_shuffle", "split-seed0", "Split · shuffle (#4)", "Đối chứng."),
    ("t1_graph", "split-noadapter-seed0", "Split: Graph trên CLIP thô (#4)", "+0.4 MRR."),
    ("t1_qa", "untrained-noadapter-seed0", "QA: khớp nhãn không học (#5)", "Gần như không giúp."),
    ("t1_qa_ablation_no_edge", "seed0", "QA · no_edge = CLIP+Adapter (#5)", "Mốc so sánh đúng."),
    ("t1_qa", "seed0", "QA: Adapter + Graph theo truy vấn (#5)", "Lần đầu đồ thị thêm được ~+1 MRR (nhóm direct)."),
    ("t1_qa_ablation_no_rel", "seed0", "QA · no_rel (#6)", "Bỏ cạnh nhãn–nhãn: không giảm ⇒ sg_rel không giúp."),
    ("t1_qa", "rel5-seed0", "QA · sg_rel thưa ≥5 (#7)", "Cạnh thưa hơn cũng không giúp."),
    ("t1_qa", "cn-seed0", "QA + ConceptNet", "Phép thử cho nhóm indirect."),
]


def _clip_row():
    """Dòng CLIP tính lại bằng chính Protocol của Thắng (seed 0 + toàn pool) — khớp t1_clip (JOURNAL #2)."""
    if S._clip_baseline is None:
        P = S.ev.Protocol.load(S.bundle)
        res = P.evaluate(lambda c, p: c["sim"], name="CLIP", split="test", seeds=[0], full=True)
        ps, fl = res["per_seed"][0], res["full"]
        S._clip_baseline = {"exp": "t1_clip", "tag": "", "run_id": "protocol(seed 0)",
                            "full": {k: fl.get(k) for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10")},
                            "gallery": {k: ps.get(k) for k in ("R@1", "MRR")},
                            "by_group": {g: v.get("MRR") for g, v in (ps.get("by_group") or {}).items()},
                            "alpha": None, "history": [], "alpha_curve": [], "graph": None}
    return S._clip_baseline


@app.get("/api/ablation")
def ablation():
    runs = [_clip_row()]
    for f in sorted((ROOT / "results").glob("t1_*/*.json")):
        d = json.loads(f.read_text())
        r, m = d.get("run", {}), d.get("metrics", {})
        if not m.get("test_full"):
            continue
        tag = r.get("tag") or ""
        if "smoke" in tag:
            continue
        exp = r.get("exp")
        hist, curve = [], []
        ti = ROOT / "runs" / exp / r.get("run_id", "") / "train_info.json"
        if ti.exists():
            info = json.loads(ti.read_text())
            if info:
                hist = info[0].get("history", [])
                curve = info[0].get("alpha_curve", [])

        def g(key, block="test_full"):
            v = m.get(block, {}).get(key)
            return v.get("mean") if isinstance(v, dict) else v

        runs.append({"exp": exp, "tag": tag, "run_id": r.get("run_id"),
                     "full": {k: g(k) for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10")},
                     "gallery": {k: g(k, "test") for k in ("R@1", "MRR")},
                     "by_group": {gn: (v.get("MRR") or {}).get("mean") for gn, v in (m.get("by_group") or {}).items()},
                     "alpha": (m.get("alpha") or {}).get("values"), "history": hist, "alpha_curve": curve,
                     "graph": (r.get("extra") or {}).get("graph")})
    story = []
    for exp, tag, title, note in STORY:
        hit = [x for x in runs if x["exp"] == exp and x["tag"] == tag]
        if hit:
            story.append({**hit[-1], "title": title, "note": note})
    return {"story": story, "all": runs}


# ---------------------------------------------------------------- tĩnh
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
