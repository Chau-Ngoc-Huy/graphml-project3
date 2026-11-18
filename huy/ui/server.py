"""MMXAI Lab UI — thử từng biến thể mô hình, xem dữ liệu từng bước, xem ablation trên dữ liệu THẬT.

Chạy:   make ui            (hoặc: .venv/bin/python -m uvicorn ui.server:app --port 8765)
Mở:     http://127.0.0.1:8765

Biến thể = mọi thư mục runs/t1_qa*/<run_id>/ có model.pt (trừ lượt `smoke`). Nạp dần ở nền sau khi khởi động
(mỗi biến thể dựng lại đồ thị của nó ~20–60 s); gọi tới biến thể chưa nạp thì nạp ngay.

Biến môi trường (tùy chọn):
  MMXAI_BUNDLE   bundle Nhiệm vụ 1 (mặc định data/task1_bundle)
  MMXAI_RUN      run_id (tên thư mục) của biến thể mặc định
  MMXAI_THREADS  số luồng torch (mặc định 2 — máy không bị lag)
  MMXAI_PRELOAD  0 = không nạp sẵn các biến thể ở nền

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
from mmxai.graph.build import CONCEPT, CONTAINS, TYPE_NAMES, build_graph
from mmxai.graph.qa import QAData, QAScorer

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
torch.set_num_threads(int(os.environ.get("MMXAI_THREADS", "2")))
K = 100  # số ứng viên CLIP được xếp hạng lại (giống giao thức của Thắng)

# ---------------------------------------------------------------- tên hiển thị của biến thể
EXP_TITLE = {
    "t1_qa_ablation_no_edge": ("CLIP + Adapter (không đồ thị)", "Mốc so sánh: chỉ lớp chiếu học được, không cạnh nào."),
    "t1_qa": ("QA-Graph: contains + sg_rel", "Khái niệm của câu ↔ nhãn của ảnh, nhãn lan truyền qua cạnh sg_rel."),
    "t1_qa_ablation_no_rel": ("QA-Graph: chỉ contains", "Bỏ cạnh nhãn–nhãn: chỉ còn ảnh ↔ nhãn."),
    "t1_qa_cn": ("QA-Graph + ConceptNet", "contains + sg_rel + ConceptNet (1000 khái niệm cầu nối)."),
    "t1_qa_cn_no_sg": ("QA-Graph + ConceptNet, bỏ sg_rel", "contains + ConceptNet."),
    "t1_qa_cn_shuffle": ("Đối chứng: xáo cạnh (shuffle)", "Ảnh mang nhãn sai ⇒ phải tụt dưới Adapter."),
}
ORDER = ["t1_qa_ablation_no_edge", "t1_qa", "t1_qa_ablation_no_rel", "t1_qa_cn", "t1_qa_cn_no_sg", "t1_qa_cn_shuffle"]


def _describe(exp: str, tag: str, cfg: dict) -> tuple[str, str, int]:
    base = exp
    kind = ""
    for k in ("_gcn", "_sage"):
        if exp.endswith(k):
            base, kind = exp[: -len(k)], k[1:].upper()
    title, note = EXP_TITLE.get(base, (exp, ""))
    if base == "t1_qa" and "untrained" in tag:
        title, note = "Khớp nhãn CLIP, không học", "Không huấn luyện, không adapter."
    elif base == "t1_qa" and "rel5" in tag:
        title, note = "QA-Graph: sg_rel thưa (≥5 lần)", "Chỉ giữ cặp nhãn đồng xuất hiện ≥ 5 lần trên train."
    elif base == "t1_qa" and (cfg.get("extra") or "").endswith("huy_extra_cn"):
        title, note = "QA-Graph + ConceptNet", "contains + sg_rel + ConceptNet."
    gnn = kind or (cfg.get("gnn") or {}).get("kind", "gat").upper()
    src = "Kaggle 3 seed · model seed 0" if "kaggle" in tag else "máy · seed 0"
    pri = ORDER.index(base) if base in ORDER else len(ORDER)
    return f"{title} · {gnn} · {'Kaggle' if 'kaggle' in tag else 'máy'}", f"{note} ({src})", pri * 10 + (0 if "kaggle" in tag else 5) + (1 if kind else 0)


class Variant:
    """Một mô hình đã huấn luyện + đồ thị của nó. Nạp lười (lock riêng)."""

    def __init__(self, run: Path):
        self.run, self.id = run, run.name
        self.cfg = json.loads((run / "config.json").read_text())
        self.exp = run.parent.name
        tag = self.cfg.get("tag") or run.name.split(f"{self.exp}-", 1)[-1]
        self.tag = tag
        self.title, self.note, self.priority = _describe(self.exp, tag, self.cfg)
        self.loaded, self.error = False, None
        self._lock = threading.Lock()

    def info(self):
        return {"id": self.id, "title": self.title, "note": self.note, "exp": self.exp, "loaded": self.loaded,
                "error": self.error, "alpha": getattr(self, "alpha", None),
                "graph": self.cfg.get("graph"), "gnn": (self.cfg.get("gnn") or {}).get("kind"),
                "extra": self.cfg.get("extra")}

    def ensure(self):
        if self.loaded:
            return self
        with self._lock:
            if not self.loaded:
                try:
                    self._load()
                    self.loaded = True
                    print(f"[ui] nạp xong biến thể {self.id} ({self.title})", flush=True)
                except Exception as e:  # noqa: BLE001 — báo lỗi lên UI thay vì làm chết server
                    self.error = f"{type(e).__name__}: {e}"
                    raise
        return self

    def _load(self):
        cfg = self.cfg
        b = S.bundle_for(cfg.get("extra"))
        gcfg = cfg.get("graph", {})
        self.g = S.graph_for(b, cfg.get("extra"), gcfg.get("variant", "full"), int(gcfg.get("min_rel_count", 1)))
        self.b = b
        self.qd = QAData(self.g, S.cap_concepts)
        self.model = QAScorer(b.image_emb.shape[1], **cfg.get("gnn", {}))
        self.model.load_state_dict(torch.load(self.run / "model.pt", map_location="cpu"))
        self.model.eval()
        info = json.loads((self.run / "train_info.json").read_text())[0]
        self.alpha = float(info["alpha"])
        with torch.no_grad():
            self.H = self.model.label_h(self.qd.x_lab, self.qd.ei_lab)
            self.img_ad = F.normalize(self.model.skip(S.x_img), dim=-1)
        self.label_idx = {l: i for i, l in enumerate(self.g.labels)}
        ei, et, n = self.g.data.edge_index, self.g.data.edge_type, self.g.n_img
        lab = (ei[0] >= n) & (ei[1] >= n)
        self.lab_nbrs: dict[int, list[tuple[int, int]]] = {}
        for s, d, t in zip((ei[0][lab] - n).tolist(), (ei[1][lab] - n).tolist(), et[lab].tolist()):
            self.lab_nbrs.setdefault(d, []).append((s, t))
        self.img_count = Counter((ei[0][et == CONTAINS] - n).tolist())

    def labels_of(self, row: int) -> list[int]:
        return [l for l in self.qd.img_labels[row].tolist() if l >= 0]

    def concept_ids(self, names):
        return list(dict.fromkeys(self.label_idx[n] for n in names if n in self.label_idx))


class State:
    """Dữ liệu dùng chung cho mọi biến thể."""

    def __init__(self):
        self.bundle = Path(os.environ.get("MMXAI_BUNDLE", ROOT / "data" / "task1_bundle"))
        if not self.bundle.is_absolute():
            self.bundle = ROOT / self.bundle
        self.ev = load_eval_module(self.bundle)
        self._bundles, self._graphs, self._lock = {}, {}, threading.Lock()
        b = self.bundle_for(None)
        self.b = b
        self.x_img = torch.from_numpy(b.image_emb)
        self.cap_by_id = {c["caption_id"]: c for c in b.captions}
        self.cap_row = {c["caption_id"]: i for i, c in enumerate(b.captions)}
        self.cap_concepts = {cid: c["concepts"] for cid, c in self.cap_by_id.items()}
        self.n_vocab = len(json.loads((self.bundle / "data" / "object_vocab_train.json").read_text()))
        self.vocab = set(json.loads((self.bundle / "data" / "object_vocab_train.json").read_text()))
        self.vocab = {normalize_label(v, {}) for v in self.vocab}
        self.meta = {}
        with open(self.bundle / "data" / "images.jsonl", encoding="utf-8") as f:
            for line in f:
                im = json.loads(line)
                self.meta[im["idx"]] = {"url": im.get("url"), "coco_url": im.get("coco_url"), "split": im["split"],
                                        "objects": im["objects"], "relations": im["relations"]}
        self.row_of = {iid: r for r, iid in enumerate(b.image_ids)}
        self.pool = {s: np.array([self.row_of[i] for i in b.splits[s]]) for s in ("val", "test")}
        self._clip, self._clip_lock = None, threading.Lock()
        self._clip_baseline = None
        runs = [p for d in sorted((ROOT / "runs").glob("t1_qa*")) for p in sorted(d.iterdir())
                if (p / "model.pt").exists() and (p / "config.json").exists() and "smoke" not in p.name]
        self.variants = {v.id: v for v in sorted((Variant(p) for p in runs), key=lambda v: (v.priority, v.id))}
        want = os.environ.get("MMXAI_RUN", "")
        want = Path(want).name if want else ""
        if want not in self.variants:
            pref = [v for v in self.variants.values() if v.exp == "t1_qa_cn" and "kaggle" in v.tag] or \
                   [v for v in self.variants.values() if v.exp == "t1_qa"] or list(self.variants.values())
            want = pref[0].id if pref else ""
        if not want:
            raise SystemExit("Không có biến thể nào (runs/t1_qa*/<run>/model.pt). Chạy `make local EXP=t1_qa ARGS='--seed 0'`.")
        self.default = want

    # ---- bundle / đồ thị dùng chung giữa các biến thể
    def resolve_extra(self, extra):
        if not extra:
            return None
        for cand in (Path(extra), ROOT / extra, self.bundle / extra, self.bundle / Path(extra).name):
            if cand.exists():
                return cand.resolve()
        raise FileNotFoundError(f"Không thấy thư mục extra '{extra}'")

    def bundle_for(self, extra):
        key = str(self.resolve_extra(extra) or "")
        with self._lock:
            if key not in self._bundles:
                b = load_task1(self.bundle, key or None)
                if self._bundles:  # dùng chung mảng lớn với bundle gốc (giống hệt nhau, chỉ khác nhãn/khái niệm)
                    base = next(iter(self._bundles.values()))
                    b.image_emb, b.caption_emb, b.captions = base.image_emb, base.caption_emb, base.captions
                self._bundles[key] = b
            return self._bundles[key]

    def graph_for(self, b, extra, variant, min_rel):
        key = (str(self.resolve_extra(extra) or ""), variant, min_rel)
        with self._lock:
            if key not in self._graphs:
                self._graphs[key] = build_graph(b, variant, min_rel)
            return self._graphs[key]

    def var(self, vid: str | None) -> Variant:
        v = self.variants.get(vid or self.default)
        if v is None:
            raise HTTPException(404, f"Không có biến thể '{vid}'")
        try:
            return v.ensure()
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Nạp biến thể lỗi: {e}") from e

    # ---- CLIP text (lười)
    def encode_text(self, text: str) -> np.ndarray:
        with self._clip_lock:
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
        out, i = [], 0
        while i < len(words):
            hit = None
            for n in (3, 2, 1):
                ph = " ".join(words[i:i + n])
                for cand in (ph, ph[:-1] if ph.endswith("s") else None, ph[:-2] if ph.endswith("es") else None):
                    if cand and normalize_label(cand, {}) in self.vocab:
                        hit = (normalize_label(cand, {}), n)
                        break
                if hit:
                    break
            if hit and hit[0] not in out and hit[0] not in {"a", "the", "of", "on", "in"}:
                out.append(hit[0])
            i += hit[1] if hit else 1
        return out

    def img_card(self, row: int, V: Variant | None = None, **extra) -> dict:
        iid = self.b.image_ids[row]
        m = self.meta[iid]
        labs = [V.g.labels[l] for l in V.labels_of(row)] if V else []
        return {"row": row, "idx": iid, "url": m["url"], "coco_url": m["coco_url"], "split": m["split"],
                "labels": labs, **extra}


S: State | None = None
app = FastAPI(title="MMXAI Lab")


def _preload():
    for v in list(S.variants.values()):
        try:
            v.ensure()
        except Exception as e:  # noqa: BLE001
            print(f"[ui] lỗi nạp {v.id}: {e}", flush=True)


@app.on_event("startup")
def _startup():
    global S
    S = State()
    S.var(S.default)
    print(f"[ui] {len(S.variants)} biến thể; mặc định {S.default}", flush=True)
    if os.environ.get("MMXAI_PRELOAD", "1") != "0":
        threading.Thread(target=_preload, daemon=True).start()


@app.get("/api/variants")
def variants():
    return {"default": S.default, "items": [v.info() for v in S.variants.values()]}


# ---------------------------------------------------------------- tổng quan dữ liệu theo bước
@app.get("/api/overview")
def overview(variant: str | None = None):
    V = S.var(variant)
    b, g = S.b, V.g
    caps_by = Counter((c.get("group") or "?") for c in b.captions)
    split_caps = Counter()
    for c in b.captions:
        split_caps[S.meta[c["image_id"]]["split"]] += 1
    st = g.stats
    lpi = [(V.qd.img_labels[r] >= 0).sum().item() for r in range(g.n_img)]
    return {
        "variant": V.info(), "model_run": V.id, "alpha": V.alpha,
        "config": {"graph": V.cfg.get("graph"), "gnn": V.cfg.get("gnn"), "extra": V.cfg.get("extra")},
        "learned": {"w": V.model.w.item(), "tau_match": V.model.log_tau.exp().item(), "gate": V.model.gate.item()},
        "steps": [
            {"id": "raw", "title": "1 · Dữ liệu gốc", "who": "Yến / Thắng",
             "items": [["Ảnh VG ∩ COCO", len(b.image_ids)], ["Caption COCO", len(b.captions)],
                       ["Ảnh train / val / test", f"{len(b.splits['train'])} / {len(b.splits['val'])} / {len(b.splits['test'])}"],
                       ["Caption train / val / test", f"{split_caps['train']} / {split_caps['val']} / {split_caps['test']}"]]},
            {"id": "clip", "title": "2 · Đặc trưng CLIP", "who": "Thắng",
             "items": [["Mô hình", "ViT-B-32 / laion2b (đóng băng)"], ["Chiều vector", b.image_emb.shape[1]],
                       ["Ứng viên mỗi truy vấn", f"top-{K}"]]},
            {"id": "graph", "title": "3 · Đồ thị (biến thể đang chọn)", "who": "Huy",
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
                       ["Điểm cuối", f"α·z(cos CLIP) + (1−α)·z(điểm mô hình), α = {V.alpha}"]]},
        ],
        "label_degree_top": [[g.labels[l], c] for l, c in V.img_count.most_common(15)],
        "rel_hist": Counter(r.split("⁻")[0] for r in g.rel_name.values()).most_common(12),
    }


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
    variant: str | None = None          # biến thể A
    compare: str | None = None          # biến thể B (tùy chọn)


def _query(req):
    """→ (text, q, concepts, gold_row, group)"""
    if req.caption_id is not None:
        c = S.cap_by_id.get(req.caption_id)
        if c is None:
            raise HTTPException(404, "caption_id không có")
        return (c["text"], S.b.caption_emb[S.cap_row[req.caption_id]],
                req.concepts if req.concepts is not None else c["concepts"], S.row_of[c["image_id"]], c.get("group"))
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Nhập câu truy vấn")
    text = req.text.strip()
    return text, S.encode_text(text), (req.concepts if req.concepts is not None else S.extract_concepts(text)), None, None


def _rank(sc):
    o = np.argsort(-sc, kind="stable")
    r = np.empty(len(o), int)
    r[o] = np.arange(1, len(o) + 1)
    return r


def _score(V: Variant, q, cand, sim, concepts):
    cids = V.concept_ids(concepts)
    with torch.no_grad():
        qa = F.normalize(V.model.q_proj(torch.from_numpy(q)[None]), dim=-1)[0]
        ad = (V.img_ad[torch.from_numpy(cand)] @ qa).numpy()
        if cids:
            M = V.model.concept_image_match(V.H, torch.tensor(cids), V.qd.img_labels[torch.from_numpy(cand)]).numpy()
            gs = M.mean(0)
        else:
            M, gs = np.zeros((0, len(cand))), np.zeros(len(cand))
    total = ad + V.model.w.item() * gs
    fused = S.ev.fuse(sim[None], total[None], V.alpha)[0]
    return {"cids": cids, "ad": ad, "gs": gs, "M": M, "fused": fused, "r_graph": _rank(gs), "r_final": _rank(fused)}


def _cands(q, split):
    pool = S.pool[split]
    sims = S.b.image_emb[pool] @ q
    order = np.argsort(-sims)[:K]
    return pool, pool[order], sims[order]


@app.post("/api/search")
def search(req: SearchReq):
    text, q, concepts, gold, group = _query(req)
    pool, cand, sim = _cands(q, req.split)
    r_clip = _rank(sim)
    out_vars = {}
    for slot, vid in (("A", req.variant), ("B", req.compare)):
        if slot == "B" and not vid:
            continue
        V = S.var(vid)
        sc = _score(V, q, cand, sim, concepts)

        def rows(key, V=V, sc=sc):
            rk = sc[key]
            o = np.argsort(rk)[: req.k]
            return [S.img_card(int(cand[j]), V, rank=int(rk[j]), clip_rank=int(r_clip[j]), final_rank=int(sc["r_final"][j]),
                               clip=float(sim[j]), adapter=float(sc["ad"][j]), graph=float(sc["gs"][j]),
                               fused=float(sc["fused"][j]), is_gold=bool(gold is not None and cand[j] == gold),
                               variant=V.id) for j in o]

        hit = np.where(cand == gold)[0] if gold is not None else []
        out_vars[slot] = {"variant": V.info(), "w": V.model.w.item(), "alpha": V.alpha,
                          "concepts": [V.g.labels[i] for i in sc["cids"]],
                          "concepts_unknown": [n for n in concepts if n not in V.label_idx],
                          "final": rows("r_final"), "graph": rows("r_graph"),
                          "gold_rank": int(sc["r_final"][hit[0]]) if len(hit) else None,
                          "gold_graph_rank": int(sc["r_graph"][hit[0]]) if len(hit) else None}
    VA = S.var(req.variant)
    clip_rows = [S.img_card(int(cand[j]), VA, rank=int(r_clip[j]), clip_rank=int(r_clip[j]), final_rank=None,
                            clip=float(sim[j]), adapter=None, graph=None, fused=None,
                            is_gold=bool(gold is not None and cand[j] == gold), variant=None)
                 for j in np.argsort(r_clip)[: req.k]]
    hit = np.where(cand == gold)[0] if gold is not None else []
    gold_info = None if gold is None else {"row": int(gold), "in_top_k": bool(len(hit)),
                                           "clip_rank": int(r_clip[hit[0]]) if len(hit) else None,
                                           "card": S.img_card(int(gold), VA)}
    return {"text": text, "group": group, "concepts_input": concepts, "pool": req.split, "pool_size": int(len(pool)),
            "K": K, "clip": clip_rows, "variants": out_vars, "gold": gold_info}


@app.post("/api/gold_all")
def gold_all(req: SearchReq):
    """Ảnh đúng xếp hạng mấy ở MỌI biến thể đã nạp (chỉ với caption có ảnh đúng)."""
    text, q, concepts, gold, group = _query(req)
    if gold is None:
        raise HTTPException(400, "Chỉ dùng được với caption có ảnh đúng")
    pool, cand, sim = _cands(q, req.split)
    hit = np.where(cand == gold)[0]
    rows = [{"id": "clip", "title": "CLIP (baseline của Thắng)", "rank": int(_rank(sim)[hit[0]]) if len(hit) else None}]
    for v in S.variants.values():
        if not v.loaded:
            rows.append({"id": v.id, "title": v.title, "note": v.note, "rank": None, "pending": True})
            continue
        sc = _score(v, q, cand, sim, concepts)
        rows.append({"id": v.id, "title": v.title, "note": v.note, "alpha": v.alpha,
                     "rank": int(sc["r_final"][hit[0]]) if len(hit) else None,
                     "graph_rank": int(sc["r_graph"][hit[0]]) if len(hit) else None,
                     "top1": int(cand[np.argmin(sc["r_final"])])})
    return {"text": text, "group": group, "in_top_k": bool(len(hit)), "K": K, "rows": rows}


# ---------------------------------------------------------------- giải thích + đồ thị con của một ảnh
class ExplainReq(BaseModel):
    row: int
    concepts: list[str] = []
    max_other_labels: int = 40
    variant: str | None = None


@app.post("/api/explain")
def explain(req: ExplainReq):
    V = S.var(req.variant)
    row = req.row
    labs = V.labels_of(row)
    cids = V.concept_ids(req.concepts)
    n_img = V.g.n_img
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
    tau = V.model.log_tau.exp().item()
    with torch.no_grad():
        for c in cids:
            u = F.normalize(V.model.c_proj(V.H[c]), dim=-1)
            cos = (V.H[torch.tensor(labs)] @ u).numpy() if labs else np.zeros(0)
            if len(cos):
                ex = np.exp((cos - cos.max()) / tau)
                wsm = ex / ex.sum()
                m = float(tau * np.log(ex.sum()) + cos.max())
            else:
                wsm, m = cos, 0.0
            top = np.argsort(-cos)[:3]
            path = _two_hop(V, c, labs[top[0]]) if len(top) and labs[top[0]] != c else None
            matches.append({"concept": V.g.labels[c], "in_image": c in labs, "m": m,
                            "top": [{"label": V.g.labels[labs[j]], "cos": float(cos[j]), "weight": float(wsm[j])}
                                    for j in top], "path": path})
    hot = {V.label_idx[mt["top"][0]["label"]] for mt in matches if mt["top"]}
    show = list(dict.fromkeys([*hot, *labs]))[: len(hot) + req.max_other_labels]
    for l in show:
        add_node(f"lab:{l}", V.g.labels[l], "label" if l < S.n_vocab else "concept", hot=l in hot)
        add_edge(f"lab:{l}", img_id, "contains", hot=l in hot)
    for mt, c in zip(matches, cids):
        qid = f"q:{c}"
        add_node(qid, V.g.labels[c], "query")
        if mt["top"]:
            best = V.label_idx[mt["top"][0]["label"]]
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
        for s, t in V.lab_nbrs.get(l, []):
            if s in shown and s < l and extra < 80:
                add_edge(f"lab:{s}", f"lab:{l}", TYPE_NAMES[t], rel=V.g.rel_name.get((s + n_img, l + n_img), ""))
                extra += 1
    return {"variant": V.info(), "card": S.img_card(row, V), "matches": matches,
            "graph_score": float(np.mean([mt["m"] for mt in matches])) if matches else 0.0,
            "elements": {"nodes": list(nodes.values()), "edges": edges},
            "counts": {"labels_total": len(labs), "labels_shown": len(show), "label_label_edges": extra}}


def _two_hop(V: Variant, c: int, target: int):
    """Đường ngắn nhất ≤ 2 bước c → x → target trên đồ thị nhãn (sg_rel / ConceptNet), để minh họa câu gián tiếp."""
    name = V.g.labels.__getitem__
    direct = [t for s, t in V.lab_nbrs.get(target, []) if s == c]
    if direct:
        return {"nodes": [{"idx": c, "name": name(c)}, {"idx": target, "name": name(target), "edge": TYPE_NAMES[direct[0]]}]}
    into_t = {s: t for s, t in V.lab_nbrs.get(target, [])}
    for s, t in V.lab_nbrs.get(c, []):  # đồ thị nhãn hai chiều
        if s in into_t:
            return {"nodes": [{"idx": c, "name": name(c)}, {"idx": s, "name": name(s), "edge": TYPE_NAMES[t]},
                              {"idx": target, "name": name(target), "edge": TYPE_NAMES[into_t[s]]}]}
    return None


# ---------------------------------------------------------------- graph builder: đầu vào → đầu ra của MỘT ảnh
@app.get("/api/images")
def images(split: str = "test", label: str = "", n: int = 24, seed: int = 0, variant: str | None = None):
    V = S.var(variant)
    rows = [S.row_of[i] for i in S.b.splits.get(split, [])]
    if label:
        l = V.label_idx.get(normalize_label(label, {}))
        if l is None:
            raise HTTPException(404, f"Không có nhãn '{label}'")
        rows = [r for r in rows if bool((V.qd.img_labels[r] == l).any())]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False) if rows else []
    return {"total": len(rows), "items": [S.img_card(int(rows[i]), V) for i in pick]}


@app.get("/api/build/{row}")
def build(row: int, two_hop: int = 0, variant: str | None = None):
    """Tái hiện build_graph cho một ảnh: objects/relations gốc → nhãn giữ/bỏ → cạnh contains, sg_rel, ConceptNet."""
    V = S.var(variant)
    if not 0 <= row < V.g.n_img:
        raise HTTPException(404, "row ngoài phạm vi")
    iid = S.b.image_ids[row]
    m = S.meta[iid]
    vocab, n_img = V.label_idx, V.g.n_img
    is_train = m["split"] == "train"
    gvar = (V.cfg.get("graph") or {}).get("variant", "full")
    objs = Counter(m["objects"])
    objects = []
    for o, c in objs.most_common():
        nl = normalize_label(o, {})
        objects.append({"raw": o, "norm": nl, "count": c, "kept": nl in vocab,
                        "images_with_label": V.img_count.get(vocab[nl], 0) if nl in vocab else 0})
    labs = V.labels_of(row)
    labset = set(labs)
    rels = []
    for s, p, o in m["relations"]:
        ns, no = normalize_label(s, {}), normalize_label(o, {})
        both = ns in vocab and no in vocab and ns != no
        in_graph = both and (vocab[ns] + n_img, vocab[no] + n_img) in V.g.rel_name
        why = ("dùng (ảnh train) — gộp với các ảnh train khác" if is_train and in_graph else
               "ảnh không thuộc train → không đóng góp cạnh (chống rò rỉ)" if not is_train and both else
               "một đầu không thuộc bộ nhãn → bỏ" if not both else
               "biến thể này không có cạnh sg_rel / bị lọc bởi min_rel_count")
        rels.append({"s": s, "p": p, "o": o, "used": bool(is_train and in_graph), "edge_exists": bool(in_graph),
                     "graph_rel": V.g.rel_name.get((vocab[ns] + n_img, vocab[no] + n_img), "") if both else "",
                     "why": why})
    nodes, edges, seen = {}, [], set()
    img_id = f"img:{row}"
    nodes[img_id] = {"id": img_id, "label": f"ảnh #{iid}", "kind": "image"}
    contains = []
    for l in labs:
        nodes[f"lab:{l}"] = {"id": f"lab:{l}", "label": V.g.labels[l], "kind": "label"}
        edges.append({"source": f"lab:{l}", "target": img_id, "kind": "contains"})
        contains.append({"label": V.g.labels[l], "images_with_label": V.img_count.get(l, 0)})
    among, concept, outside, seen_out = [], [], Counter(), set()
    for l in labs:
        for s, t in V.lab_nbrs.get(l, []):
            key = (min(s, l), max(s, l), t)
            if s in labset:
                if key in seen:
                    continue
                seen.add(key)
                rel = V.g.rel_name.get((s + n_img, l + n_img), "")
                among.append({"a": V.g.labels[s], "b": V.g.labels[l], "type": TYPE_NAMES[t], "rel": rel})
                edges.append({"source": f"lab:{s}", "target": f"lab:{l}", "kind": TYPE_NAMES[t], "rel": rel})
            else:
                if (s, t, l) not in seen_out:  # đếm số NHÃN của ảnh mà s nối tới (cạnh sg_rel có thể bị trùng)
                    seen_out.add((s, t, l))
                    outside[(s, t)] += 1
                if t == CONCEPT and key not in seen:
                    seen.add(key)
                    concept.append({"label": V.g.labels[l], "rel": V.g.rel_name.get((s + n_img, l + n_img), ""),
                                    "concept": V.g.labels[s]})
    hop2 = [{"name": V.g.labels[s], "type": TYPE_NAMES[t], "links": c} for (s, t), c in outside.most_common(30)]
    if two_hop:
        for h in hop2[: int(two_hop)]:
            s = V.label_idx[h["name"]]
            nid = f"lab:{s}"
            nodes[nid] = {"id": nid, "label": h["name"], "kind": "concept" if s >= S.n_vocab else "hop2"}
            for l in labs:
                if any(x == s for x, _ in V.lab_nbrs.get(l, [])):
                    edges.append({"source": nid, "target": f"lab:{l}", "kind": h["type"]})
    kept = sum(1 for o in objects if o["kept"])
    return {"variant": V.info(), "graph_variant": gvar, "card": S.img_card(row, V), "is_train": is_train,
            "stats": {"objects_raw": sum(objs.values()), "objects_unique": len(objs), "labels_kept": kept,
                      "objects_dropped": len(objs) - kept, "relations_raw": len(rels),
                      "relations_used": sum(r["used"] for r in rels), "contains": len(contains),
                      "label_label_edges": len(among), "concept_edges": len(concept), "hop2_labels": len(outside)},
            "objects": objects, "relations": rels, "contains": contains, "among": among, "concept": concept,
            "hop2": hop2, "elements": {"nodes": list(nodes.values()), "edges": edges},
            "min_rel_count": int((V.cfg.get("graph") or {}).get("min_rel_count", 1))}


# ---------------------------------------------------------------- khám phá một nhãn
@app.get("/api/label/{name}")
def label(name: str, variant: str | None = None):
    V = S.var(variant)
    l = V.label_idx.get(normalize_label(name, {}))
    if l is None:
        raise HTTPException(404, f"Không có nhãn '{name}' trong biến thể này")
    n = V.g.n_img
    nb, seen = [], set()
    for s, t in V.lab_nbrs.get(l, []):
        if (s, t) in seen:  # build_graph thêm cạnh sg_rel 2 lần khi có cả (a,b) và (b,a) — hiển thị một lần
            continue
        seen.add((s, t))
        nb.append({"name": V.g.labels[s], "type": TYPE_NAMES[t], "rel": V.g.rel_name.get((s + n, l + n), ""),
                   "images": V.img_count.get(s, 0)})
    nb.sort(key=lambda x: -x["images"])
    with torch.no_grad():
        sim = (V.H @ V.H[l]).numpy()
    close = [(V.g.labels[i], float(sim[i])) for i in np.argsort(-sim)[1:11]]
    return {"variant": V.info(), "name": V.g.labels[l], "kind": "label" if l < S.n_vocab else "concept",
            "images": V.img_count.get(l, 0), "neighbors": nb[:60], "n_neighbors": len(nb), "nearest_after_gnn": close}


@app.get("/api/labels")
def labels(q: str = "", variant: str | None = None):
    V = S.var(variant)
    q = q.lower().strip()
    return [l for l in V.g.labels if q in l][:20]


# ---------------------------------------------------------------- ablation
STORY = [
    ("t1_clip", "", "CLIP (baseline của Thắng)", "Không học gì; cos(câu, ảnh).", "main"),
    ("t1_graph", "seed0-dau-tien", "Legacy GNN (#2)", "Gộp nhãn + lớp chiếu; tưởng đồ thị giúp +2.8 MRR.", "history"),
    ("t1_ablation_no_edge", "seed0", "Legacy · no_edge (#3)", "Bỏ hết cạnh vẫn +2.8 ⇒ mức tăng là của adapter.", "history"),
    ("t1_ablation_shuffle", "seed0", "Legacy · shuffle (#3)", "Xáo cạnh vẫn như cũ ⇒ đồ thị không được dùng.", "history"),
    ("t1_graph", "split-seed0", "Split: Adapter + Graph gộp (#4)", "Nhánh đồ thị tách rời: không thêm gì.", "history"),
    ("t1_graph", "split-noadapter-seed0", "Split: Graph trên CLIP thô (#4)", "+0.4 MRR.", "history"),
    ("t1_qa", "untrained-noadapter-seed0", "QA: khớp nhãn không học (#5)", "Gần như không giúp.", "history"),
    ("t1_qa_ablation_no_edge", "kaggle-3seed", "CLIP + Adapter (không đồ thị)", "Mốc so sánh đúng.", "main"),
    ("t1_qa", "kaggle-3seed", "+ QA-Graph (contains + sg_rel)", "Khái niệm ↔ nhãn của ảnh.", "main"),
    ("t1_qa_ablation_no_rel", "kaggle-3seed", "+ QA-Graph (chỉ contains)", "Bỏ cạnh nhãn–nhãn.", "main"),
    ("t1_qa_cn", "kaggle-3seed", "+ QA-Graph + ConceptNet", "contains + sg_rel + CN.", "main"),
    ("t1_qa_cn_no_sg", "kaggle-3seed", "+ QA-Graph + ConceptNet, bỏ sg_rel", "contains + CN.", "main"),
    ("t1_qa_cn_shuffle", "kaggle-3seed", "Đối chứng: xáo cạnh", "Phải tụt dưới Adapter.", "main"),
    ("t1_qa_cn_gcn", "kaggle-3seed", "contains + sg_rel + CN · GCN", "", "gnn"),
    ("t1_qa_cn_sage", "kaggle-3seed", "contains + sg_rel + CN · SAGE", "", "gnn"),
    ("t1_qa_cn_no_sg_gcn", "kaggle-3seed", "contains + CN · GCN", "", "gnn"),
    ("t1_qa_cn_no_sg_sage", "kaggle-3seed", "contains + CN · SAGE", "", "gnn"),
]


def _clip_row():
    """Dòng CLIP tính lại bằng chính Protocol của Thắng (3 seed + toàn pool)."""
    if S._clip_baseline is None:
        P = S.ev.Protocol.load(S.bundle)
        res = P.evaluate(lambda c, p: c["sim"], name="CLIP", split="test", seeds=[0, 1, 2], full=True)
        ms, fl = res["mean_std"], res["full"]
        bg = {}
        for gname in ("direct", "indirect", "no_concept"):
            vals = [r["by_group"][gname]["MRR"] for r in res["per_seed"] if gname in r.get("by_group", {})]
            if vals:
                bg[gname] = float(np.mean(vals))
        mean = lambda k: (ms.get(k) or {}).get("mean") if isinstance(ms.get(k), dict) else ms.get(k)  # noqa: E731
        S._clip_baseline = {"exp": "t1_clip", "tag": "", "run_id": "protocol (3 seed)",
                            "full": {k: fl.get(k) for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10")},
                            "full_std": {}, "gallery": {k: mean(k) for k in ("R@1", "MRR")},
                            "by_group": bg, "alpha": None, "history": [], "alpha_curve": [], "seeds": 3}
    return S._clip_baseline


@app.get("/api/ablation")
def ablation():
    runs = [_clip_row()]
    for f in sorted((ROOT / "results").glob("t1_*/*.json")):
        d = json.loads(f.read_text())
        r, m = d.get("run", {}), d.get("metrics", {})
        if not m.get("test_full") or "smoke" in (r.get("tag") or ""):
            continue
        exp, tag = r.get("exp"), r.get("tag") or ""
        hist, curve = [], []
        ti = ROOT / "runs" / exp / r.get("run_id", "") / "train_info.json"
        if ti.exists():
            info = json.loads(ti.read_text())
            if info:
                hist, curve = info[0].get("history", []), info[0].get("alpha_curve", [])

        def g(key, block="test_full", stat="mean"):
            v = m.get(block, {}).get(key)
            return v.get(stat) if isinstance(v, dict) else v

        runs.append({"exp": exp, "tag": tag, "run_id": r.get("run_id"),
                     "full": {k: g(k) for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10")},
                     "full_std": {k: g(k, stat="std") for k in ("R@1", "MRR")},
                     "gallery": {k: g(k, "test") for k in ("R@1", "MRR")},
                     "by_group": {gn: (v.get("MRR") or {}).get("mean") for gn, v in (m.get("by_group") or {}).items()},
                     "alpha": (m.get("alpha") or {}).get("values"), "history": hist, "alpha_curve": curve,
                     "seeds": len((m.get("alpha") or {}).get("values") or [])})
    story = []
    for exp, tag, title, note, section in STORY:
        hit = [x for x in runs if x["exp"] == exp and x["tag"] == tag]
        if hit:
            story.append({**hit[-1], "title": title, "note": note, "section": section})
    return {"story": story, "all": runs}


# ---------------------------------------------------------------- tĩnh
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
