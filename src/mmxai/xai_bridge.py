"""Cầu nối mô hình QA-Graph đã huấn luyện (Huy) ↔ khung xAI của Dương (`duong/xai`, Nhiệm vụ 3).

Dương cần (duong/01_XAI_Explainability_Plan.md §8): graph schema, graph builder, checkpoint và một hàm forward nhận đồ thị đã
bị xóa/mask cạnh. Module này cung cấp đúng hai thứ khung của Dương dùng:

    bridge = QAGraphBridge("runs/psg_qa_cn/<run>", "data/psg_bundle")
    cases  = bridge.cases(caption_ids, k=20)                 # truy vấn + ảnh đúng + top-k ứng viên CLIP (giao thức chung)
    kg     = bridge.export_graph(cases)                      # dict JSON theo duong/xai/io.load_graph (nodes/edges)
    scorer = CallableScorerAdapter(bridge.score_fn, bridge.local_edges_fn)   # của Dương

score_fn chạy LẠI mô hình trên đồ thị nhận vào — không có điểm tính sẵn:
  - cạnh `contains` (nhãn → ảnh) quyết định danh sách nhãn của ảnh ứng viên (img_labels);
  - cạnh nhãn–nhãn (`sg_rel`, `concept`) là đầu vào của GNN → H = label_h(X, cạnh còn lại) được tính lại mỗi lần;
  - export_graph đưa ra các cạnh nhãn–nhãn đi THẲNG vào (nhãn của ứng viên ∪ khái niệm của câu) — bước cuối của mọi đường
    khái niệm → nhãn → ảnh. Cạnh xa hơn vẫn nằm trong mô hình và luôn được giữ khi chấm (không bị coi là xóa); cạnh đã export
    mà vắng trong đồ thị nhận vào thì bị xóa thật khỏi đầu vào GNN. (Vùng nhìn đủ 2 bước gần như cả đồ thị nhãn — VG ~128k
    cạnh — quá lớn cho bỏ-từng-cạnh.)
  - final_score = fuse(cos CLIP, adapter + w·s_G, α) — đúng công thức xếp hạng của giao thức (z-score trong tập ứng viên).

Mã nút: ảnh = "img:<image_id>" (external_id = image_id, ứng viên truyền vào là str(image_id)); nhãn/khái niệm = "lab:<tên>".
Mã cạnh: "lab:<k>" (k = chỉ số trong đồ thị nhãn của QAData), "contains:<image_id>:<k_nhãn>". Kiểu nút/quan hệ khớp
duong/configs/graph_schema.json (image, concept; contains, sg_rel, concept).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.contract import normalize_label
from .data.bundle import load_bundle
from .experiments.vg_graph import load_eval_module
from .graph.build import CONTAINS, TYPE_NAMES, build_graph
from .graph.qa import QAData, QAScorer

STOP = {"a", "the", "of", "on", "in"}


class QAGraphBridge:
    def __init__(self, run_dir: str | Path, bundle_dir: str | Path, device: str = "cpu"):
        self.run, self.bundle = Path(run_dir), Path(bundle_dir)
        self.cfg = json.loads((self.run / "config.json").read_text())
        self.ev = load_eval_module(self.bundle)
        extra = self.cfg.get("extra")
        b = load_bundle(self.bundle, (self.bundle / Path(extra).name) if extra else None)
        gc = self.cfg.get("graph", {})
        self.g = build_graph(b, gc.get("variant", "full"), int(gc.get("min_rel_count", 1)),
                             rel_merge=gc.get("rel_merge", "directed"))
        self.b = b
        self.qd = QAData(self.g, {c["caption_id"]: c["concepts"] for c in b.captions})
        self.model = QAScorer(b.image_emb.shape[1], **self.cfg.get("gnn", {}))
        self.model.load_state_dict(torch.load(self.run / "model.pt", map_location="cpu"))
        self.model.eval()
        self.alpha = float(json.loads((self.run / "train_info.json").read_text())[0]["alpha"])
        self.x_img = torch.from_numpy(b.image_emb)
        self.row_of = {iid: r for r, iid in enumerate(b.image_ids)}
        self.lab_idx = {l: k for k, l in enumerate(self.g.labels)}
        n = self.g.n_img
        ei, et = self.g.data.edge_index, self.g.data.edge_type
        lab = (ei[0] >= n) & (ei[1] >= n)
        self.lab_type = et[lab].tolist()                            # cùng thứ tự với qd.ei_lab
        self.lab_src, self.lab_dst = self.qd.ei_lab[0].tolist(), self.qd.ei_lab[1].tolist()
        self._src_np, self._dst_np = np.asarray(self.lab_src), np.asarray(self.lab_dst)
        self._q_cache: dict[str, tuple[np.ndarray, list[int]]] = {}
        self._scope = None
        self.into: dict[int, list[int]] = {}
        for k, d in enumerate(self.lab_dst):
            self.into.setdefault(d, []).append(k)
        self.hops = len(self.model.convs)
        self.cap_by_text = {c["text"].strip().lower(): c for c in b.captions}
        self.cap_row = {c["caption_id"]: r for r, c in enumerate(b.captions)}
        self._clip = None

    # ------------------------------------------------------------------ truy vấn
    def _encode(self, text: str) -> np.ndarray:
        if self._clip is None:
            import open_clip

            meta = json.loads((self.bundle / "features" / "meta.json").read_text())
            m, _, _ = open_clip.create_model_and_transforms(meta["model"], pretrained=meta["pretrained"])
            self._clip = (m.eval(), open_clip.get_tokenizer(meta["model"]))
        m, tok = self._clip
        with torch.no_grad():
            return F.normalize(m.encode_text(tok([text])).float(), dim=-1)[0].numpy()

    def _concepts(self, text: str) -> list[str]:
        """Câu tự do: khớp 1–3 từ liên tiếp với từ vựng (giống UI); caption có sẵn dùng trường concepts của bundle."""
        words, out, i = re.findall(r"[a-z]+", text.lower()), [], 0
        while i < len(words):
            hit = None
            for n in (3, 2, 1):
                ph = " ".join(words[i:i + n])
                for cand in (ph, ph[:-1] if ph.endswith("s") else None, ph[:-2] if ph.endswith("es") else None):
                    if cand and normalize_label(cand, {}) in self.lab_idx:
                        hit = (normalize_label(cand, {}), n)
                        break
                if hit:
                    break
            if hit and hit[0] not in out and hit[0] not in STOP:
                out.append(hit[0])
            i += hit[1] if hit else 1
        return out

    def query(self, text: str) -> tuple[np.ndarray, list[int]]:
        """(vector CLIP của câu, chỉ số đỉnh khái niệm). Caption của bundle: dùng đặc trưng + khái niệm đã trích sẵn."""
        if text in self._q_cache:
            return self._q_cache[text]
        c = self.cap_by_text.get(text.strip().lower())
        if c is not None:
            q, names = self.b.caption_emb[self.cap_row[c["caption_id"]]].astype(np.float32), c["concepts"]
        else:
            q, names = self._encode(text), self._concepts(text)
        out = (q, list(dict.fromkeys(self.lab_idx[x] for x in names if x in self.lab_idx)))
        self._q_cache[text] = out
        return out

    def cases(self, caption_ids, k: int = 20, split: str = "test") -> list[dict]:
        """Truy vấn từ caption của bundle: ứng viên = top-k CLIP trong pool split (ảnh đúng được thêm nếu rơi ngoài top-k)."""
        pool = np.array([self.row_of[i] for i in self.b.splits[split]])
        out = []
        for cid in caption_ids:
            cap = self.b.captions[self.cap_row[int(cid)]]
            q, _ = self.query(cap["text"])
            sims = self.b.image_emb[pool] @ q
            top = [int(self.b.image_ids[r]) for r in pool[np.argsort(-sims)[:k]]]
            gold = int(cap["image_id"])
            if gold not in top:
                top[-1] = gold
            out.append({"query_id": f"cap{cid}", "query_text": cap["text"], "group": self._group(cap),
                        "image_id": str(gold), "candidate_image_ids": [str(i) for i in top]})
        return out

    def _group(self, cap) -> str:
        return next((c.get("group") for c in [cap] if c.get("group")), "unknown")

    # ------------------------------------------------------------------ đồ thị
    def _receptive(self, seeds: set[int]) -> set[int]:
        """Mọi cạnh nhãn–nhãn ảnh hưởng tới H của các đỉnh `seeds` (≤ self.hops bước ngược chiều cạnh)."""
        edges, front, seen = set(), set(seeds), set(seeds)
        for _ in range(self.hops):
            nxt = set()
            for d in front:
                for k in self.into.get(d, []):
                    edges.add(k)
                    s = self.lab_src[k]
                    if s not in seen:
                        nxt.add(s)
            seen |= nxt
            front = nxt
        return edges

    def export_graph(self, cases: list[dict]) -> dict:
        """Đồ thị con đủ để chấm mọi ứng viên của các truy vấn (JSON cho duong/xai/io.load_graph)."""
        imgs = sorted({int(i) for c in cases for i in c["candidate_image_ids"]})
        labs, contains = set(), []
        for iid in imgs:
            for l in self.qd.img_labels[self.row_of[iid]].tolist():
                if l >= 0:
                    labs.add(l)
                    contains.append((iid, l))
        for c in cases:
            labs |= set(self.query(c["query_text"])[1])
        lab_edges = {k for d in labs for k in self.into.get(d, [])}
        self._scope = np.zeros(len(self.lab_src), bool)
        self._scope[sorted(lab_edges)] = True
        for k in sorted(lab_edges):
            labs |= {self.lab_src[k], self.lab_dst[k]}
        n = self.g.n_img
        nodes = [{"id": f"img:{i}", "type": "image", "label": f"image {i}", "external_id": str(i)} for i in imgs]
        nodes += [{"id": f"lab:{self.g.labels[l]}", "type": "concept", "label": self.g.labels[l]} for l in sorted(labs)]
        edges = [{"id": f"contains:{i}:{l}", "source": f"lab:{self.g.labels[l]}", "target": f"img:{i}",
                  "relation": "contains", "provenance": "scene graph / panoptic labels of the image",
                  "support_status": "supported"} for i, l in contains]
        for k in sorted(lab_edges):
            s, d, t = self.lab_src[k], self.lab_dst[k], self.lab_type[k]
            rel = TYPE_NAMES[t]
            edges.append({"id": f"lab:{k}", "source": f"lab:{self.g.labels[s]}", "target": f"lab:{self.g.labels[d]}",
                          "relation": rel, "provenance": "train scene graphs" if rel == "sg_rel" else "ConceptNet",
                          "support_status": "supported" if rel == "sg_rel" else "uncertain",
                          "metadata": {"predicate": self.g.rel_name.get((s + n, d + n), rel)}})
        return {"nodes": nodes, "edges": edges,
                "meta": {"run": self.run.name, "bundle": self.bundle.name, "alpha": self.alpha, "gnn_layers": self.hops}}

    # ------------------------------------------------------------------ chấm điểm (forward thật trên đồ thị nhận vào)
    @staticmethod
    def _iid(node_id: str) -> int:
        return int(node_id.split(":", 1)[1])

    def _parse(self, graph):
        present, labels_of = set(graph.edges), {}
        for e in graph.edges.values():
            if e.relation == "contains":
                labels_of.setdefault(self._iid(e.target), []).append(self.lab_idx[e.source.split(":", 1)[1]])
        if self._scope is None:
            raise RuntimeError("gọi export_graph() trước khi chấm điểm")
        keep = ~self._scope                                                # ngoài phạm vi export: luôn giữ
        keep[[int(e[4:]) for e in present if e.startswith("lab:")]] = True  # trong phạm vi: giữ nếu còn trong đồ thị
        return labels_of, torch.from_numpy(keep)

    @torch.no_grad()
    def score_fn(self, query: str, candidate_image_ids, graph) -> dict:
        q, cids = self.query(query)
        labels_of, keep = self._parse(graph)
        H = self.model.label_h(self.qd.x_lab, self.qd.ei_lab[:, keep])
        iids = [int(i) for i in candidate_image_ids]
        rows = torch.tensor([self.row_of[i] for i in iids])
        L = max(1, max((len(labels_of.get(i, [])) for i in iids), default=1))
        il = torch.full((len(iids), L), -1, dtype=torch.long)
        for r, i in enumerate(iids):
            v = sorted(set(labels_of.get(i, [])))
            if v:
                il[r, : len(v)] = torch.tensor(v)
        if cids:
            gs = self.model.concept_image_match(H, torch.tensor(cids), il).mean(0).numpy()
        else:
            gs = np.zeros(len(iids))
        qz = F.normalize(self.model.q_proj(torch.from_numpy(q)[None]), dim=-1)[0]
        ad = (F.normalize(self.model.skip(self.x_img[rows]), dim=-1) @ qz).numpy()
        graph_s = self.model.w.item() * gs
        sim = self.b.image_emb[rows.numpy()] @ q
        fused = self.ev.fuse(sim[None], (ad + graph_s)[None], self.alpha)[0]
        return {str(i): {"clip_score": float(sim[r]), "graph_score": float(graph_s[r]), "final_score": float(fused[r])}
                for r, i in enumerate(iids)}

    def local_edges_fn(self, query: str, image_id: str, graph) -> list[str]:
        """Ứng viên giải thích của một ảnh: contains của ảnh + cạnh nhãn–nhãn đi THẲNG vào (nhãn của ảnh ∪ khái niệm của câu),
        tức bước cuối của đường khái niệm → nhãn → ảnh.

        Vùng nhìn đầy đủ của GNN 2 tầng gần như phủ cả đồ thị nhãn (PSG: ~13k cạnh) nên bỏ-từng-cạnh trên đó không khả thi;
        cạnh bước 2 vẫn có trong đồ thị và vẫn được tính khi chấm điểm, chỉ không được đề cử làm giải thích."""
        iid, present = int(image_id), set(graph.edges)
        own = [e.id for e in graph.edges.values() if e.relation == "contains" and self._iid(e.target) == iid]
        seeds = {self.lab_idx[graph.edges[e].source.split(":", 1)[1]] for e in own} | set(self.query(query)[1])
        first = sorted(k for d in seeds for k in self.into.get(d, []))
        return own + [f"lab:{k}" for k in first if f"lab:{k}" in present]
