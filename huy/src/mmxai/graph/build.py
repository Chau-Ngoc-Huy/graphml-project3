"""Dựng đồ thị tĩnh (PyG Data) từ Bundle.

Đỉnh: [0, N_img) là ảnh, [N_img, N_img + N_lab) là nhãn/khái niệm (đặc trưng = CLIP ảnh / CLIP text của tên nhãn).
Cạnh CÓ HƯỚNG (src → dst, tin nhắn chảy theo hướng này):
  type 0 CONTAINS  nhãn → ảnh            mọi ảnh (đây là dữ liệu của chính ứng viên, không phải thống kê)
  type 1 SG_REL    nhãn ↔ nhãn           gộp từ scene graph của ẢNH TRAIN (thống kê → chỉ dùng train, đề mục 6)
  type 2 CONCEPT   nhãn ↔ khái niệm      ConceptNet/Wikidata (tri thức ngoài, không phụ thuộc split)
Ảnh không bao giờ gửi tin → đặc trưng ảnh val/test không lan sang đỉnh khác, không ảnh hưởng tới huấn luyện.

Biến thể cho ablation (cfg.graph.variant):
  full      đủ 3 loại cạnh
  no_rel    chỉ giữ CONTAINS (bỏ quan hệ scene graph + khái niệm)
  no_concept bỏ CONCEPT
  no_sg     bỏ SG_REL, giữ CONCEPT (chỉ tri thức ConceptNet giữa các nhãn; JOURNAL #6–#7: sg_rel không giúp)
  no_edge   không cạnh nào (GNN chỉ còn self-loop ≈ MLP trên đặc trưng ảnh) → tách tác dụng của lớp chiếu khỏi đồ thị
  shuffle   xáo đầu mút cạnh, giữ bậc (ảnh giữ số nhãn, nhãn giữ số lần xuất hiện)
  random    đồ thị ngẫu nhiên cùng số cạnh mỗi loại
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from ..data.contract import Bundle

CONTAINS, SG_REL, CONCEPT = 0, 1, 2
TYPE_NAMES = {CONTAINS: "contains", SG_REL: "sg_rel", CONCEPT: "concept"}
VARIANTS = ("full", "no_rel", "no_concept", "no_sg", "no_edge", "shuffle", "random")


@dataclass
class Graph:
    data: "torch_geometric.data.Data"  # x [N, D], edge_index [2, E], edge_type [E]
    n_img: int
    labels: list[str]
    img_row: dict[int, int]            # image_id -> node index
    rel_name: dict[tuple[int, int], str]  # (src, dst) -> tên quan hệ hay gặp nhất (để giải thích)
    stats: dict

    def node_name(self, i: int) -> str:
        return f"image:{i}" if i < self.n_img else self.labels[i - self.n_img]


def build_graph(b: Bundle, variant: str = "full", min_rel_count: int = 1, seed: int = 0) -> Graph:
    from torch_geometric.data import Data

    if variant not in VARIANTS:
        raise ValueError(f"graph.variant phải thuộc {VARIANTS}, nhận {variant!r}")
    n_img = len(b.image_ids)
    lab_idx = {l: n_img + i for i, l in enumerate(b.labels)}
    img_row = {iid: r for r, iid in enumerate(b.image_ids)}
    src, dst, typ = [], [], []
    rel_name: dict[tuple[int, int], str] = {}
    missing = Counter()

    for iid, labs in b.image_labels.items():
        for l in set(labs):
            if l in lab_idx:
                src.append(lab_idx[l]); dst.append(img_row[iid]); typ.append(CONTAINS)
            else:
                missing[l] += 1

    train = set(b.splits["train"])
    pair_pred: dict[tuple[int, int], Counter] = {}
    for iid in train:
        for s, p, o in b.image_rels.get(iid, []):
            if s in lab_idx and o in lab_idx and s != o:
                pair_pred.setdefault((lab_idx[s], lab_idx[o]), Counter())[p] += 1
    for (u, v), cnt in pair_pred.items():
        if sum(cnt.values()) >= min_rel_count:
            p = cnt.most_common(1)[0][0]
            for a, c in ((u, v), (v, u)):
                src.append(a); dst.append(c); typ.append(SG_REL)
            rel_name[(u, v)] = p
            rel_name[(v, u)] = f"{p}⁻¹"

    for h, r, t in b.concept_edges:
        if h in lab_idx and t in lab_idx and h != t:
            u, v = lab_idx[h], lab_idx[t]
            for a, c in ((u, v), (v, u)):
                src.append(a); dst.append(c); typ.append(CONCEPT)
            rel_name[(u, v)] = r
            rel_name[(v, u)] = f"{r}⁻¹"

    ei = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    et = torch.tensor(typ, dtype=torch.long)
    ei, et = _apply_variant(ei, et, variant, n_img, len(b.labels), seed)

    x = torch.from_numpy(np.concatenate([b.image_emb, b.label_emb], 0))
    data = Data(x=x, edge_index=ei, edge_type=et, num_nodes=x.shape[0])
    data.img_mask = torch.arange(x.shape[0]) < n_img
    stats = {"n_img": n_img, "n_label": len(b.labels), "variant": variant,
             **{f"E_{TYPE_NAMES[t]}": int((et == t).sum()) for t in TYPE_NAMES},
             "missing_labels": len(missing), "missing_top": [l for l, _ in missing.most_common(10)]}
    if variant in ("shuffle", "random"):
        rel_name = {}
    return Graph(data, n_img, b.labels, img_row, rel_name, stats)


def _apply_variant(ei, et, variant, n_img, n_lab, seed):
    if variant == "full":
        return ei, et
    if variant == "no_edge":
        return ei[:, :0], et[:0]
    if variant == "no_rel":
        m = et == CONTAINS
        return ei[:, m], et[m]
    if variant == "no_concept":
        m = et != CONCEPT
        return ei[:, m], et[m]
    if variant == "no_sg":
        m = et != SG_REL
        return ei[:, m], et[m]
    g = torch.Generator().manual_seed(seed)
    ei = ei.clone()
    for t in (CONTAINS, SG_REL, CONCEPT):
        m = (et == t).nonzero().flatten()
        if len(m) == 0:
            continue
        if variant == "shuffle":   # hoán vị đầu nguồn trong cùng loại → giữ phân bố bậc ra/vào
            ei[0, m] = ei[0, m[torch.randperm(len(m), generator=g)]]
        else:                      # random: nguồn là nhãn ngẫu nhiên; đích giữ loại (ảnh cho CONTAINS, nhãn cho loại khác)
            ei[0, m] = n_img + torch.randint(0, n_lab, (len(m),), generator=g)
            if t != CONTAINS:
                ei[1, m] = n_img + torch.randint(0, n_lab, (len(m),), generator=g)
    return ei, et
