"""Điểm đồ thị PHỤ THUỘC TRUY VẤN (arch=qa): khái niệm của câu ↔ nhãn của ảnh, so khớp trên nhúng nhãn đã qua GNN.

Vì sao (JOURNAL #4): gộp ~17 nhãn của ảnh thành MỘT vector (arch split) làm mất thông tin "nhãn nào khớp khái niệm nào";
khớp tên đơn giản còn hơn. Ở đây mỗi khái niệm của caption được so riêng với từng nhãn của ảnh:

    H        = normalize(X_nhãn + gate · GNN(X_nhãn, đồ thị nhãn))     GNN CHỈ chạy trên nhãn/khái niệm (sg_rel, ConceptNet)
    u_c      = normalize(C · H_c)                                       khái niệm c của câu (là một đỉnh nhãn), C khởi tạo I
    m(c, i)  = τ · logsumexp_{l ∈ nhãn(i)} cos(u_c, H_l) / τ            max mềm trên nhãn của ảnh i (cạnh contains)
    s_G(q,i) = trung bình_{c ∈ khái niệm(q)} m(c, i)
    score    = cos(P·q, S·x_i) [adapter, tùy chọn] + w · s_G(q, i)

- Câu trực tiếp: c là một nhãn của ảnh → cos(u_c, H_c) ≈ 1.
- Câu gián tiếp: c không có trong ảnh → điểm đến từ nhãn của ảnh có H gần H_c nhờ lan truyền qua cạnh sg_rel/ConceptNet
  (đường đi c → nhãn láng giềng → ảnh), chính là đường đi dùng để giải thích.
- Ablation đo đúng: no_edge (không contains) ⇒ s_G = 0 ⇒ chỉ adapter; no_rel (bỏ cạnh nhãn–nhãn, chưa ConceptNet) ⇒ "khớp
  tên học được" không có cấu trúc; shuffle ⇒ ảnh mang nhãn sai.
- Chống rò rỉ: GNN không chứa đỉnh ảnh; nhãn của ảnh val/test là thông tin quan sát (scene graph), không phải nhãn đích;
  cạnh nhãn–nhãn chỉ từ ảnh train (build_graph).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import CONTAINS


class QAData:
    """Chuẩn bị tensor từ Graph: đồ thị nhãn, danh sách nhãn mỗi ảnh (đệm −1), khái niệm mỗi caption (đệm −1)."""

    def __init__(self, g, cap_concepts: dict[int, list[str]], device="cpu"):
        ei, et, n = g.data.edge_index, g.data.edge_type, g.n_img
        lab = (ei[0] >= n) & (ei[1] >= n)
        self.x_lab = g.data.x[n:].to(device)
        self.ei_lab = (ei[:, lab] - n).to(device)
        c = et == CONTAINS
        src, dst = ei[0][c] - n, ei[1][c]
        lists = [[] for _ in range(n)]
        for s, d in zip(src.tolist(), dst.tolist()):
            lists[d].append(s)
        lists = [sorted(set(v)) for v in lists]
        lmax = max(1, max(len(v) for v in lists))
        self.img_labels = torch.full((n, lmax), -1, dtype=torch.long)
        for i, v in enumerate(lists):
            if v:
                self.img_labels[i, : len(v)] = torch.tensor(v, dtype=torch.long)
        self.img_labels = self.img_labels.to(device)
        idx = {l: k for k, l in enumerate(g.labels)}
        cmax = max([1] + [len(v) for v in cap_concepts.values()])
        n_cap = max(cap_concepts) + 1 if cap_concepts else 1
        self.cap_concepts = torch.full((n_cap, cmax), -1, dtype=torch.long)
        self.n_dropped = 0
        for cid, cs in cap_concepts.items():
            ids = list(dict.fromkeys(idx[x] for x in cs if x in idx))
            self.n_dropped += len(cs) - len(ids)
            if ids:
                self.cap_concepts[cid, : len(ids)] = torch.tensor(ids)
        self.cap_concepts = self.cap_concepts.to(device)
        self.stats = {"lab_edges": int(lab.sum()), "contains": int(c.sum()), "labels_per_img_max": lmax,
                      "imgs_without_labels": int(sum(1 for v in lists if not v)),
                      "caps_without_concepts": int((self.cap_concepts[:, 0] < 0).sum())}


class QAScorer(nn.Module):
    def __init__(self, dim: int, kind: str = "gat", hidden: int = 256, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1, adapter: bool = True, graph_weight: float = 1.0, tau_match: float = 0.1, **_):
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv

        self.kind = kind
        dims = [dim] + [hidden] * (layers - 1) + [dim]
        convs = []
        for a, b in zip(dims[:-1], dims[1:]):
            convs.append({"gcn": lambda: GCNConv(a, b), "sage": lambda: SAGEConv(a, b),
                          "gat": lambda: GATConv(a, b, heads=heads, concat=False, dropout=dropout)}[kind]())
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.skip = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        for lin in (self.c_proj, self.skip, self.q_proj):
            nn.init.eye_(lin.weight)
        self.adapter = adapter
        if not adapter:
            self.skip.weight.requires_grad_(False)
            self.q_proj.weight.requires_grad_(False)
        self.w = nn.Parameter(torch.tensor(float(graph_weight)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_match)))
        self.last_attention = None

    # ---- nhánh đồ thị
    def label_h(self, x, ei, keep_attention: bool = False):
        h, att = x, []
        for i, conv in enumerate(self.convs):
            if self.kind == "gat" and keep_attention:
                h, (e, a) = conv(h, ei, return_attention_weights=True)
                att.append((e, a.mean(-1) if a.dim() > 1 else a))
            else:
                h = conv(h, ei)
            if i < len(self.convs) - 1:
                h = F.dropout(F.elu(h), self.dropout, self.training)
        self.last_attention = att or None
        return F.normalize(x + self.gate * h, dim=-1)

    def concept_image_match(self, H, concepts: torch.Tensor, img_labels: torch.Tensor) -> torch.Tensor:
        """concepts [Uc] chỉ số đỉnh nhãn, img_labels [M, L] (−1 đệm) → m(c, i) [Uc, M]; ảnh không có nhãn → 0."""
        u = F.normalize(self.c_proj(H[concepts]), dim=-1)
        sim = u @ H.T                                              # [Uc, NL]
        valid = img_labels >= 0
        s = sim[:, img_labels.clamp(min=0)]                        # [Uc, M, L]
        s = s.masked_fill(~valid[None], -1e4)  # không dùng -inf: ảnh 0 nhãn ⇒ logsumexp(-inf) ⇒ gradient NaN qua where
        tau = self.log_tau.exp()
        m = tau * torch.logsumexp(s / tau, -1)
        return torch.where(valid.any(-1)[None], m, torch.zeros_like(m))

    def graph_matrix(self, H, cap_conc: torch.Tensor, img_labels: torch.Tensor) -> torch.Tensor:
        """cap_conc [B, C] (−1 đệm), img_labels [M, L] → s_G [B, M] (caption không có khái niệm → 0)."""
        uniq, inv = torch.unique(cap_conc.clamp(min=0), return_inverse=True)
        m = self.concept_image_match(H, uniq, img_labels)          # [Uc, M]
        per = m[inv]                                                # [B, C, M]
        valid = (cap_conc >= 0).float()[..., None]
        return (per * valid).sum(1) / valid.sum(1).clamp(min=1)

    # ---- adapter
    def adapter_cos(self, q, x_img):
        return F.normalize(self.q_proj(q), dim=-1) @ F.normalize(self.skip(x_img), dim=-1).T

    def total(self, adapter_s, graph_s):
        return adapter_s + self.w * graph_s


def info_nce_logits(logits, img_ids, tau):
    logits = logits / tau
    pos = (img_ids[:, None] == img_ids[None, :])
    l1 = -(torch.logsumexp(logits.masked_fill(~pos, -1e9), 1) - torch.logsumexp(logits, 1))
    l2 = -(torch.logsumexp(logits.T.masked_fill(~pos.T, -1e9), 1) - torch.logsumexp(logits.T, 1))
    return (l1.mean() + l2.mean()) / 2


@torch.no_grad()
def qa_scores(model: QAScorer, qd: QAData, x_img: torch.Tensor, q: np.ndarray, cap_ids: np.ndarray,
              cand_rows: np.ndarray, device, chunk_img: int = 512) -> np.ndarray:
    """Điểm (adapter + w·s_G) cho [Q, k] ứng viên. Tính sẵn m(c, i) cho mọi khái niệm × ảnh ứng viên (rẻ: ≤2000 × N)."""
    model.eval()
    H = model.label_h(qd.x_lab, qd.ei_lab)
    cc = qd.cap_concepts[torch.from_numpy(np.asarray(cap_ids)).long().to(device)]       # [Q, C]
    rows = torch.from_numpy(np.asarray(cand_rows)).long().to(device)                     # [Q, k]
    used_img = torch.unique(rows)
    pos_img = torch.full((x_img.shape[0],), -1, dtype=torch.long, device=device)
    pos_img[used_img] = torch.arange(len(used_img), device=device)
    uc = torch.unique(cc[cc >= 0]) if bool((cc >= 0).any()) else torch.zeros(1, dtype=torch.long, device=device)
    pos_c = torch.full((qd.x_lab.shape[0],), -1, dtype=torch.long, device=device)
    pos_c[uc] = torch.arange(len(uc), device=device)
    M = torch.cat([model.concept_image_match(H, uc, qd.img_labels[used_img[s:s + chunk_img]])
                   for s in range(0, len(used_img), chunk_img)], 1)                        # [Uc, U_img]
    pc = pos_c[cc.clamp(min=0)]                                                            # [Q, C]
    per = M[pc[:, :, None], pos_img[rows][:, None, :]]                                    # [Q, C, k]
    valid = (cc >= 0).float()[..., None]
    gs = (per * valid).sum(1) / valid.sum(1).clamp(min=1)
    qz = F.normalize(model.q_proj(torch.from_numpy(q).to(device)), dim=-1)
    iz = F.normalize(model.skip(x_img[rows]), dim=-1)                                     # [Q, k, D]
    ad = torch.einsum("qd,qkd->qk", qz, iz)
    return model.total(ad, gs).cpu().numpy()


def train_qa(model: QAScorer, qd: QAData, x_img: torch.Tensor, cap_emb: np.ndarray, cap_ids: np.ndarray,
             cap_img_rows: np.ndarray, val_fn, cfg: dict, device, log=print) -> dict:
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    Q = torch.from_numpy(cap_emb).to(device)
    CI = torch.from_numpy(np.asarray(cap_ids)).long().to(device)
    R = torch.from_numpy(cap_img_rows).long().to(device)
    bs, tau = int(cfg.get("batch_size", 256)), float(cfg.get("tau", 0.05))
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(int(cfg.get("epochs", 20))):
        model.train()
        perm = torch.randperm(len(Q), device=device)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            H = model.label_h(qd.x_lab, qd.ei_lab)
            imgs = R[b]
            gs = model.graph_matrix(H, qd.cap_concepts[CI[b]], qd.img_labels[imgs])
            logits = model.total(model.adapter_cos(Q[b], x_img[imgs]), gs)
            loss = info_nce_logits(logits, imgs, tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        v = val_fn(model)
        w, t, gt = model.w.item(), model.log_tau.exp().item(), model.gate.item()
        hist.append({"epoch": ep, "loss": tot / len(perm), "val_MRR": v, "w": w, "tau_match": t, "gate": gt})
        log(f"  epoch {ep:3d} loss {tot / len(perm):.4f} val_MRR {v:.4f} w {w:.3f} tau {t:.3f} gate {gt:.3f}")
        if v > best + 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(cfg.get("patience", 5)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_MRR": best, "epochs_run": len(hist), "history": hist}
