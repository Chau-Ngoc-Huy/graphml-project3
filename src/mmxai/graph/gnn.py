"""GNN (GCN / GAT / GraphSAGE) tạo nhúng ảnh có thông tin đồ thị + huấn luyện tương phản với caption TRAIN.

graph_score(q, ảnh i) = cos(P·q, z_i), z_i = normalize(S·x_i + GNN(x)_i). P, S khởi tạo bằng ma trận đơn vị nên
trước khi học, graph_score ≈ cosine CLIP; GNN học thêm phần tri thức từ láng giềng (nhãn, khái niệm).

Chống rò rỉ: chỉ caption của ảnh train làm truy vấn huấn luyện; early stopping và α chọn trên val; test chỉ dùng lúc đánh giá.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    def __init__(self, dim: int, kind: str = "gat", hidden: int = 256, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv

        self.kind = kind
        dims = [dim] + [hidden] * (layers - 1) + [dim]
        convs = []
        for a, b in zip(dims[:-1], dims[1:]):
            if kind == "gcn":
                convs.append(GCNConv(a, b))
            elif kind == "sage":
                convs.append(SAGEConv(a, b))
            elif kind == "gat":
                convs.append(GATConv(a, b, heads=heads, concat=False, dropout=dropout))
            else:
                raise ValueError(f"gnn.kind phải là gcn|gat|sage, nhận {kind!r}")
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout
        self.skip = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.skip.weight)
        nn.init.eye_(self.q_proj.weight)
        self.gate = nn.Parameter(torch.tensor(0.1))  # GNN bắt đầu đóng góp nhỏ → ổn định
        self.last_attention = None

    def forward(self, x, edge_index, keep_attention: bool = False):
        h = x
        att = []
        for i, conv in enumerate(self.convs):
            if self.kind == "gat" and keep_attention:
                h, (ei, a) = conv(h, edge_index, return_attention_weights=True)
                att.append((ei, a.mean(-1) if a.dim() > 1 else a))
            else:
                h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.dropout(F.elu(h), self.dropout, self.training)
        self.last_attention = att or None
        return F.normalize(self.skip(x) + self.gate * h, dim=-1)

    def query(self, q):
        return F.normalize(self.q_proj(q), dim=-1)


def info_nce(qz, iz, img_ids, tau):
    """Contrastive nhiều dương: caption cùng ảnh trong batch không bị tính là âm."""
    logits = qz @ iz.T / tau
    pos = (img_ids[:, None] == img_ids[None, :]).float()
    l1 = -(torch.logsumexp(logits.masked_fill(pos == 0, -1e9), 1) - torch.logsumexp(logits, 1))
    l2 = -(torch.logsumexp(logits.T.masked_fill(pos.T == 0, -1e9), 1) - torch.logsumexp(logits.T, 1))
    return (l1.mean() + l2.mean()) / 2


@torch.no_grad()
def graph_scores(model, data, q: np.ndarray, cand_rows: np.ndarray, device) -> np.ndarray:
    """Điểm đồ thị cho từng (truy vấn, ứng viên): q [Q,D], cand_rows [Q,k] chỉ số đỉnh ảnh → [Q,k]."""
    model.eval()
    z = model(data.x.to(device), data.edge_index.to(device))
    qz = model.query(torch.from_numpy(q).to(device))
    zc = z[torch.from_numpy(cand_rows).to(device)]           # [Q,k,D]
    return torch.einsum("qd,qkd->qk", qz, zc).cpu().numpy()


class BatchSubgraph:
    """Ảnh chỉ NHẬN tin (cạnh nhãn→ảnh), không gửi → nhúng của ảnh trong batch chỉ phụ thuộc đỉnh nhãn và chính nó.
    Đồ thị con {mọi nhãn} ∪ {ảnh trong batch} cho kết quả CHÍNH XÁC như chạy cả đồ thị nhưng nhanh hơn N_img/batch lần."""

    def __init__(self, data, n_img: int, device):
        ei = data.edge_index
        self.x = data.x.to(device)
        self.n_img, self.n = n_img, data.num_nodes
        to_label = ei[1] >= n_img
        assert bool((ei[0][~to_label] >= n_img).all()), "ảnh không được gửi tin (mọi cạnh vào ảnh phải xuất phát từ nhãn)"
        self.ll = ei[:, to_label].to(device)                       # cạnh vào đỉnh nhãn (giữ nguyên mọi bước)
        c = ei[:, ~to_label]                                      # cạnh vào đỉnh ảnh, sắp theo ảnh đích
        order = torch.argsort(c[1])
        self.c_src, self.c_dst = c[0][order].to(device), c[1][order].to(device)
        self.ptr = torch.searchsorted(self.c_dst, torch.arange(n_img + 1, device=device))

    def __call__(self, rows: torch.Tensor):
        """rows: đỉnh ảnh (có thể lặp) → (x_sub, ei_sub, vị trí của rows trong đồ thị con)."""
        uniq, inv = torch.unique(rows, return_inverse=True)
        n_lab = self.n - self.n_img
        # đánh số lại: nhãn → [0, n_lab), ảnh batch → [n_lab, n_lab + |uniq|)
        remap = torch.full((self.n,), -1, dtype=torch.long, device=rows.device)
        remap[self.n_img:] = torch.arange(n_lab, device=rows.device)
        remap[uniq] = n_lab + torch.arange(len(uniq), device=rows.device)
        starts, cnt = self.ptr[uniq], self.ptr[uniq + 1] - self.ptr[uniq]
        offs = torch.arange(int(cnt.sum()), device=rows.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        idx = torch.repeat_interleave(starts, cnt) + offs
        src = torch.cat([self.ll[0], self.c_src[idx]])
        dst = torch.cat([self.ll[1], self.c_dst[idx]])
        x_sub = torch.cat([self.x[self.n_img:], self.x[uniq]])
        return x_sub, torch.stack([remap[src], remap[dst]]), n_lab + inv


def train(model, data, n_img: int, cap_emb: np.ndarray, cap_img_rows: np.ndarray, val_fn, cfg: dict, device,
          log=print) -> dict:
    """cap_emb/cap_img_rows: caption TRAIN và đỉnh ảnh tương ứng. val_fn(model) -> MRR trên val (early stopping)."""
    model.to(device)
    sub = BatchSubgraph(data, n_img, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    Q = torch.from_numpy(cap_emb).to(device)
    R = torch.from_numpy(cap_img_rows).long().to(device)
    bs, tau = int(cfg.get("batch_size", 256)), float(cfg.get("tau", 0.05))
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(int(cfg.get("epochs", 20))):
        model.train()
        perm = torch.randperm(len(Q), device=device)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            xs, es, pos = sub(R[b])
            z = model(xs, es)[pos]
            loss = info_nce(model.query(Q[b]), z, R[b], tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        v = val_fn(model)
        hist.append({"epoch": ep, "loss": tot / len(perm), "val_MRR": v})
        log(f"  epoch {ep:3d} loss {tot / len(perm):.4f} val_MRR(graph) {v:.4f}")
        if v > best + 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(cfg.get("patience", 5)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_MRR_graph": best, "epochs_run": len(hist), "history": hist}
