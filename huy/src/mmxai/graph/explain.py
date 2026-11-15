"""Giải thích mức cơ bản (cấp 1) + kiểm tra fidelity, làm nền cho Nhiệm vụ 3 (Dương).

Giải thích lấy từ CHÍNH cơ chế tính graph_score: z_ảnh do GNN 2 tầng tổng hợp từ láng giềng ≤ 2 bước, nên mọi đường đi
nhãn → ảnh và khái niệm → nhãn → ảnh trong đồ thị con 2 bước của ảnh là thứ thực sự đi vào điểm. Đóng góp của một đường đi
= cos(truy vấn, đặc trưng đỉnh đầu) × trọng số cạnh dọc đường (attention GAT, hoặc 1/bậc vào với GCN/SAGE).

Fidelity: xóa các cạnh của đường đi giải thích rồi tính lại graph_score trên đồ thị con (chính xác vì GNN chỉ nhìn 2 bước);
so với xóa cùng số cạnh ngẫu nhiên. Giải thích trung thực ⇒ điểm giảm nhiều hơn hẳn so với xóa ngẫu nhiên.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .build import TYPE_NAMES


def _edge_weights(model, x, ei, n):
    """Trọng số cạnh (src,dst) mỗi tầng. GAT: attention thật; GCN/SAGE: 1/bậc vào (cách gộp trung bình)."""
    model(x, ei, keep_attention=True)
    if model.last_attention:
        return [{(int(s), int(d)): float(a) for s, d, a in zip(e[0], e[1], att)} for e, att in model.last_attention]
    deg = torch.bincount(ei[1], minlength=n).clamp(min=1).float()
    w = {(s, d): float(1 / deg[d]) for s, d in ei.T.tolist()}
    return [w] * len(model.convs)


@torch.no_grad()
def explain(model, g, q: np.ndarray, img_row: int, top_m: int = 3, n_random: int = 5, seed: int = 0, device="cpu") -> dict:
    from torch_geometric.utils import k_hop_subgraph

    model.eval()
    L = len(model.convs)
    subset, sub_ei, center, emask = k_hop_subgraph(img_row, L, g.data.edge_index, relabel_nodes=True,
                                                   num_nodes=g.data.num_nodes, flow="source_to_target")
    c = int(center[0])
    x = g.data.x[subset].to(device)
    sub_ei = sub_ei.to(device)
    etype = g.data.edge_type[emask]
    qz = model.query(torch.from_numpy(q[None]).to(device))[0]

    def score(ei):
        return float(qz @ model(x, ei)[c])

    base = score(sub_ei)
    W = _edge_weights(model, x, sub_ei, len(subset))
    sim = (F.normalize(x, dim=-1) @ qz).cpu().numpy()
    eidx = {(int(s), int(d)): k for k, (s, d) in enumerate(sub_ei.T.tolist())}
    into_c = [(s, d) for (s, d) in eidx if d == c and s != c]
    paths = []
    for (l, _) in into_c:
        w1 = W[-1].get((l, c), 0.0)
        paths.append({"nodes": [l, c], "edges": [eidx[(l, c)]], "contrib": float(sim[l] * w1)})
        if L >= 2:
            for (m, d) in eidx:
                if d == l and m not in (l, c):
                    w0 = W[-2].get((m, l), 0.0)
                    paths.append({"nodes": [m, l, c], "edges": [eidx[(m, l)], eidx[(l, c)]],
                                  "contrib": float(sim[m] * w0 * w1)})
    paths.sort(key=lambda p: -p["contrib"])
    # top_m đường đi cho MỖI độ dài: attention chia cho nhiều cạnh vào nhãn nên đường 2 bước luôn nhỏ hơn đường 1 bước,
    # nhưng chính đường 2 bước (khái niệm → nhãn → ảnh) mới giải thích truy vấn gián tiếp.
    top = [p for n in (2, 3) for p in [q for q in paths if len(q["nodes"]) == n][:top_m]]

    def describe(p):
        glob = [int(subset[i]) for i in p["nodes"]]
        parts = [g.node_name(glob[0])]
        for k in range(len(glob) - 1):
            e = p["edges"][k]
            rel = g.rel_name.get((glob[k], glob[k + 1]), TYPE_NAMES[int(etype[e])])
            parts += [f"-[{rel}]->", g.node_name(glob[k + 1])]
        return " ".join(parts)

    drop = sorted({e for p in top for e in p["edges"]})
    keep = torch.ones(sub_ei.shape[1], dtype=torch.bool)
    keep[drop] = False
    after = score(sub_ei[:, keep.to(device)]) if drop else base
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random if drop else 0):
        k2 = torch.ones(sub_ei.shape[1], dtype=torch.bool)
        k2[rng.choice(sub_ei.shape[1], size=len(drop), replace=False)] = False
        rand.append(base - score(sub_ei[:, k2.to(device)]))
    return {
        "graph_score": base,
        "paths": [{"path": describe(p), "hops": len(p["nodes"]) - 1, "contrib": p["contrib"]} for p in top],
        "fidelity": {"delta_explanation": base - after, "delta_random_mean": float(np.mean(rand)) if rand else 0.0,
                     "n_removed": len(drop), "n_edges_subgraph": int(sub_ei.shape[1]),
                     "sparsity": len(drop) / max(1, int(sub_ei.shape[1]))},
    }
