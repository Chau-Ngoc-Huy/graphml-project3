"""Adapter nhẹ trên đặc trưng CLIP đóng băng — phần MỞ RỘNG của Khánh, không thuộc AdaMF-MAT.

Cùng ý tưởng adapter của Huy (`huy/src/mmxai/graph/qa.py`: q_proj / skip khởi tạo = ma trận đơn vị, InfoNCE, dừng sớm theo MRR val):
hai phép tuyến tính D×D (caption, ảnh) khởi tạo = đơn vị, huấn luyện InfoNCE (tau 0.05) trên caption + ảnh của split TRAIN, chọn epoch theo MRR trên VAL.
Test không được dùng. Điểm adapter thay cosine CLIP trong cùng top-K ứng viên của giao thức chung.

Vì sao có file này: adapter tự nó đã tăng MRR (Huy: 60.40 → 63.14 toàn pool). Để biết AdaMF-MAT đóng góp bao nhiêu thì phải so
"CLIP+Adapter" với "CLIP+Adapter+AdaMF-MAT", không so với CLIP thô.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.i = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.q.weight)
        nn.init.eye_(self.i.weight)

    def text(self, x):
        return F.normalize(self.q(x), dim=-1)

    def image(self, x):
        return F.normalize(self.i(x), dim=-1)


@torch.no_grad()
def adapted_sim(model, P, c, device="cpu", chunk=500):
    """cos(adapter(caption), adapter(ảnh ứng viên)) → [Nq, K], cùng hình dạng với c["sim"]."""
    model.eval().to(device)
    out = np.zeros(c["sim"].shape, np.float32)
    for s in range(0, len(out), chunk):
        q = torch.from_numpy(P.cap[c["q_cap"][s:s + chunk]]).to(device)
        x = torch.from_numpy(P.img[c["cand"][s:s + chunk]]).to(device)  # [b, K, D]
        out[s:s + chunk] = torch.einsum("bd,bkd->bk", model.text(q), model.image(x)).cpu().numpy()
    return out


def _val_mrr(model, P, device):
    from mmxai_eval import rank_metrics, rerank_ranks
    c = P.candidates("val", None)
    return rank_metrics(rerank_ranks(c, adapted_sim(model, P, c, device)))["MRR"]


def train_adapter(P, seed=0, epochs=20, patience=4, batch_size=512, tau=0.05, lr=1e-3, weight_decay=1e-4, device=None, log=print):
    """Huấn luyện trên caption/ảnh TRAIN; trả về mô hình ở epoch có MRR val tốt nhất."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = Adapter(P.img.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_imgs = P.pool("train")
    cap_idx = np.array([ci for ii in train_imgs for ci in P.caps_of.get(ii, [])])
    cap_img = P.cap_img[cap_idx]
    Q = torch.from_numpy(P.cap[cap_idx]).to(device)
    X = torch.from_numpy(P.img).to(device)
    R = torch.from_numpy(cap_img.astype(np.int64)).to(device)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Q), device=device)
        tot = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            imgs = R[b]
            logits = model.text(Q[b]) @ model.image(X[imgs]).T / tau
            same = (imgs[:, None] == imgs[None, :]) & ~torch.eye(len(b), dtype=torch.bool, device=device)  # cùng ảnh ≠ âm
            loss = F.cross_entropy(logits.masked_fill(same, -1e9), torch.arange(len(b), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        v = _val_mrr(model, P, device)
        hist.append({"epoch": ep, "loss": tot / len(perm), "val_MRR": v})
        log(f"  adapter epoch {ep:2d} loss {tot / len(perm):.4f} val_MRR {v:.2f}")
        if v > best + 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.history, model.best_val_mrr = hist, best
    return model.eval()
