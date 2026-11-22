"""Kiểm chứng ví dụ Hình 2: woman khớp person là nhờ cạnh IsA / nhờ GNN, hay đã có sẵn trong vectơ CLIP?

    cd report && ../.venv/bin/python measure_isa.py      (cần data/psg_bundle, models/psg_qa_cn_seed0)

So ba trường hợp trên cùng mô hình psg_qa_cn seed 0 đã huấn luyện, cùng truy vấn và 100 ảnh ứng viên của Hình 2:
  đầy đủ · bỏ riêng cạnh woman↔person · tắt GNN (γ = 0, H = norm(X)).
Đồng thời đo mức GNN làm đổi vectơ nhãn trên toàn đồ thị: cos(H_l, norm X_l) và ‖γ·GNN(X)‖ / ‖X‖.
Cuối cùng đo cùng các đại lượng đó trên cả bốn checkpoint GAT seed 0 (VG/PSG, có/không ConceptNet).
Ghi ra figs/isa_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmxai.graph.qa import qa_scores  # noqa: E402
from mmxai.xai_bridge import QAGraphBridge  # noqa: E402

CAPTION = "a woman throwing a frisbee in a green field."
torch.set_num_threads(2)
br = QAGraphBridge(ROOT / "models" / "psg_qa_cn_seed0", ROOT / "data" / "psg_bundle")
P = br.ev.Protocol.load(ROOT / "data" / "psg_bundle")
c = P.candidates("test", None)
caps = [json.loads(l) for l in open(ROOT / "data" / "psg_bundle" / "data" / "captions.jsonl", encoding="utf-8")]
k = next(k for k, qi in enumerate(c["q_cap"]) if caps[int(qi)]["text"].strip().lower() == CAPTION)
qi = int(c["q_cap"][k]); gold = int(caps[qi]["img"])
cand, sim = np.asarray(c["cand"][k]), np.asarray(c["sim"][k])
rows = np.array([[br.row_of[int(i)] for i in cand]])
W, PER = br.lab_idx["woman"], br.lab_idx["person"]
ei_full, gate0 = br.qd.ei_lab.clone(), br.model.gate.data.clone()
m = br.model


def measure(name):
    with torch.no_grad():
        H = m.label_h(br.qd.x_lab, br.qd.ei_lab)
        u = F.normalize(m.c_proj(H[W:W + 1]), dim=-1)[0]
        mm = float(m.concept_image_match(H, torch.tensor([W]), br.qd.img_labels[[br.row_of[gold]]])[0, 0])
        g = qa_scores(m, br.qd, br.x_img, P.cap[[qi]], np.array([qi]), rows, "cpu")[0]
    fused = br.ev.fuse(sim[None], g[None], br.alpha)[0]
    rank = int(np.where(cand[np.argsort(-fused, kind="stable")] == gold)[0][0]) + 1
    return {"case": name, "cos_woman_person": float(u @ H[PER]), "m_woman_gold": mm, "rank_gold": rank}


res = [measure("đầy đủ")]
s, d = br.qd.ei_lab
keep = ~(((s == W) & (d == PER)) | ((s == PER) & (d == W)))
n_removed = int((~keep).sum())
br.qd.ei_lab = ei_full[:, keep]
res.append(measure(f"bỏ cạnh woman↔person ({n_removed} cạnh)"))
br.qd.ei_lab = ei_full
m.gate.data.zero_()
res.append(measure("tắt GNN (γ = 0)"))
m.gate.data.copy_(gate0)


def fused_gold():
    with torch.no_grad():
        g = qa_scores(m, br.qd, br.x_img, P.cap[[qi]], np.array([qi]), rows, "cpu")[0]
    f = br.ev.fuse(sim[None], g[None], br.alpha)[0]
    return float(f[np.where(cand == gold)[0][0]]), int(np.where(cand[np.argsort(-f, kind="stable")] == gold)[0][0]) + 1


# Xóa từng cạnh (như phép đo fidelity): cạnh contains của ảnh đúng, và cạnh IsA woman↔person
s0, _ = fused_gold()
loo, grow = [], br.row_of[gold]
row = br.qd.img_labels[grow].clone()
for j, l in enumerate(row.tolist()):
    if l < 0:
        continue
    br.qd.img_labels[grow, j] = -1
    s, r = fused_gold()
    br.qd.img_labels[grow] = row.clone()
    loo.append({"edge": f"{br.g.labels[l]} -contains-> ảnh đúng", "score_drop": s0 - s, "rank_gold_after": r})
br.qd.ei_lab = ei_full[:, keep]
s, r = fused_gold()
br.qd.ei_lab = ei_full
loo.append({"edge": "woman <-IsA-> person", "score_drop": s0 - s, "rank_gold_after": r})

with torch.no_grad():
    X = br.qd.x_lab
    h = X
    for i, conv in enumerate(m.convs):
        h = conv(h, br.qd.ei_lab)
        if i < len(m.convs) - 1:
            h = F.elu(h)
    H = F.normalize(X + m.gate * h, dim=-1)
    shift = {"gamma": float(m.gate), "cos_H_normX_mean": float((H * F.normalize(X, dim=-1)).sum(-1).mean()),
             "cos_H_normX_min": float((H * F.normalize(X, dim=-1)).sum(-1).min()),
             "ratio_gammaGNN_over_X_mean": float(((m.gate * h).norm(dim=-1) / X.norm(dim=-1)).mean())}

out = {"caption": CAPTION, "cases": res, "leave_one_edge_out": loo, "label_shift": shift}
(Path(__file__).resolve().parent / "figs" / "isa_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))

# Mức GNN làm đổi vectơ nhãn trên cả bốn checkpoint GAT seed 0 (số liệu của mục VI-B, quan sát 4)
for mdl, ds in [("psg_qa_seed0", "psg"), ("psg_qa_cn_seed0", "psg"), ("vg_qa_seed0", "vg"), ("vg_qa_cn_seed0", "vg")]:
    b = QAGraphBridge(ROOT / "models" / mdl, ROOT / "data" / f"{ds}_bundle")
    X, h = b.qd.x_lab, b.qd.x_lab
    with torch.no_grad():
        for i, conv in enumerate(b.model.convs):
            h = conv(h, b.qd.ei_lab)
            if i < len(b.model.convs) - 1:
                h = F.elu(h)
        cos = (F.normalize(X + b.model.gate * h, dim=-1) * F.normalize(X, dim=-1)).sum(-1)
        out.setdefault("label_shift_all", {})[mdl] = {
            "gamma": float(b.model.gate), "cos_H_normX_mean": float(cos.mean()),
            "ratio_gammaGNN_over_X_mean": float(((b.model.gate * h).norm(dim=-1) / X.norm(dim=-1)).mean())}
(Path(__file__).resolve().parent / "figs" / "isa_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out["label_shift_all"], indent=1))
