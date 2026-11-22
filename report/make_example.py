"""Số liệu + ảnh cho ví dụ minh họa ở Hình 1 và mục IV-C (một truy vấn thật trong tập test PSG).

    ../.venv/bin/python make_example.py ["<caption>"]      (chạy trong report/; cần data/psg_bundle, data/psg_raw, models/psg_qa_cn_seed0)

Ghi ra figs/example.json (khái niệm, nhãn khớp nhất, m(c,i), cạnh đồ thị, hạng CLIP → QA-Graph, ảnh top theo CLIP)
và figs/ex_*.jpg (ảnh PSG trích từ parquet, thu nhỏ). Mọi số lấy từ mô hình psg_qa_cn seed 0 đã huấn luyện.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmxai.graph.qa import qa_scores  # noqa: E402
from mmxai.xai_bridge import QAGraphBridge  # noqa: E402

CAPTION = sys.argv[1] if len(sys.argv) > 1 else "a woman throwing a frisbee in a green field."
BUNDLE, MODEL, RAW = ROOT / "data" / "psg_bundle", ROOT / "models" / "psg_qa_cn_seed0", ROOT / "data" / "psg_raw" / "data"
OUT = Path(__file__).resolve().parent / "figs"
torch.set_num_threads(2)

br = QAGraphBridge(MODEL, BUNDLE)
P = br.ev.Protocol.load(BUNDLE)
c = P.candidates("test", None)
caps = [json.loads(l) for l in open(BUNDLE / "data" / "captions.jsonl", encoding="utf-8")]
imgs = [json.loads(l) for l in open(BUNDLE / "data" / "images.jsonl", encoding="utf-8")]
k = next(k for k, qi in enumerate(c["q_cap"]) if caps[int(qi)]["text"].strip().lower() == CAPTION.strip().lower())
qi = int(c["q_cap"][k]); cap = caps[qi]; gold = int(cap["img"])
cand = np.asarray(c["cand"][k]); sim = np.asarray(c["sim"][k])
rows = np.array([[br.row_of[int(i)] for i in cand]])
g = qa_scores(br.model, br.qd, br.x_img, P.cap[[qi]], np.array([qi]), rows, "cpu")[0]
fused = br.ev.fuse(sim[None], g[None], br.alpha)[0]
order_clip, order_qa = np.argsort(-sim, kind="stable"), np.argsort(-fused, kind="stable")
rank_clip = int(np.where(cand[order_clip] == gold)[0][0]) + 1
rank_qa = int(np.where(cand[order_qa] == gold)[0][0]) + 1

with torch.no_grad():
    H = br.model.label_h(br.qd.x_lab, br.qd.ei_lab)
    lab = br.g.labels
    def match(concept, img):
        cid = br.lab_idx[concept]
        u = F.normalize(br.model.c_proj(H[cid:cid + 1]), dim=-1)[0]
        ls = [l for l in br.qd.img_labels[br.row_of[img]].tolist() if l >= 0]
        cos = {lab[l]: float(u @ H[l]) for l in ls}
        m = float(br.model.concept_image_match(H, torch.tensor([cid]), br.qd.img_labels[[br.row_of[img]]])[0, 0])
        best = max(cos, key=cos.get)
        return {"concept": concept, "in_image": concept in cos, "best_label": best, "cos_best": cos[best], "m": m,
                "cos_all": dict(sorted(cos.items(), key=lambda x: -x[1]))}

def edges_between(a, b):
    """Cạnh nhãn–nhãn giữa a và b (1 bước) hoặc qua một nhãn trung gian (2 bước), kèm tên quan hệ."""
    n, A, B = br.g.n_img, br.lab_idx[a], br.lab_idx[b]
    src, dst = br.lab_src, br.lab_dst
    out = {(s, d): t for s, d, t in zip(src, dst, br.lab_type)}
    name = lambda s, d: br.g.rel_name.get((s + n, d + n), "")
    if (A, B) in out:
        return [[a, name(A, B), b]]
    for (s, d), t in out.items():
        if s == A and (d, B) in out:
            return [[a, name(A, d), lab[d]], [lab[d], name(d, B), b]]
    return []

concepts = [x for x in cap["concepts"] if x in br.lab_idx]
detail = [match(x, gold) for x in concepts]
for d in detail:
    d["path"] = [] if d["in_image"] else edges_between(d["concept"], d["best_label"])
top_clip = [int(cand[i]) for i in order_clip[:3]]

def save(img_idx, name):
    pid = imgs[img_idx]["psg_id"]
    for f in sorted(RAW.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            ids = pf.read_row_group(rg, columns=["image_id"]).column(0).to_pylist()
            if pid in ids:
                b = pf.read_row_group(rg, columns=["image"]).column(0).to_pylist()[ids.index(pid)]["bytes"]
                im = Image.open(io.BytesIO(b)).convert("RGB")
                im.thumbnail((480, 480))
                im.save(OUT / name, quality=88)
                return
    raise SystemExit(f"không thấy ảnh psg_id={pid}")

save(gold, "ex_gold.jpg")
for r, i in enumerate(top_clip[:2]):
    if i != gold:
        save(i, f"ex_clip{r + 1}.jpg")
res = {"caption": cap["text"].strip(), "group": cap["group"], "concepts": concepts, "gold_psg_id": imgs[gold]["psg_id"],
       "gold_labels": imgs[gold]["objects"], "gold_relations": imgs[gold]["relations"],
       "rank_clip": rank_clip, "rank_qa": rank_qa, "alpha": br.alpha, "detail": detail,
       "graph_score": float(np.mean([d["m"] for d in detail])),
       "clip_top": [{"rank": r + 1, "psg_id": imgs[i]["psg_id"], "labels": imgs[i]["objects"], "is_gold": i == gold,
                     "rank_after": int(np.where(cand[order_qa] == i)[0][0]) + 1} for r, i in enumerate(top_clip)]}
(OUT / "example.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
