"""Chạy khung xAI của Dương (duong/xai, KHÔNG sửa) trên mô hình QA-Graph thật qua mmxai.xai_bridge.

  python tools/xai_bridge_demo.py --model models/psg_qa_cn_seed0 --bundle data/psg_bundle --n 10 --k 20
  python tools/xai_bridge_demo.py --model models/vg_t1_qa_cn_seed0 --bundle data/task1_bundle --n 10

Chọn n caption test (nửa direct, nửa indirect, cố định theo seed), ứng viên = top-k CLIP (+ ảnh đúng), export đồ thị con theo
định dạng duong/xai/io.load_graph, rồi cho mỗi truy vấn: EdgeContributionExplainer (bỏ từng cạnh, mô hình chấm lại) và
XAIEvaluator (fidelity so với xóa ngẫu nhiên, sparsity, validity theo duong/configs/graph_schema.json).
Kiểm tra đầu tiên: final_score của cầu nối khớp điểm giao thức (qa_scores + fuse) trên đồ thị đầy đủ.
Ra: <out>/graph.json, cases.json, results.jsonl. CPU ~1–2 phút / truy vấn / giá trị k.

Cần (không có trong git):
  checkpoint  Kaggle riêng tư huychau/mmxai-huy-models  → models/{vg_t1_qa,vg_t1_qa_cn,psg_qa,psg_qa_cn}_seed0/
  bundle      Kaggle riêng tư huychau/mmxai-task1-bundle (VG) → data/task1_bundle/,
              huychau/mmxai-psg-bundle (PSG) → data/psg_bundle/
  vd: kaggle datasets download huychau/mmxai-huy-models -p models --unzip
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "duong"))
from mmxai.graph.qa import qa_scores  # noqa: E402
from mmxai.xai_bridge import QAGraphBridge  # noqa: E402
from xai.evaluate_xai import XAIEvaluator  # noqa: E402
from xai.explain import EdgeContributionExplainer  # noqa: E402
from xai.graph import Edge, KnowledgeGraph, Node  # noqa: E402
from xai.io import load_schema  # noqa: E402
from xai.scorer_adapter import CallableScorerAdapter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="thư mục có model.pt, config.json, train_info.json (vd models/psg_qa_cn_seed0)")
ap.add_argument("--bundle", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--k", type=int, default=20, help="số ứng viên CLIP mỗi truy vấn")
ap.add_argument("--sizes", type=int, nargs="+", default=[1, 3, 5])
ap.add_argument("--repeats", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(2)
out = Path(a.out or ROOT / "results" / "xai_bridge" / Path(a.model).name)
out.mkdir(parents=True, exist_ok=True)

br = QAGraphBridge(a.model, a.bundle)
test = set(br.b.splits["test"])
rng = np.random.default_rng(a.seed)
pick = []
for grp in ("direct", "indirect"):
    pool = [c["caption_id"] for c in br.b.captions if c["image_id"] in test and c.get("group") == grp]
    pick += rng.choice(pool, size=min(len(pool), a.n - a.n // 2 if grp == "direct" else a.n // 2), replace=False).tolist()
cases = br.cases(pick, k=a.k)
G = br.export_graph(cases)
(out / "graph.json").write_text(json.dumps(G, ensure_ascii=False))
(out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
kg = KnowledgeGraph(
    nodes={n["id"]: Node(n["id"], n["type"], n.get("label", ""), n.get("external_id")) for n in G["nodes"]},
    edges={e["id"]: Edge(e["id"], e["source"], e["target"], e["relation"], provenance=e.get("provenance"),
                         support_status=e["support_status"], metadata=e.get("metadata", {})) for e in G["edges"]})
print(f"{len(cases)} truy vấn · đồ thị con {len(kg.nodes)} nút, {len(kg.edges)} cạnh", flush=True)

c0 = cases[0]
q, _ = br.query(c0["query_text"])
rows = np.array([[br.row_of[int(i)] for i in c0["candidate_image_ids"]]])
ref = qa_scores(br.model, br.qd, br.x_img, q[None], np.array([int(c0["query_id"][3:])]), rows, "cpu")[0]
fz = br.ev.fuse((br.b.image_emb[rows[0]] @ q)[None], ref[None], br.alpha)[0]
s = br.score_fn(c0["query_text"], c0["candidate_image_ids"], kg)
err = max(abs(s[i]["final_score"] - fz[r]) for r, i in enumerate(c0["candidate_image_ids"]))
print(f"kiểm tra: |final cầu nối − giao thức| = {err:.2e}", flush=True)
assert err < 1e-4

scorer = CallableScorerAdapter(br.score_fn, br.local_edges_fn)
schema = load_schema(ROOT.parent / "duong" / "configs" / "graph_schema.json")
with open(out / "results.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        for k in a.sizes:
            t = time.time()
            ex = EdgeContributionExplainer(top_k=k).explain(c["query_text"], c["image_id"], c["candidate_image_ids"], kg, scorer)
            m = XAIEvaluator(schema, random_repeats=a.repeats, seed=a.seed).evaluate(
                c["query_text"], c["image_id"], c["candidate_image_ids"], kg, scorer, ex)
            row = {"query_id": c["query_id"], "group": c["group"], "k": k,
                   "explanation": [[kg.edges[e].source, kg.edges[e].relation, kg.edges[e].target] for e in ex.edge_ids],
                   **{kk: v for kk, v in m.to_dict().items() if not isinstance(v, (list, dict, tuple))}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{c['query_id']} {c['group']:<8} k={k} rank {row['rank_before']}→{row['rank_after']} "
                  f"Δfinal {row['final_score_drop']:+.3f} (ngẫu nhiên {row['random_final_score_drop_mean']:+.3f}) "
                  f"{time.time() - t:.0f}s | {row['explanation']}", flush=True)
print(f"→ {out}")
