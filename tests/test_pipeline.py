"""Kiểm tra pipeline graph-rerank trên dữ liệu giả: metric, chống rò rỉ, đồ thị con theo batch, chạy end-to-end."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mmxai.core.runner import main
from mmxai.data.contract import load
from mmxai.data.fake import generate
from mmxai.graph.build import CONTAINS, SG_REL, VARIANTS, build_graph
from mmxai.graph.gnn import BatchSubgraph, GraphEncoder
from mmxai.retrieval.core import metrics_from_ranks, ranks_of_gold

SMALL = {"n_images": 200, "seed": 1}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return load(generate(tmp_path_factory.mktemp("fake"), **SMALL))


def test_ranks_and_metrics():
    scores = np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.3], [0.1, 0.2, 0.3]])
    cand = np.array([[7, 3, 1], [7, 3, 1], [7, 3, 1]])
    r = ranks_of_gold(scores, cand, np.array([7, 7, 9]))  # hạng 1, hạng 3, không có trong ứng viên
    assert r.tolist() == [1, 3, 0]
    m = metrics_from_ranks(r)
    assert m["R@1"] == pytest.approx(1 / 3) and m["R@5"] == pytest.approx(2 / 3)
    assert m["MRR"] == pytest.approx((1 + 1 / 3) / 3)


def test_splits_disjoint(bundle):
    s = {k: set(v) for k, v in bundle.splits.items()}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])


def test_sg_edges_only_from_train(bundle):
    """Cạnh quan hệ scene graph là thống kê → chỉ từ ảnh train (đề mục 6)."""
    test_img = bundle.splits["test"][0]
    bundle.image_rels[test_img] = bundle.image_rels[test_img] + [("concept0", "leak", "concept1")]
    g = build_graph(bundle, "full")
    assert "leak" not in g.rel_name.values()
    lab = {l: g.n_img + i for i, l in enumerate(bundle.labels)}
    ei, et = g.data.edge_index, g.data.edge_type
    assert not (((ei[0] == lab["concept0"]) & (ei[1] == lab["concept1"])) & (et == SG_REL)).any()


def test_images_never_send(bundle):
    for v in VARIANTS:
        g = build_graph(bundle, v)
        assert bool((g.data.edge_index[0] >= g.n_img).all()), v
        assert int((g.data.edge_type == CONTAINS).sum()) == (0 if v == "no_edge" else
                                                              int((build_graph(bundle).data.edge_type == CONTAINS).sum()))


@pytest.mark.parametrize("kind", ["gcn", "gat", "sage"])
def test_batch_subgraph_exact(bundle, kind):
    g = build_graph(bundle, "full")
    torch.manual_seed(0)
    m = GraphEncoder(bundle.image_emb.shape[1], kind).eval()
    rows = torch.tensor([3, 1, 3, 50])
    with torch.no_grad():
        full = m(g.data.x, g.data.edge_index)[rows]
        xs, es, pos = BatchSubgraph(g.data, g.n_img, "cpu")(rows)
        part = m(xs, es)[pos]
    assert torch.allclose(full, part, atol=1e-5)


def test_clip_graph_end_to_end(tmp_path):
    run_dir = main(["--exp", "clip_graph", "--limit", "5", "--runs-dir", str(tmp_path), "--no-publish",
                    "--set", f"fake={json.dumps(SMALL)}", "--set", "n_explain=2", "--set", "gnn.kind=sage"])
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["run"]["error"] is None
    for key in ("clip", "clip_graph", "graph_only"):
        assert 0 <= m["metrics"][key]["all"]["MRR"] <= 1
    ex = json.loads((run_dir / "explanations.json").read_text())
    assert len(ex) == 2 and ex[0]["paths"]
