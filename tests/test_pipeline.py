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


@pytest.mark.parametrize("arch", ["legacy", "split"])
@pytest.mark.parametrize("kind", ["gcn", "gat", "sage"])
def test_batch_subgraph_exact(bundle, kind, arch):
    g = build_graph(bundle, "full")
    torch.manual_seed(0)
    m = GraphEncoder(bundle.image_emb.shape[1], kind, arch=arch).eval()
    rows = torch.tensor([3, 1, 3, 50])
    with torch.no_grad():
        full = m(g.data.x, g.data.edge_index, g.data.img_mask)[rows]
        xs, es, pos, mk = BatchSubgraph(g.data, g.n_img, "cpu")(rows)
        part = m(xs, es, mk)[pos]
    assert torch.allclose(full, part, atol=1e-5)


def _perturb_weights(m):
    """Thoát khỏi khởi tạo đơn vị để kiểm tra trên mô hình 'đã học' bất kỳ."""
    with torch.no_grad():
        for prm in m.parameters():
            prm.add_(0.05 * torch.randn_like(prm))


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_split_graph_half_ignores_own_image_features(bundle, kind):
    """split: nửa đồ thị của nhúng ảnh không phụ thuộc đặc trưng CLIP của chính ảnh (chỉ từ láng giềng).
    GAT ngoại lệ có chủ ý: đặc trưng ảnh (đỉnh đích) chỉ chọn TRỌNG SỐ attention giữa các nhãn láng giềng; đầu ra vẫn là
    tổ hợp lồi của tin nhắn từ nhãn → no_edge vẫn là hằng (test_split_no_edge_is_adapter_only phủ cả GAT)."""
    g = build_graph(bundle, "full")
    torch.manual_seed(0)
    m = GraphEncoder(bundle.image_emb.shape[1], kind, arch="split").eval()
    _perturb_weights(m)
    x2 = g.data.x.clone()
    x2[: g.n_img] = torch.randn_like(x2[: g.n_img])
    d = bundle.image_emb.shape[1]
    with torch.no_grad():
        a = m(g.data.x, g.data.edge_index, g.data.img_mask)[: g.n_img, d:]
        b = m(x2, g.data.edge_index, g.data.img_mask)[: g.n_img, d:]
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("kind", ["gcn", "gat", "sage"])
def test_split_no_edge_is_adapter_only(bundle, kind):
    """split + no_edge: điểm đồ thị là hằng theo ảnh → thứ hạng = thứ hạng của riêng adapter cos(Pq, Sx)."""
    g = build_graph(bundle, "no_edge")
    torch.manual_seed(0)
    m = GraphEncoder(bundle.image_emb.shape[1], kind, arch="split").eval()
    _perturb_weights(m)
    q = torch.from_numpy(bundle.caption_emb[:20])
    with torch.no_grad():
        s = m.query(q) @ m(g.data.x, g.data.edge_index, g.data.img_mask)[: g.n_img].T
        ad = torch.nn.functional.normalize(m.q_proj(q), dim=-1) @ \
            torch.nn.functional.normalize(m.skip(g.data.x[: g.n_img]), dim=-1).T
    diff = s - ad
    assert torch.allclose(diff, diff[:, :1].expand_as(diff), atol=1e-5)


def test_split_without_adapter_freezes_projections(bundle):
    m = GraphEncoder(bundle.image_emb.shape[1], "gat", arch="split", adapter=False)
    assert not m.skip.weight.requires_grad and not m.q_proj.weight.requires_grad and m.g_proj.weight.requires_grad


def test_clip_graph_end_to_end(tmp_path):
    run_dir = main(["--exp", "clip_graph", "--limit", "5", "--runs-dir", str(tmp_path), "--no-publish",
                    "--set", f"fake={json.dumps(SMALL)}", "--set", "n_explain=2", "--set", "gnn.kind=sage",
                    "--set", "gnn.arch=split"])
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["run"]["error"] is None
    for key in ("clip", "clip_graph", "graph_only"):
        assert 0 <= m["metrics"][key]["all"]["MRR"] <= 1
    ex = json.loads((run_dir / "explanations.json").read_text())
    assert len(ex) == 2 and ex[0]["paths"]


def _tiny_rel_bundle():
    """3 ảnh train của ví dụ trong JOURNAL: A (man wearing shirt, man holding controller),
    B (man wears shirt, shirt on man), C (woman wearing shirt, man holding controller)."""
    from mmxai.data.contract import Bundle
    labs = ["man", "shirt", "controller", "woman"]
    rels = {0: [("man", "wearing", "shirt"), ("man", "holding", "controller")],
            1: [("man", "wears", "shirt"), ("shirt", "on", "man")],
            2: [("woman", "wearing", "shirt"), ("man", "holding", "controller")]}
    return Bundle(image_ids=[0, 1, 2], image_emb=np.zeros((3, 4), np.float32),
                  image_labels={0: ["man", "shirt", "controller"], 1: ["man", "shirt"], 2: ["woman", "shirt", "man"]},
                  image_rels=rels, captions=[], caption_emb=np.zeros((0, 4), np.float32),
                  splits={"train": [0, 1, 2], "val": [], "test": []}, labels=labs,
                  label_emb=np.eye(4, dtype=np.float32))


def test_rel_merge_undirected_no_duplicate_edges():
    """'man wearing shirt' và 'shirt on man' là một cặp → mỗi chiều đúng 1 cạnh (bản directed cũ: 2 cạnh)."""
    b = _tiny_rel_bundle()
    n = len(b.image_ids)
    man, shirt = n + 0, n + 1

    def count(g, s, d):
        ei, et = g.data.edge_index, g.data.edge_type
        return int(((ei[0] == s) & (ei[1] == d) & (et == SG_REL)).sum())

    old, new = build_graph(b, "full", rel_merge="directed"), build_graph(b, "full", rel_merge="undirected")
    assert count(old, man, shirt) == 2 and count(old, shirt, man) == 2       # lỗi cũ, giữ để tái lập kết quả VG
    assert count(new, man, shirt) == 1 and count(new, shirt, man) == 1
    assert int((new.data.edge_type == SG_REL).sum()) == 6                   # 3 cặp {man,shirt},{man,controller},{woman,shirt}
    assert new.rel_name[(man, shirt)] == "wearing" and new.rel_name[(shirt, man)] == "wearing⁻¹"
    with pytest.raises(ValueError):
        build_graph(b, "full", rel_merge="sideways")
