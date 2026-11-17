"""Điểm đồ thị phụ thuộc truy vấn (graph/qa.py)."""

import numpy as np
import pytest
import torch

from mmxai.data.contract import load
from mmxai.data.fake import generate
from mmxai.graph.build import build_graph
from mmxai.graph.qa import QAData, QAScorer, qa_scores


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    generate(d, n_images=200, n_concepts=4, children_per_concept=3, n_extra=6, dim=32, seed=0)
    b = load(d)
    # khái niệm của caption = 2 nhãn đầu của ảnh đích (câu "trực tiếp")
    # caption_id của dữ liệu thật là số nguyên (idx của Thắng) → dùng vị trí caption làm id
    cap_conc = {k: b.image_labels[c["image_id"]][:2] for k, c in enumerate(b.captions)}
    return b, cap_conc


def _model(b, kind="gat", **kw):
    torch.manual_seed(0)
    return QAScorer(b.image_emb.shape[1], kind, hidden=16, heads=2, **kw).eval()


def test_direct_concept_matches_own_image_at_init(setup):
    """Trước khi học (gate nhỏ, C = I): khái niệm có trong ảnh ⇒ m(c, i) gần cos = 1, cao hơn ảnh không chứa nó."""
    b, cap_conc = setup
    g = build_graph(b, "full")
    qd = QAData(g, cap_conc)
    m = _model(b)
    with torch.no_grad():
        H = m.label_h(qd.x_lab, qd.ei_lab)
        M = m.concept_image_match(H, torch.arange(len(g.labels)), qd.img_labels)
    has = torch.zeros_like(M, dtype=torch.bool)
    for i in range(g.n_img):
        for l in qd.img_labels[i][qd.img_labels[i] >= 0]:
            has[l, i] = True
    assert M[has].mean() > M[~has].mean() + 0.1


@pytest.mark.parametrize("kind", ["gcn", "gat", "sage"])
def test_no_edge_graph_score_is_zero(setup, kind):
    b, cap_conc = setup
    g = build_graph(b, "no_edge")
    qd = QAData(g, cap_conc)
    m = _model(b, kind)
    cc = qd.cap_concepts[: 10]
    with torch.no_grad():
        gs = m.graph_matrix(m.label_h(qd.x_lab, qd.ei_lab), cc, qd.img_labels[:30])
    assert torch.all(gs == 0)


def test_eval_path_matches_train_path(setup):
    """qa_scores (đánh giá, tính sẵn ma trận khái niệm×ảnh) = adapter + w·graph_matrix (đường huấn luyện)."""
    b, cap_conc = setup
    g = build_graph(b, "full")
    qd = QAData(g, cap_conc)
    m = _model(b)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.from_numpy(b.image_emb)
    caps = b.captions[:8]
    cid = np.arange(len(caps))
    q = b.caption_emb[:8]
    rng = np.random.default_rng(0)
    cand = rng.integers(0, g.n_img, size=(8, 5))
    fast = qa_scores(m, qd, x, q, cid, cand, "cpu")
    with torch.no_grad():
        H = m.label_h(qd.x_lab, qd.ei_lab)
        for j in range(8):
            rows = torch.from_numpy(cand[j])
            gs = m.graph_matrix(H, qd.cap_concepts[cid[j]][None], qd.img_labels[rows])[0]
            ad = m.adapter_cos(torch.from_numpy(q[j:j + 1]), x[rows])[0]
            assert np.allclose(fast[j], m.total(ad, gs).numpy(), atol=1e-5)
