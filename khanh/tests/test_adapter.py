"""Adapter: khởi tạo = đơn vị (điểm = cosine CLIP), huấn luyện chạy được trên dữ liệu giả, hình dạng đúng giao thức."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import adapter as AD  # noqa: E402
from mmxai_eval import Protocol  # noqa: E402


def test_identity_init_equals_clip_cosine(work_dir):
    P = Protocol.load(str(work_dir))
    c = P.candidates("test", 0)
    model = AD.Adapter(P.img.shape[1])
    s = AD.adapted_sim(model, P, c)
    assert s.shape == c["sim"].shape
    assert np.allclose(s, c["sim"], atol=1e-3)  # adapter chưa học = cosine của CLIP


def test_train_returns_best_val_state(work_dir):
    P = Protocol.load(str(work_dir))
    m = AD.train_adapter(P, seed=0, epochs=3, patience=2, batch_size=128, device="cpu", log=lambda *_: None)
    assert len(m.history) >= 1 and m.best_val_mrr >= max(h["val_MRR"] for h in m.history) - 1e-9
    c = P.candidates("test", 0)
    assert AD.adapted_sim(m, P, c).shape == c["sim"].shape
