"""Bộ chuyển AdaMF-MAT: dựng MMKG từ Protocol của Thắng đúng định dạng và không rò rỉ."""
import json

import numpy as np
import pytest

from conftest import fake_text_encode

torch = pytest.importorskip("torch")
from mmxai.adamf import adamf_protocol as A  # noqa: E402
from mmxai_eval import Protocol  # noqa: E402  (../thang, đọc không sửa)


def build(work_dir, tmp_path, variant="full"):
    P = Protocol.load(str(work_dir))
    repo = tmp_path / "repo"
    meta = A.build_mmkg(P, str(repo), "T" + A.SUFFIX[variant], fake_text_encode, variant, n_train_images=150,
                        min_triple=1, top_pred=10, run_nn=False)
    return P, repo, meta


def rows(repo, name, fname):
    lines = (repo / "benchmarks" / name / fname).read_text().splitlines()
    assert int(lines[0]) == len(lines) - 1  # dòng đầu = số bộ ba (định dạng OpenKE)
    return [tuple(map(int, l.split())) for l in lines[1:]]


def test_files_consistent(work_dir, tmp_path):
    P, repo, meta = build(work_dir, tmp_path)
    ent = int((repo / "benchmarks" / "T" / "entity2id.txt").read_text().splitlines()[0])
    rel = int((repo / "benchmarks" / "T" / "relation2id.txt").read_text().splitlines()[0])
    tr = rows(repo, "T", "train2id.txt")
    assert max(max(h, t) for h, t, _ in tr) < ent and max(r for _, _, r in tr) < rel
    vis, txt = torch.load(repo / "embeddings" / "T-visual.pth"), torch.load(repo / "embeddings" / "T-textual.pth")
    assert vis.shape[0] == txt.shape[0] == ent
    assert len(rows(repo, "T", "test2id.txt")) >= 1


def test_no_val_test_relations_in_train(work_dir, tmp_path):
    """Quan hệ concept→concept trong train chỉ được đếm từ ảnh train; ảnh val/test chỉ góp 'contains'."""
    P, repo, meta = build(work_dir, tmp_path)
    C = meta["n_concepts"]
    train_rel = {(meta["vocab"][h], meta["preds"][r - 1], meta["vocab"][t]) for h, t, r in rows(repo, "T", "train2id.txt") if r != 0}
    from_train = {tuple(x) for ii in P.pool("train") for x in P.images[ii]["relations"]}
    assert train_rel <= from_train
    assert all(h >= C for h, t, r in rows(repo, "T", "train2id.txt") if r == 0)  # contains: đầu là ảnh


def test_variants(work_dir, tmp_path):
    tr, rel = {}, {}
    for v in ("full", "shuffle", "drop"):
        P, repo, meta = build(work_dir, tmp_path, v)
        tr[v] = rows(repo, "T" + A.SUFFIX[v], "train2id.txt")
        rel[v] = [x for x in tr[v] if x[2] != 0]  # quan hệ concept→concept (r != contains)
    contains = {v: [x for x in tr[v] if x[2] == 0] for v in tr}
    assert contains["full"] == contains["shuffle"] == contains["drop"]  # cạnh ảnh–concept y hệt ở cả 3 biến thể
    assert rel["drop"] == []  # drop: không còn quan hệ concept
    assert len(rel["full"]) == len(rel["shuffle"]) > 0  # shuffle: cùng số bộ ba
    assert sorted((h, r) for h, _, r in rel["full"]) == sorted((h, r) for h, _, r in rel["shuffle"])  # giữ đầu + nhãn quan hệ
    assert sorted(t for _, t, _ in rel["full"]) == sorted(t for _, t, _ in rel["shuffle"])  # giữ tập đuôi, chỉ hoán vị
    assert rel["full"] != rel["shuffle"]  # nhưng cấu trúc quan hệ bị phá
