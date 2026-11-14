# mmxai-lab — Truy vấn text→image có giải thích bằng đồ thị (Project 3, Graph ML)

Pipeline Nhiệm vụ 2 cấp 1: **CLIP top-k → GNN (GCN/GAT/GraphSAGE) trên đồ thị ảnh–nhãn–khái niệm → xếp hạng lại
α·cos + (1−α)·graph_score → đường đi giải thích + fidelity**. Chạy được ngay trên dữ liệu giả cùng định dạng Visual Genome;
dữ liệu thật chỉ cần đổi đường dẫn (`docs/data-contract.md`).

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # máy không GPU
.venv/bin/pip install -e '.[dev]'                                             # + '.[clip]' nếu trích đặc trưng CLIP
cp local.mk.example local.mk && cp .env.example .env                           # LAB_USER, KAGGLE_USER, DATASET
make test
```

## Chạy

```bash
make local EXP=clip_baseline                         # CLIP thuần (dữ liệu giả)
make local EXP=clip_graph ARGS="--limit 5"            # lượt thử: 1 seed, 2 epoch
make local EXP=clip_graph                             # GAT, 3 seed, mean ± std
make local EXP=ablation_shuffle                       # ablation (no_rel, no_concept, no_edge, shuffle, random)
make local EXP=clip_graph ARGS="--set data=data/vg_real --set gnn.kind=sage --set train.lr=5e-4"
make table                                            # bảng Markdown + LaTeX → results/tables/
```

Trên Kaggle (GPU T4): `make push EXP=clip_graph ARGS="--set data=<tên dataset> --limit 5 --tag thu"` → `make wait` →
`make pull` → `.venv/bin/python tools/show_run.py`. Nhiều thí nghiệm: `make queue EXPS="clip_graph clip_graph_sage"`.

## Đầu ra mỗi lượt (`runs/<exp>/<run_id>/`)

| File | Nội dung |
|---|---|
| `metrics.json` | `clip`, `clip_graph` (mean), `clip_graph_std`, `graph_only` (α=0) × nhóm `all/direct/indirect`; α; fidelity |
| `explanations.json` | top-1 của `n_explain` truy vấn test: đường đi 1–2 bước + đóng góp, fidelity (Δ khi xóa cạnh giải thích vs ngẫu nhiên), sparsity |
| `alpha_curves.json` | metric val theo α (độ nhạy) |
| `train_info.json` | loss, MRR val theo epoch, mỗi seed |
| `ranks_test.json` | hạng của ảnh đúng cho từng caption test → phân tích 5 thành công / 5 thất bại |
| `model.pt` | trọng số GNN seed đầu (cho Nhiệm vụ 3) |

## Cấu trúc

```
src/mmxai/data/       contract.py (nạp dữ liệu theo hợp đồng) · fake.py (dữ liệu giả định dạng VG)
src/mmxai/graph/      build.py (đồ thị + biến thể ablation) · gnn.py (GNN, InfoNCE, đồ thị con theo batch) · explain.py
src/mmxai/retrieval/  core.py (top-k CLIP, rerank, R@K/MRR/nDCG, chọn α trên val) — dùng chung cho mọi mô hình
src/mmxai/experiments/ clip_baseline, clip_graph(+_gcn, _sage), ablation_*  ·  configs/experiments/*.yaml
tools/                make_split, fetch_conceptnet, extract_clip, report_table · make_kernel/pull_kernel (Kaggle)
docs/                 plan.md, data-contract.md, paper_src/ (đề bài)  ·  results/JOURNAL.md
```

Harness dựng bằng skill `paper-lab` (run_id, metrics.json kèm môi trường + commit, Kaggle từ terminal).
