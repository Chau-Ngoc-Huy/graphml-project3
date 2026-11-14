# CLAUDE.md — mmxai-lab

Project 3 (Graph ML): truy vấn text→image có giải thích bằng đồ thị. Repo này là phần **Nhiệm vụ 2 cấp 1 của Huy**
(đồ thị, GNN, rerank, ablation) và là pipeline đánh giá dùng chung. Không có repo paper gốc (`vendor/` không dùng).

Đọc trước: `docs/paper_src/Project03_Multimodal_XAI.pdf` (đề), `docs/plan.md` (tuần, trạng thái, hạn chế),
`docs/data-contract.md` (định dạng dữ liệu), `results/JOURNAL.md` (lượt mới nhất ở trên).

## Vòng lặp

```
make local EXP=<exp> ARGS="--limit 5"        # máy local, dữ liệu giả nếu không --set data=
make push  EXP=<exp> ARGS="--set data=<slug> --limit 5"   # Kaggle thử; make wait; make pull; tools/show_run.py
make push  EXP=<exp> ARGS="--set data=<slug>"              # lượt đầy đủ (3 seed) sau khi lượt thử sạch
make summary && make table                   # SUMMARY.md, results/tables/tab_*.{md,tex} → đoạn JOURNAL, commit results/
```

Thí nghiệm = `src/mmxai/experiments/<exp>.py` + `configs/experiments/<exp>.yaml` (ablation/biến thể chỉ là config kế thừa
`clip_graph.yaml` + module 2 dòng `from .clip_graph import run`).

## Luật (từ đề, mục 6 — thầy sẽ hỏi)

- Caption của ảnh val/test không dùng huấn luyện; early stopping và α chọn trên **val**; test chỉ để báo cáo.
- Cạnh thống kê (quan hệ scene graph gộp theo nhãn) chỉ từ ảnh **train** (`build_graph`, có test).
- Ảnh chỉ nhận tin (cạnh nhãn→ảnh) → đặc trưng ảnh val/test không lan sang đỉnh khác (có test `test_images_never_send`).
- Mọi mô hình dùng chung `splits.json` và `retrieval/core.py`; ≥ 3 seed cho lượt chính, báo mean ± std.
- Giải thích phải đến từ chính cơ chế tính điểm (`graph/explain.py` dùng attention/bậc của GNN đã huấn luyện) và phải
  qua kiểm tra fidelity.
- Không commit `data/`, `runs/`, `cache/`, `build/`, `local.mk`, `.env`, token. Commit `results/**/*.json`, JOURNAL, code, docs.
- Trên CPU giới hạn `cpu_threads` (mặc định 4): 16 luồng chậm hơn ~15 lần với đồ thị nhỏ.
