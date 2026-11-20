# Kế hoạch — Nhiệm vụ 2 cấp 1 (Huy): đồ thị, GNN, xếp hạng lại

Đề: `../shared/Project03_Multimodal_XAI.pdf` (mục 4.2, 5, 6, 7). Phân công: `../shared/phan-cong.docx`. Hợp đồng dữ liệu chung: `../shared/data-contract.md`.
Trạng thái: ⬜ chưa · 🔶 đang · ✅ xong.

## Tuần 1 — Thiết kế + pipeline trên dữ liệu giả ✅
- Schema đồ thị: đỉnh ảnh / nhãn / khái niệm; cạnh `contains` (nhãn→ảnh), `sg_rel` (nhãn↔nhãn, chỉ từ ảnh train),
  `concept` (ConceptNet). Ảnh chỉ nhận tin → không rò rỉ từ ảnh val/test (`src/mmxai/graph/build.py`).
- graph_score = cos(P·q, z_ảnh), z_ảnh = GNN 2 tầng (GCN/GAT/SAGE), huấn luyện InfoNCE với caption train (`graph/gnn.py`).
- Rerank α·cos + (1−α)·graph_score, α chọn trên val (`retrieval/core.py`); giải thích + fidelity (`graph/explain.py`).
- Dữ liệu giả cùng định dạng VG (`data/fake.py`); `make test` 23 test xanh; kernel Kaggle giả lập local chạy được.
- Việc còn lại: thống nhất `docs/data-contract.md` với Yến (split, synonyms) và Thắng (features, model CLIP).

## Tuần 2 — Chạy trên Visual Genome thật ⬜
1. Nhận data từ Yến/Thắng → `tools/fetch_conceptnet.py` → `tools/extract_clip.py --labels` → kiểm tra bằng
   `make local EXP=clip_baseline ARGS="--set data=<dir> --set max_images=5000"`.
2. Upload dataset Kaggle private; `make push EXP=clip_graph ARGS="--set data=<slug> --limit 5 --tag thu"` → đọc
   `sec_per_sample`, ước lượng giờ GPU, ghi JOURNAL.
3. Lượt đầy đủ GAT 3 seed (`--set train.batch_size=512`), chọn siêu tham số trên val (lr, hidden, layers, tau, topk).

## Tuần 3 — So sánh + ablation ⬜
1. `clip_graph_gcn`, `clip_graph_sage` (3 seed mỗi cái) → bảng CLIP vs CLIP+Graph.
2. `make ablations ARGS="--set data=<slug>"` (no_rel, no_concept, no_edge, shuffle, random).
3. Độ nhạy α (`alpha_curves.json`), so sánh các cách tạo cạnh (mở rộng). Giao `model.pt` + `explanations.json` cho Dương,
   pipeline + split cho Khánh (cấp 2 chạy cùng `retrieval/core.py`).

## Tuần 4 — Chốt ⬜
`make table` → `results/tables/tab_*.tex` vào báo cáo; README chạy lại; phần đồ thị/GNN trong báo cáo; ôn vấn đáp
(cách tạo cạnh, vì sao không rò rỉ, vì sao ablation chứng minh vai trò cấu trúc).

## Ngân sách GPU (Kaggle 30 h/tuần/tài khoản)
Ước lượng sau lượt `--limit` (tuần 2). Dữ liệu giả 2000 ảnh: ~70 s/seed trên CPU. Với VG đầy đủ dự kiến < 1 h/seed trên T4
nhờ huấn luyện theo đồ thị con của batch (chỉ nhãn + ảnh trong batch, chính xác bằng chạy cả đồ thị — có test).

## Hạn chế đã biết
- Xếp hạng đường đi giải thích dùng heuristic `cos(q, đỉnh đầu) × attention`; fidelity dương (xóa cạnh giải thích làm
  điểm giảm ≫ xóa ngẫu nhiên) nhưng đường 2 bước kiểu `child -IsA⁻¹-> boy -contains-> ảnh` chưa được ưu tiên đúng.
  Hướng cho Nhiệm vụ 3: đo đóng góp bằng occlusion/gradient trên cạnh, hoặc GNNExplainer làm tham chiếu.
- Truy vấn chỉ nối vào đồ thị qua nhúng (P·q), chưa trích thực thể từ câu truy vấn để chọn đường đi tường minh.
