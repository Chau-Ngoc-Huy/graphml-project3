# CLAUDE.md — khanh/

Project 3 (Graph ML), phần **Nhiệm vụ 2 cấp 2 của Khánh**: tái lập một phương pháp từ 2024 (AdaMF-MAT) cắm vào pipeline chung, và bộ
≥20 truy vấn trực tiếp/gián tiếp. Level 1 (đồ thị, GNN, ablation) là của Huy (`../huy/`); metric/baseline/giao thức là của Thắng
(`../thang/mmxai_eval.py`); split/dữ liệu là của Yến. **Không viết lại phần của người khác**; chỉ sửa trong `khanh/`.

Đọc trước: `../shared/Project03_Multimodal_XAI.pdf` (đề), `docs/plan.md` (trạng thái), `docs/design.md` (thiết kế),
`results/JOURNAL.md` (nhật ký, mới nhất ở trên), `queries/README.md` (giới hạn bộ truy vấn).

## Luật (đề mục 6; thầy sẽ hỏi)

- Mọi phương pháp dùng chung split/fold/top-K của Thắng; α và siêu tham số chọn trên **val**, test chỉ để báo cáo.
- Cạnh quan hệ chỉ đếm từ ảnh **train**; ảnh val/test chỉ góp danh sách đối tượng (đầu vào), không góp quan hệ.
- Bộ 24 truy vấn chỉ để đánh giá: không tinh chỉnh trên đó, và các ảnh liên quan của nó không được dùng huấn luyện.
- Báo cáo mean ± std theo seed. Hiện mô hình chỉ huấn luyện 1 seed (3 seed của giao thức là 3 fold gallery).
- Mã AdaMF-MAT (zjukg) không commit vào repo (chưa kiểm giấy phép); notebook clone trên Kaggle.
- Không commit dữ liệu, ảnh, checkpoint, `.npy/.npz/.pth`, token (`~/.kaggle/access_token`, `.env`). Chỉ commit `results/**/*.json` nhỏ.

## Điều đã biết (đừng quên khi viết báo cáo)

- Tái lập AdaMF-MAT khớp paper trong ±1 điểm trên MKG-W và MKG-Y (`results/adamf_mat_repro.md`).
- Lượt chạy thật trên giao thức chung (`results/protocol_coco_full.md`): CLIP+AdaMF-MAT MRR 69.77 vs CLIP 69.23 (+0.64, có ý nghĩa)
  nhưng chỉ ở nhóm direct; nhóm indirect 68.56 < CLIP 69.24; baseline khớp tên (không quan hệ) 70.21 ≥ AdaMF-MAT. **Số liệu không ủng hộ
  "đồ thị giúp ở nhóm gián tiếp"**: báo cáo kết quả âm này, đừng viết ngược lại. Thí nghiệm riêng trên VG
  (`results/level2_vg_own_experiment.md`) cho cùng kết luận.
- Bộ truy vấn gián tiếp chưa được người kiểm tra ảnh; ghi rõ hạn chế (`queries/README.md`).

## Vòng lặp

```
make test                         # kiểm tra build MMKG, split, metric trên dữ liệu giả
make notebooks                    # sinh lại kaggle/*.ipynb từ src/ (sửa src/ rồi sinh lại, đừng sửa notebook bằng tay)
# Kaggle: import notebook → Save & Run All → tải report_protocol.txt + results_COCO_full.json → results/ + JOURNAL.md
```
