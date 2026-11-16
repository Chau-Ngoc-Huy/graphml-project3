# Nhật ký thí nghiệm — khanh/ (mới nhất ở trên)

## 2026-11-15 · AdaMF-MAT trên giao thức chung (notebook `kaggle/protocol_adamf_kaggle.ipynb`)
- Lần 1 hỏng ở stage 1: không tìm thấy dữ liệu của Thắng trong `/kaggle/input` (chưa gắn bundle). Đã thêm: liệt kê input, đường dẫn
  `task1-bundle`, dự phòng chạy lại notebook của Thắng.
- Chạy thử trên **bundle thật, CPU, mô hình khởi tạo ngẫu nhiên** (chỉ để kiểm tra đường chạy): 272 giây; baseline CLIP trên toàn pool
  test cho R@1 48.31, MRR 60.40 và baseline khớp tên cho MRR +1.06 [0.74, 1.37] — **trùng số trong README của Thắng**.
- **2026-11-16 · lượt huấn luyện thật xong** (104 epoch, 2.5 giờ): CLIP+AdaMF-MAT MRR 69.77 vs CLIP 69.23 (toàn pool +0.64, CI95 [0.33, 0.97]),
  nhưng chỉ nhờ nhóm direct (75.17 vs 69.33); nhóm indirect 68.56 < CLIP 69.24; baseline khớp tên (không quan hệ) 70.21 ≥ AdaMF-MAT.
  → **không ủng hộ** "đồ thị giúp ở nhóm gián tiếp". Chi tiết và hạn chế: `protocol_coco_full.md`; số thô: `protocol_coco_full.json`.
  TODO: thêm link phiên bản Kaggle của lượt này.

## 2026-11-15 · Thí nghiệm riêng trên Visual Genome (không dùng giao thức chung) → `level2_vg_own_experiment.md`
- 24 truy vấn, gallery 1,133 ảnh test, 1 seed. CLIP+AdaMF-MAT tốt hơn CLIP nhưng **baseline chỉ đếm đối tượng (không quan hệ) tốt hơn
  AdaMF-MAT** (MRR .881 vs .677). Lý do: ground truth và điểm đồ thị cùng dùng nhãn đối tượng đã chú thích.
- Kiểm tra ẩn đối tượng + ablation (xáo/bỏ cạnh) đã chuẩn bị (`kaggle/exploratory/level2_hidden_kaggle.ipynb`), chưa chạy (ngoài phạm vi nhiệm vụ).

## 2026-11-14 · Tái lập AdaMF-MAT (Kaggle phiên bản 6)
https://www.kaggle.com/code/khnhnguyninhquang/adamf-mat-kaggle?scriptVersionId=369425081
- MKG-W: MRR 35.68 / Hits@1 28.98 / Hits@3 38.68 / Hits@10 48.13 (paper 35.85 / 29.04 / 39.01 / 48.42).
- MKG-Y: MRR 39.10 / Hits@1 34.59 / Hits@3 41.23 / Hits@10 46.68 (paper 38.57 / 34.34 / 40.59 / 45.76).
- 1000 epoch, seed 42, 1 seed; mọi chênh lệch < 1 điểm. Chi tiết: `adamf_mat_repro.md`.
