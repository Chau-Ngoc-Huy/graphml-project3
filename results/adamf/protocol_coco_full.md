# AdaMF-MAT trên giao thức chung của Thắng (Karpathy COCO ∩ Visual Genome) — lượt chạy thật

Notebook: `kaggle/protocol_adamf_kaggle.ipynb` (Kaggle, T4, chạy 2026-11-16). Dữ liệu: bundle của Thắng (19,264 ảnh, 96,381 caption).
Số liệu thô (có hạng từng truy vấn, dùng cho kiểm định ghép cặp với Huy): `protocol_coco_full.json`; báo cáo chạy: `protocol_coco_full_report.txt`.

**Thiết lập.** MMKG: 2,000 concept + 10,264 ảnh (6,000 ảnh train + toàn bộ val/test), 31 quan hệ, 193,962 bộ ba train
(176,736 `contains` + 17,226 quan hệ concept từ ảnh train). AdaMF-MAT huấn luyện 104 epoch (85.8 s/epoch, 1 seed, lr 1e-4, dim 200).
Chấm điểm: Σ độ hợp lý của (ảnh, contains, concept) trên các concept của caption; kết hợp `α·z(cos)+(1−α)·z(graph)`, α chọn trên val
(0.75 cho AdaMF-MAT, 0.8 cho baseline khớp tên). Giao thức: 3 fold gallery 1,000 ảnh (seed 0,1,2), top-100 CLIP, 5,001 truy vấn/fold.

## Text → image, test, 1K gallery, mean ± std 3 fold
| Phương pháp | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| CLIP | 57.36 ± 0.95 | 84.08 ± 0.24 | 91.76 ± 0.33 | 69.23 ± 0.68 | 74.32 ± 0.55 |
| CLIP + khớp tên đối tượng (không quan hệ) | 58.46 ± 1.35 | 84.92 ± 0.46 | 92.42 ± 0.23 | 70.21 ± 0.84 | 75.24 ± 0.61 |
| **CLIP + AdaMF-MAT** | 58.04 ± 1.09 | 84.65 ± 0.37 | 92.16 ± 0.30 | 69.77 ± 0.74 | 74.84 ± 0.53 |

Toàn pool test (2,138 ảnh), kiểm định ghép cặp so với CLIP: AdaMF-MAT MRR **+0.64** (CI95 [0.33, 0.97]), R@1 +0.64 ([0.14, 1.15], p=0.013),
R@10 +0.57 ([0.20, 0.94]); khớp tên MRR +1.06 ([0.74, 1.37]).

## Theo nhóm truy vấn (nhóm do giao thức tự gán), MRR mean ± std
| Nhóm | n | CLIP | CLIP + khớp tên | CLIP + AdaMF-MAT |
|---|---|---|---|---|
| direct | 933 | 69.33 ± 0.89 | **77.70 ± 0.96** | 75.17 ± 1.03 |
| indirect | 4,054 | 69.24 ± 0.65 | 68.52 ± 0.84 | 68.56 ± 0.71 |
| no_concept | 13 | 60.87 | 60.87 | 60.87 |

## Kết luận (trung thực)
1. AdaMF-MAT cải thiện CLIP một chút, có ý nghĩa thống kê (+0.64 MRR toàn pool, CI không chứa 0).
2. Cải thiện đến từ nhóm **direct** (+5.8 MRR), nơi mọi concept của caption có trong danh sách đối tượng của ảnh đúng. Đó là thông tin
   **hiện diện đối tượng**, không phải suy luận quan hệ.
3. Ở nhóm **indirect** (nhóm lẽ ra kiểm tra vai trò quan hệ), AdaMF-MAT **không tốt hơn CLIP** (68.56 so với 69.24) và ngang baseline khớp tên.
4. Baseline **không dùng quan hệ** (khớp tên) bằng hoặc hơn AdaMF-MAT ở mọi chỉ số tổng. Vậy kết quả này **không ủng hộ** nhận định
   "đồ thị giúp ở nhóm gián tiếp"; trùng kết luận thí nghiệm riêng trên VG (`level2_vg_own_experiment.md`).

## Hạn chế / cần kiểm tra thêm
- Quan hệ học được yếu: kiểm tra KGC trên quan hệ concept giữ lại cho MRR 0.072, Hits@10 0.21, Hits@1 ≈ 0 (104 epoch trên 194k bộ ba, ít hơn nhiều
  so với 1000 epoch khi tái lập). Xếp hạng tính trên mọi thực thể gồm cả ảnh nên không so được với benchmark. Có thể là lý do không suy luận được quan hệ; chưa chứng minh.
- Nhóm "indirect" do giao thức gán theo chú thích (có ≥1 concept không nằm trong danh sách đối tượng của ảnh), phần lớn là đồng nghĩa/thiếu chú thích, không phải suy luận quan hệ thật.
- 1 seed huấn luyện; 3 seed của giao thức là 3 fold gallery. Chưa chạy ablation (xáo/bỏ cạnh): thuộc phần Huy, code đã sẵn (`--variant`).
- Chưa đánh giá bộ 24 truy vấn trên giao thức này.
