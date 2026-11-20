# Mở rộng: adapter + AdaMF-MAT trên giao thức chung (chạy cục bộ trên CPU, dùng lại checkpoint COCO_full đã huấn luyện)

Số thô: `protocol_coco_adapter.json` (có hạng từng truy vấn). Mã: `src/adapter.py`, `src/adamf_protocol.py eval --adapter`.
Mô hình AdaMF-MAT **giữ nguyên** như `protocol_coco_full.md` (104 epoch, 1 seed); chỉ thêm adapter và chấm lại.

**Adapter (mở rộng, không thuộc AdaMF-MAT).** Hai phép tuyến tính 512×512 khởi tạo = đơn vị (caption, ảnh), InfoNCE (tau 0.05), AdamW lr 1e-3,
huấn luyện trên caption + ảnh TRAIN, chọn epoch theo MRR val (cùng ý tưởng adapter). MRR val đạt đỉnh ngay epoch 0 (63.29) rồi giảm
→ dừng sớm; **chưa tinh chỉnh siêu tham số** (lr nhỏ hơn có thể tốt hơn). Test không được dùng. α chọn trên val.

## Text → image, test, 1K gallery, mean ± std 3 fold (MRR %) và toàn pool test (MRR %)
| Phương pháp | R@1 (1K) | MRR (1K) | MRR toàn pool |
|---|---|---|---|
| CLIP | 57.36 ± 0.95 | 69.23 ± 0.68 | 60.40 |
| CLIP + AdaMF-MAT | 58.04 ± 1.09 | 69.77 ± 0.74 | 61.04 |
| CLIP + Adapter | 59.52 ± 0.74 | 71.47 ± 0.55 | 62.41 |
| CLIP + Adapter + khớp tên (không quan hệ) | 60.98 ± 0.83 | 72.47 ± 0.58 | 63.46 |
| **CLIP + Adapter + AdaMF-MAT** | 60.29 ± 0.82 | 71.99 ± 0.58 | 63.00 |

Kiểm định ghép cặp, toàn pool test (ΔMRR, CI95): Adapter+AdaMF-MAT so với CLIP **+2.60** [2.10, 3.14]; so với **Adapter** +0.59 [0.31, 0.87];
Adapter+khớp tên so với Adapter +1.05 [0.80, 1.30].

## Theo nhóm truy vấn, MRR (1K)
| Nhóm | CLIP | + AdaMF-MAT | Adapter | Adapter + khớp tên | Adapter + AdaMF-MAT |
|---|---|---|---|---|---|
| direct (n=933) | 69.33 | 75.17 | 71.47 | 76.41 | 75.46 |
| indirect (n=4054) | 69.24 | 68.56 | 71.49 | 71.59 | 71.21 |

## Kết luận
1. Thêm adapter làm số tăng (MRR toàn pool 61.04 → 63.00), nhưng **phần lớn do adapter**: Adapter đứng một mình đã +2.01 so với CLIP.
2. Phần AdaMF-MAT đóng góp trên nền adapter là **+0.59** (có ý nghĩa, nhỏ) và chỉ ở nhóm **direct** (71.47 → 75.46). Ở nhóm **indirect**
   AdaMF-MAT không giúp (71.49 → 71.21).
3. Baseline **không dùng quan hệ** (khớp tên) trên nền adapter lại tốt hơn AdaMF-MAT (+1.05 so với +0.59 toàn pool). Kết luận không đổi:
   thông tin có ích là **hiện diện đối tượng**, không phải quan hệ học được.
4. Vẫn thấp hơn QA-Graph (cấu hình tốt nhất 65.09 toàn pool, seed 0): đồ thị huấn luyện chung với adapter cho đúng bài toán tìm ảnh.

## Hạn chế
- Adapter 1 seed, chưa tinh chỉnh; AdaMF-MAT 1 seed. Adapter và AdaMF-MAT ghép bằng α tuyến tính trên z-score, không huấn luyện chung.
- Chạy trên CPU: adapter huấn luyện bằng GPU có thể cho số hơi khác.
