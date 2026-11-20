# Kế hoạch — Nhiệm vụ 2 cấp 2 (Khánh): phương pháp mới + bộ truy vấn

Đề: `../../shared/Project03_Multimodal_XAI.pdf` (mục 5 cấp 2, 6, 7). Phân công: `../../shared/phan-cong.docx`.
Trạng thái: ⬜ chưa · 🔶 đang · ✅ xong.

## Việc của Khánh (cấp 2, 4 mục)
1. Tái lập một phương pháp từ 2024 (AdaMF-MAT), tích hợp vào pipeline và split, chạy trên ≥2 tập công khai.
2. Bộ ≥20 truy vấn đánh giá, tách trực tiếp / gián tiếp, mô tả cách xây dựng và kiểm tra.
3. Báo cáo hai nhóm riêng, phân tích vì sao đồ thị giúp (hoặc không giúp) ở nhóm gián tiếp.
4. Phối hợp Huy để so sánh công bằng với baseline cấp 1.

## Tuần 1 — Thiết kế phương pháp hai tầng ✅
`design.md`: CLIP top-K → điểm đồ thị → rerank α·z(cos)+(1−α)·z(graph); cấp 1 (GNN của Huy) và cấp 2 (AdaMF-MAT) cắm vào cùng
một khe `score_fn`.

## Tuần 2 — Triển khai truy vấn 🔶
- ✅ Tái lập AdaMF-MAT trên MKG-W, MKG-Y: khớp paper trong ±1 điểm (`../results/adamf_mat_repro.md`, Kaggle phiên bản 6).
- ✅ Bộ 24 truy vấn (12 trực tiếp, 12 gián tiếp) + mô tả (`../queries/`).
- ✅ Bộ chuyển sang giao diện của Thắng (`src/adamf_protocol.py`): dựng MMKG từ split + đặc trưng của Thắng, huấn luyện AdaMF-MAT,
  chấm điểm ứng viên, đánh giá bằng `Protocol.evaluate`. Test trên dữ liệu giả; chạy thử trên bundle thật (CPU, mô hình ngẫu nhiên)
  tái hiện đúng số baseline của Thắng.
- ✅ Lượt chạy thật trên Kaggle (2026-11-16): `kaggle/protocol_adamf_kaggle.ipynb`, kết quả `../results/protocol_coco_full.md`.

## Tuần 3 — Tích hợp, so sánh, ablation 🔶
1. ✅ Bảng trực tiếp / gián tiếp / no_concept từ `by_group` của giao thức (CLIP vs baseline khớp tên vs AdaMF-MAT) + kiểm định ghép cặp:
   AdaMF-MAT > CLIP một chút (nhờ nhóm direct), **không** giúp nhóm indirect, không hơn baseline khớp tên. Cần phân tích/giải thích trong báo cáo.
2. So sánh công bằng với CLIP+Graph của Huy (cùng split, fold, top-K) — giao kết quả + mã cho Huy.
3. Ablation cạnh quan hệ (xáo / bỏ): thuộc phần Huy; code đã hỗ trợ (`--variant shuffle|drop`) nếu cần.
4. Đánh giá bộ 24 truy vấn trên giao thức chung (`multi_relevant_metrics`), nếu thời gian.

## Tuần 4 — Chốt ⬜
Phần báo cáo của Khánh (phương pháp, tái lập, bộ truy vấn, kết quả, hạn chế); ôn vấn đáp (cách tạo cạnh, vì sao không rò rỉ, vì sao
baseline không dùng quan hệ cũng mạnh); hỗ trợ Dương chạy phương pháp tốt nhất trên MMKG tiếng Việt.

## Phụ thuộc
| Cần từ | Gì | Tình trạng |
|---|---|---|
| Thắng | bundle (`data/`, `split/`, `features/`), `mmxai_eval.py` | ✅ có (`task1-bundle` trên Kaggle) |
| Yến | split chính thức, nhãn đã chuẩn hóa, dataset thứ hai | ⬜ đang dùng split của Thắng (Karpathy ∩ VG, seed 0) |
| Huy | số CLIP+Graph, ablation | ⬜ giao sau lượt chạy |
| Dương | cần phương pháp tốt nhất của tôi cho MMKG tiếng Việt | ⬜ cuối tuần 3–4 |

## Hạn chế đã biết
- Mô hình huấn luyện 1 seed; 3 seed của giao thức là 3 fold gallery.
- Ground truth truy vấn VG lấy từ chú thích, điểm đồ thị cũng dùng nhãn đối tượng đã chú thích → baseline chỉ đếm đối tượng rất mạnh
  (thí nghiệm riêng: `../results/level2_vg_own_experiment.md`). Kiểm tra ẩn đối tượng (`kaggle/exploratory/level2_hidden_kaggle.ipynb`) chưa chạy.
- Bộ 24 truy vấn chưa được người kiểm tra ảnh (xem `../queries/README.md`).
- AdaMF-MAT là KG completion: ghép vào truy vấn ảnh bằng độ hợp lý của (ảnh, contains, concept); đây là lựa chọn thiết kế của tôi.
