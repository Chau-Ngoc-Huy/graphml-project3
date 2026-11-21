# Phân công viết báo cáo

Mỗi người giữ trọn một khối liền nhau từ trên xuống trong `report/report.tex`, tự lấy số từ kho mã, không phải chờ người khác.

| Khối | Người | Các mục | Lấy số / nội dung từ đâu |
|---|---|---|---|
| 1 | **Yến** | Tóm tắt · I Giới thiệu · II Công trình liên quan · III Dữ liệu và độ đo | `README.md`, `docs/PIPELINE.md`, `data/psg_bundle/psg_stats.json`; số chính trong `results/tables/` |
| 2 | **Huy** | IV Phương pháp (A–E) · V Thiết lập thực nghiệm | Mã `src/mmxai/` (QA-Graph, AdaMF-MAT, `xai_bridge.py`), `configs/`, Bảng I hiện có |
| 3 | **Khánh** | VI Kết quả (A–E): so sánh tổng thể VG/PSG, ablation, trực tiếp/gián tiếp, tái lập MKG-W/Y, đánh giá giải thích | `results/tables/vg_official.md`, `results/tables/psg_official.md`, `results/psg_vs_adamf.md`, `results/adamf/`; số giải thích: `make xai DS=psg N=10` |
| 4 | **Dương** | VII Phân tích trường hợp · VIII Thảo luận · IX Hạn chế và hướng phát triển · X Kết luận | 5 thành công + 5 thất bại lấy từ MMXAI Lab (`make ui`, `/api/explain`: hạng CLIP → hạng QA-Graph + đường đi giải thích); `results/JOURNAL.md` |
| 5 | **Thắng** | XI Phân công · XII Khai báo · Phụ lục A, B, C · Tài liệu tham khảo · **phụ trách định dạng toàn báo cáo** | Phụ lục B chép từ `docs/PIPELINE.md`; Phụ lục C đối chiếu với `docs/assignment/Project03_Multimodal_XAI.pdf` |

## Quy ước chung (giữ đúng như bản hiện tại để các khối liền mạch)

- Tên bộ dữ liệu: **VG∩COCO**, **PSG∩COCO**.
- Tên mô hình: **CLIP**, **CLIP + Adapter**, **QA-Graph**, **CLIP + khớp tên**, **AdaMF-MAT**.
- Ký hiệu: $r$ (hạng ảnh đúng), $K$ (số ứng viên), $\alpha$, $s$ và công thức (1) ở mục IV-A.
- Thuật ngữ: "quy trình đánh giá", "truy vấn trực tiếp / gián tiếp", "toàn bộ ảnh test", "phân tích loại bỏ (ablation)".
- Số liệu: MRR tính theo %, dấu phẩy thập phân (64,76).

## Cách làm

1. Làm trên nhánh `final`: `git pull` trước khi sửa.
2. Chỉ sửa trong các mục của mình trong `report/report.tex`.
3. Chạy `make report` (→ `build/report/report.pdf`), không lỗi mới push.
4. Xóa hết `\todo{...}` / `[CẦN BỔ SUNG]` trong mục của mình trước khi báo xong.

## Phụ trách định dạng (Thắng) — rà sau khi các khối đã ghép

- Bảng và hình được nhắc trong chữ trước khi xuất hiện; chú thích bảng đặt **trên**, chú thích hình đặt **dưới**; chú thích tự đủ nghĩa (bộ dữ liệu, độ đo, số seed, chữ in đậm nghĩa là gì).
- Dấu thập phân thống nhất: hình hiện dùng dấu chấm (60.40) — sửa trong `report/make_figs.py`.
- Cùng một con số, cùng một tên gọi giống nhau ở tóm tắt, bảng, chữ và kết luận.
- Không còn `\todo` / `[CẦN BỔ SUNG]`; không còn tên người ngoài mục XI.
- Tài liệu tham khảo đúng chuẩn IEEE (tách mục [13] đang gộp hai tài liệu).
- Kiểm tra số trang theo đề ("khoảng 12–18 trang"; hiện 7 trang hai cột).
- Nhận xét chi tiết đã có: `report/reviews/REVIEW-r1-2026-11-21.md`.
