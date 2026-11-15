# Nhật ký thực nghiệm — mmxai-lab

Viết tay, lượt mới nhất ở trên. Mỗi lượt đầy đủ (hoặc mỗi kết luận từ nhiều lượt thử) một đoạn. Số lấy từ `metrics.json` /
`REPRODUCTION.md`, không ước lượng. Paper: Project 3 - Truy vấn đa phương thức có giải thích bằng đồ thị (Graph ML).

## Mẫu đoạn

```
## #N · YYYY-MM-DD · <run_id> (`--tag ...`)

- **Mục đích:** tái lập Bảng X dòng Y / kiểm tra giả thuyết Z / ablation ...
- **Cấu hình:** model, dữ liệu, tham số khác mặc định (`--set ...`), compute (Kaggle T4 / Colab), commit vendor.
- **Kết quả:** metric chính (paper: a · mình: b · Δ) ; thời gian/mẫu, VRAM; lỗi parse/crash nếu có.
- **Nhận xét & kết luận:** khớp/không khớp vì sao; khác biệt so với paper; bước tiếp theo.
- **Output:** `runs/<exp>/<run_id>/`
```

## #1 · 2026-11-14 · kiểm tra pipeline trên dữ liệu GIẢ (local CPU, 1 seed, chưa phải kết quả báo cáo)

- **Mục đích:** xác nhận pipeline phát hiện được đóng góp của cấu trúc đồ thị trước khi tốn GPU cho VG thật.
- **Cấu hình:** `clip_graph` (GAT 2 tầng), dữ liệu giả 2000 ảnh / 4000 caption / 80 nhãn (`img_noise=0.15, cap_noise=0.1`),
  `train.batch_size=64, patience=8, epochs≤60`, seed 0; biến thể `full`, `no_edge`, `shuffle`.
- **Kết quả (MRR test, all / direct / indirect):** CLIP 0.321 / 0.459 / 0.103 · full 0.399 / 0.509 / 0.223 ·
  no_edge 0.332 / 0.459 / 0.130 · shuffle 0.331 / 0.467 / 0.116. Fidelity (full): xóa cạnh giải thích Δ ≈ 0.10–0.13,
  xóa ngẫu nhiên cùng số cạnh Δ ≈ 0.0001. ~70 s/seed trên CPU 4 luồng.
- **Nhận xét:** cải thiện tập trung ở nhóm gián tiếp (×2 MRR) và biến mất khi bỏ hoặc xáo cạnh → pipeline tách được vai trò
  của cấu trúc, đúng thiết kế dữ liệu giả. Hai lỗi sửa trong lúc làm: (1) huấn luyện cả đồ thị mỗi bước quá chậm → đồ thị con
  theo batch (chính xác, có test); (2) torch 16 luồng chậm ~15× → `cpu_threads: 4`. Kernel Kaggle sửa lỗi `VENDOR["name"]=None`
  và cài `torch_geometric` qua `vendor.yaml: pip`.
- **Tiếp theo:** nhận dữ liệu VG thật (Yến) + features (Thắng) theo `docs/data-contract.md`; `make push --limit 5` trên Kaggle.
- **Output:** `runs/clip_graph/20261114-04*` (local, không commit).

## #0 · 2026-11-14 · khởi tạo

- Dựng harness bằng skill `paper-lab`; `make test` và `make local EXP=smoke` chạy được.
- Kế hoạch: `docs/plan.md`. Bảng cần tái lập: `docs/paper_numbers.yaml`.
