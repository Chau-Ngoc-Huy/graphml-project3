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

## #0 · 2026-11-14 · khởi tạo

- Dựng harness bằng skill `paper-lab`; `make test` và `make local EXP=smoke` chạy được.
- Kế hoạch: `docs/plan.md`. Bảng cần tái lập: `docs/paper_numbers.yaml`.
