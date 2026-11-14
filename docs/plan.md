# Kế hoạch thực nghiệm — mmxai-lab

Tài liệu này dành cho Claude Code và người làm. Mỗi giai đoạn có tiêu chí xong; cuối giai đoạn dừng báo kết quả và chờ xác nhận.
Trạng thái: ⬜ chưa · 🔶 đang · ✅ xong.

## Giai đoạn A — Hiểu paper và repo ⬜
1. `docs/paper.md` đủ mục 1–6; `docs/paper_numbers.yaml` có mọi số sẽ tái lập.
2. `docs/repo-audit.md` đủ; lệnh nhỏ nhất chạy được trên máy (mục 6).

## Giai đoạn B — Harness chạy end-to-end ⬜
1. `make test` xanh; `make local EXP=smoke`; `make push EXP=smoke ARGS="--limit 4"` → `make pull` về được metrics.
2. Adapter đầu tiên (`experiments/<exp>.py`) chạy `--limit 4` local và trên Kaggle, metric đúng tên khóa trong `paper_numbers.yaml`.

## Giai đoạn C — Tái lập ⬜
1. Chạy đầy đủ từng dòng ưu tiên cao của bảng chính; `make compare`; JOURNAL mỗi lượt.
2. Số không khớp: tìm nguyên nhân (seed, phiên bản, dữ liệu, bước huấn luyện) trước khi kết luận; ghi "Khác biệt" vào `docs/paper.md`.

## Giai đoạn D — Mở rộng ⬜
1. Mỗi ý tưởng = một exp mới + config + đoạn JOURNAL (giả thuyết → kết quả → kết luận).

## Giai đoạn E — Báo cáo ⬜
1. `make compare`, `make tex` → bảng vào báo cáo (skill `latex-report`); slide từ JOURNAL + REPRODUCTION.
