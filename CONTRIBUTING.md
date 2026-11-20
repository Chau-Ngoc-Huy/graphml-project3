# Quy ước làm việc

- Không commit `data/`, `runs/`, `models/`, `cache/`, `build/`, `logs/`, `local.mk`, `.env`, token Kaggle. Commit code, cấu hình,
  `results/**/*.json` (kết quả nhỏ, có hạng từng truy vấn) và `results/JOURNAL.md` cùng commit với code tạo ra chúng.
- Thêm thí nghiệm: `configs/experiments/<exp>.yaml` (kế thừa bằng `extends:`) + `src/mmxai/experiments/<exp>.py`;
  chạy `make test` rồi `make local EXP=<exp> ARGS="--limit 4"` trước lượt đầy đủ (3 seed).
- Mọi phương pháp chấm qua `src/mmxai_eval.py` (một bản duy nhất) để so sánh công bằng; đổi giao thức = đổi mọi kết quả.
- Mã nhúng trong notebook Kaggle (`notebooks/kaggle/adamf/`) sinh lại bằng `make notebooks` sau khi sửa `src/mmxai/adamf/`.
- Kernel Kaggle chạy dưới tài khoản trong `local.mk` (`KAGGLE_USER`); dataset/model riêng tư — chủ sở hữu thêm cộng tác viên.
- Commit nhỏ, message nói rõ làm gì; mỗi lượt chạy đầy đủ một đoạn trong `results/JOURNAL.md`.
- Nhánh `final` là cấu trúc nộp; thay đổi trên `main` (cấu trúc cũ) đưa sang bằng `make sync` (`docs/migration-map.tsv`).
