# Quy ước làm việc

- `run_id` chứa `LAB_USER` nên file kết quả của mỗi người không bao giờ trùng; commit `results/**/*.json` cùng PR với code tạo ra nó.
- Không commit `data/`, `runs/`, `cache/`, `build/`, `local.mk`, `.env`, token. Đầu ra thô chia sẻ qua Drive.
- Mỗi người dùng tài khoản Kaggle của mình (kernel `<KAGGLE_USER>/mmxai-lab-<exp>`); dataset private, chủ dataset thêm cộng tác viên.
- Thêm thí nghiệm: tạo `experiments/<exp>.py` + `configs/experiments/<exp>.yaml`, `make test` xanh, `--limit 4` trước lượt đầy đủ.
- Đụng `core/`, `tools/`, `Makefile`, `configs/vendor.yaml`: báo cả nhóm (ảnh hưởng mọi kết quả). Không sửa `vendor/`.
- Nhánh `exp/<tên>/<việc>`; PR vào `main`; mỗi lượt đầy đủ một đoạn `results/JOURNAL.md`.
