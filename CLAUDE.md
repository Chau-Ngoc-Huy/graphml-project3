# CLAUDE.md — mmxai-lab

Thực nghiệm paper **Project 3 - Truy vấn đa phương thức có giải thích bằng đồ thị (Graph ML)** (). Code gốc: `vendor/` ( @ ``,
KHÔNG sửa trong đó — mọi thay đổi nằm ở adapter `src/mmxai/experiments/` hoặc config). Dựng bằng skill `paper-lab`.

Đọc trước khi làm: `docs/paper.md` (claim, bảng cần tái lập, setup), `docs/repo-audit.md` (entrypoint, lệnh gốc ↔ bảng, bẫy),
`docs/plan.md` (giai đoạn, trạng thái). Nhật ký: `results/JOURNAL.md` (lượt mới nhất ở trên). So với paper: `results/REPRODUCTION.md`.

## Vòng lặp

```
make local EXP=<exp> ARGS="--limit 4"        # máy local, vài phút, trước mọi thứ
make push  EXP=<exp> ARGS="--limit 4"        # Kaggle thử; make wait; make pull; tools/show_run.py
make push  EXP=<exp>                          # lượt đầy đủ (chỉ sau khi lượt thử sạch)
make compare && make summary                 # REPRODUCTION.md, SUMMARY.md → viết đoạn JOURNAL, commit results/
```

Thí nghiệm = `src/mmxai/experiments/<exp>.py` (hàm `run(ctx)`, mẫu `_template.py`) + `configs/experiments/<exp>.yaml`.
`run_id = <UTC>_<LAB_USER>_<exp>[-tag]`; `runs/<exp>/<run_id>/` (thô, không commit) → `results/<exp>/<run_id>.json` (metric, commit).
`--set a.b=v` ghi đè config; `--tag` nói rõ đang thử gì; `--limit N` = lượt thử (SUMMARY tách riêng).

## Luật

- Luôn `--limit` nhỏ trước lượt đầy đủ (GPU Kaggle 30 h/tuần, kernel tối đa 12 h, mỗi EXP một kernel một lúc).
- Không commit `data/`, `runs/`, `cache/`, `build/`, `local.mk`, `.env`, token. Chỉ commit `results/**/*.json`, `JOURNAL.md`, `REPRODUCTION.md`, code, docs.
- Số trong paper nằm ở `docs/paper_numbers.yaml` (chép từ bảng gốc, ghi rõ bảng/dòng/cột); không gõ số vào báo cáo bằng tay, dùng `make tex`.
- Seed cố định; metrics.json tự ghi commit harness + commit vendor + phiên bản thư viện + GPU. Đổi commit vendor bằng `make vendor-update REF=...`.
- Mỗi lượt đầy đủ hoặc mỗi kết luận → một đoạn trong `results/JOURNAL.md` (mẫu trong file). Lỗi kernel: `runs/<exp>/_failed/*/kaggle.log`, sửa, `make test`, push lại.
- Khác biệt với paper (dataset, model nhỏ hơn, ít bước, phần cứng) ghi vào `docs/paper.md` mục "Khác biệt" và cột ghi chú của REPRODUCTION.

## Cấu trúc

```
src/mmxai/run.py                 python -m mmxai.run --exp <exp> [--limit N] [--tag t] [--set k=v]
src/mmxai/core/                  config (yaml + extends + --set), env (seed, phiên bản, secret), vendor (chạy code gốc, bắt số), runner, results
src/mmxai/experiments/           smoke (không GPU), _template, <exp>.py — adapter cho từng ô/bảng của paper
configs/experiments/<exp>.yaml     tham số; configs/vendor.yaml: repo gốc (url, commit, install); configs/models.yaml: model + revision
vendor//            submodule repo gốc (make vendor)
tools/                             make_kernel, pull_kernel (Kaggle), compare (paper vs mình), export_tex, show_run, note, run_queue, import_runs, vendor_pin
docs/                              paper.md, repo-audit.md, plan.md, paper_numbers.yaml, paper_src/ (PDF/tex, không commit nếu có bản quyền)
results/                           <exp>/<run_id>.json, <exp>/SUMMARY.md (make summary), JOURNAL.md, REPRODUCTION.md (make compare), tex/
colab/run.ipynb                    dự phòng khi hết GPU Kaggle: chạy cùng CLI, tải runs.zip về rồi `make import`
```
