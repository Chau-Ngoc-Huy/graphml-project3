# mmxai-lab

Môi trường thực nghiệm cho paper **Project 3 - Truy vấn đa phương thức có giải thích bằng đồ thị (Graph ML)** (), dựng bằng skill `paper-lab`.
Code gốc của tác giả nằm nguyên ở `vendor/` (submodule, ghim ``); harness bao ngoài lo
cấu hình, run_id, seed, ghi phiên bản, chạy trên Kaggle/Colab từ terminal, và so số với paper.

## Cài đặt

```bash
git clone --recurse-submodules <repo này> && cd mmxai-lab
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp local.mk.example local.mk && cp .env.example .env        # điền tên, Kaggle user, key
make check && make test && make local EXP=smoke
```

## Chạy

| Lệnh | Việc |
|---|---|
| `make local EXP=<exp> ARGS="--limit 4"` | chạy trên máy (lượt thử) |
| `make push EXP=<exp> ARGS="--limit 4"` → `make wait` → `make pull` | Kaggle GPU; kết quả về `runs/<exp>/<run_id>/` |
| `make queue EXPS="a b c"` | nhiều thí nghiệm tuần tự trên Kaggle |
| `make compare` | `results/REPRODUCTION.md`: số paper vs số mình |
| `make summary` | `results/<exp>/SUMMARY.md`, `index.csv` |
| `make tex ARGS="--table table2 --out ../report/tables/tab_table2.tex"` | bảng LaTeX cho báo cáo |
| `make colab` | hướng dẫn chạy dự phòng trên Colab |

Chi tiết: `CLAUDE.md` (quy ước), `docs/paper.md` (paper đọc thế nào), `docs/repo-audit.md` (repo gốc chạy thế nào),
`results/JOURNAL.md` (nhật ký thực nghiệm).
