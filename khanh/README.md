# khanh/ — Nhiệm vụ 2 cấp 2: phương pháp mới (AdaMF-MAT) + bộ truy vấn trực tiếp/gián tiếp

Phần của Khánh trong project 3 (Graph ML). Chỉ chủ folder sửa trong đây.

**Ý tưởng:** AdaMF-MAT (LREC-COLING 2024, multimodal KG completion) cắm vào pipeline chung của Thắng (`../thang/mmxai_eval.py`) như một
`score_fn`: CLIP top-100 → điểm đồ thị từ AdaMF-MAT → `α·z(cos) + (1−α)·z(graph)` (α chọn trên val). Cùng split, cùng fold
gallery 1K, cùng top-K với CLIP thuần và CLIP+Graph của Huy nên so sánh công bằng. Thiết kế đầy đủ: `docs/design.md`.

## Trạng thái (cập nhật `docs/plan.md`)

| Việc | Trạng thái |
|---|---|
| Tái lập AdaMF-MAT trên MKG-W, MKG-Y (khớp paper trong ±1 điểm) | ✅ `results/adamf_mat_repro.md` |
| Bộ 24 truy vấn trực tiếp/gián tiếp + mô tả | ✅ `queries/` |
| Bộ chuyển sang giao diện của Thắng (`score_fn`) + lượt chạy thật trên Kaggle | ✅ `results/protocol_coco_full.md` |
| Bảng trực tiếp vs gián tiếp | ✅ AdaMF-MAT > CLIP một chút (+0.64 MRR) nhưng chỉ ở nhóm direct; **không giúp nhóm indirect**, không hơn baseline khớp tên |
| So sánh công bằng với Huy, báo cáo | ⬜ giao Huy số liệu + mã; viết báo cáo |

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # máy không GPU
.venv/bin/pip install -e '.[dev]'
make test
```

AdaMF-MAT **không** nằm trong repo (không chắc giấy phép): notebook tự `git clone https://github.com/zjukg/AdaMF-MAT` trên Kaggle.

## Chạy

```bash
make queries DATA=../../data/scene_graphs.json   # sinh lại bộ truy vấn (cần scene_graphs.json, không commit)
make notebooks                                    # sinh kaggle/*.ipynb từ src/ (notebook nhúng nguyên mã nguồn)
```

Trên Kaggle (GPU T4 x2, Internet On): import `kaggle/protocol_adamf_kaggle.ipynb`, thêm input dataset `task1-bundle`
(đầu ra của Thắng: `data/`, `split/`, `features/`), **Save Version → Save & Run All** (~3 giờ). Nếu không có bundle, notebook tự
chạy lại notebook của Thắng (cần `yenphamnhu/visual-genome` + Internet). Lấy về `report_protocol.txt` + `results_COCO_full.json`.

Tái lập paper: `kaggle/adamf_mat_kaggle.ipynb` (MKG-W, MKG-Y; cần dataset embeddings của tác giả, xem notebook).

## Đầu ra

| File | Nội dung |
|---|---|
| `results/adamf_mat_repro.md` | AdaMF-MAT vs paper (MRR/Hits@k) trên MKG-W, MKG-Y + link phiên bản Kaggle |
| `results/level2_vg_own_experiment.{md,json}` | thí nghiệm riêng trên VG (24 truy vấn, kiểm tra ẩn đối tượng): **baseline không dùng quan hệ thắng AdaMF-MAT** |
| `results/JOURNAL.md` | nhật ký các lượt chạy |
| `queries/eval_queries.{json,csv}` | 24 truy vấn (12 trực tiếp, 12 gián tiếp) + ảnh liên quan; `queries/README.md` mô tả cách xây dựng và hạn chế |

## Cấu trúc

```
src/                  common.py (split, metric) · features.py (OpenCLIP đóng băng) · build_eval_queries.py (bộ truy vấn)
                      export_vg_mmkg.py (đồ thị VG riêng → định dạng AdaMF-MAT) · adamf_protocol.py (build MMKG từ Protocol, AdamfScorer, evaluate_all)
tests/                build MMKG từ dữ liệu giả đúng định dạng của Thắng, split, metric
tools/                make_notebooks.py (sinh notebook Kaggle từ src/)
kaggle/               protocol_adamf_kaggle.ipynb (chính) · adamf_mat_kaggle.ipynb (tái lập paper) · exploratory/ (thí nghiệm riêng trên VG)
queries/              bộ truy vấn đánh giá + mô tả
docs/                 plan.md · design.md · level2-notes.md
results/              JOURNAL.md, bảng kết quả (.md/.json nhỏ)
```

Đề bài, phân công, hợp đồng dữ liệu: `../shared/`. Giao thức đánh giá chung: `../thang/mmxai_eval.py`.
