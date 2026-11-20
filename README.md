# MMXAI — Truy vấn văn bản → ảnh xếp hạng lại bằng đồ thị, có giải thích (Graph ML, Project 3)

Cho một câu mô tả, hệ thống lấy top-100 ảnh theo CLIP rồi **xếp hạng lại bằng đồ thị** ảnh–nhãn–khái niệm (scene graph +
ConceptNet, GNN GAT/GCN/GraphSAGE), và **giải thích** mỗi kết quả bằng chính các cạnh/đường đi đã đóng góp vào điểm, có đo
fidelity / sparsity / validity. Đánh giá trên hai tập công khai: **Visual Genome ∩ MS-COCO** và **PSG ∩ MS-COCO**, so với
CLIP đóng băng, adapter, khớp tên và **AdaMF-MAT** (LREC-COLING 2024) trên cùng giao thức. Đề bài: `docs/assignment/`.

## Chạy

```bash
make setup                       # .venv + torch CPU + gói mmxai
make help                        # mọi lệnh
make fetch-prebuilt              # đường tắt: bundle VG/PSG + 4 checkpoint dựng sẵn (Kaggle, riêng tư)
make tables && make xai DS=psg   # bảng kết quả + giải thích trên mô hình thật
make ui                          # MMXAI Lab — http://127.0.0.1:8765
make web                         # MMXAI Observatory (trang trình bày kết quả)
```

Tái lập từ đầu (dữ liệu thô → bundle → huấn luyện 3 seed → AdaMF-MAT → bảng → giải thích; bước GPU chạy trên Kaggle qua
kaggle CLI): `make all-psg`, `make all-vg` — chi tiết từng bước, thời gian, đầu vào/ra: **`docs/PIPELINE.md`**.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/mmxai_eval.py` | Giao thức đánh giá chung: split, gallery 1K theo seed, top-K CLIP, R@k / MRR / nDCG, nhóm truy vấn trực tiếp/gián tiếp, trộn điểm α, bootstrap ghép cặp |
| `src/mmxai/data/` | Đọc bundle (ảnh, nhãn, quan hệ, caption, đặc trưng CLIP, phần bổ sung đồ thị) |
| `src/mmxai/graph/` | Dựng đồ thị (contains, sg_rel từ ảnh train, ConceptNet), GNN, **QA-Graph** (điểm phụ thuộc truy vấn), giải thích theo đường đi |
| `src/mmxai/experiments/` + `configs/experiments/` | Mỗi cấu hình thí nghiệm = 1 YAML (+ module 2 dòng); `vg_*` cho VG, `psg_*` cho PSG, `clip_graph`/`smoke` cho dữ liệu giả |
| `src/mmxai/adamf/` | AdaMF-MAT trên giao thức chung (dựng MMKG, chấm điểm), adapter, bộ ≥ 20 truy vấn |
| `src/mmxai/xai/`, `src/mmxai/xai_bridge.py` | Khung giải thích (bỏ từng cạnh, fidelity/sparsity/validity) + cầu nối tới mô hình thật |
| `src/mmxai/core/` | Chạy thí nghiệm: cấu hình, run_id, ghi kết quả, đóng gói kernel Kaggle |
| `tools/` | Dữ liệu (fetch, PSG, ConceptNet, vector nhãn), Kaggle (kaggle_job, make_kernel, publish), bảng, giải thích |
| `notebooks/kaggle/` | Notebook GPU: dựng dữ liệu VG + baseline CLIP; AdaMF-MAT (chạy bằng `tools/kaggle_job.py`) |
| `apps/lab/` | MMXAI Lab (FastAPI): thử từng biến thể, so sánh A/B, xem đồ thị + đường đi giải thích |
| `apps/observatory/` | MMXAI Observatory (React): pipeline, playground, so sánh mô hình, kết quả |
| `results/` | Kết quả đã chạy (JSON từng lượt, hạng từng truy vấn, bảng) + `JOURNAL.md` (nhật ký thí nghiệm) |
| `queries/`, `docs/`, `report/`, `slides/` | Bộ truy vấn đánh giá, tài liệu, báo cáo IEEE, slide |

`data/`, `runs/`, `models/` không commit (tải bằng `make fetch` / `make fetch-prebuilt` hoặc sinh lại bằng pipeline).

## Kết quả chính (MRR %, toàn bộ ảnh test, 3 seed)

| | VG ∩ COCO (2 138 ảnh) | PSG ∩ COCO (2 186 ảnh) |
|---|---|---|
| CLIP (đóng băng) | 60.40 | 59.59 |
| CLIP + Adapter | 63.14 ± 0.02 | 62.67 ± 0.03 |
| + QA-Graph (contains + sg_rel) | 64.76 ± 0.12 | 64.17 ± 0.09 |
| QA-Graph, xáo cạnh (ablation) | 62.67 ± 0.06 | 62.12 ± 0.08 |
| CLIP + Adapter + AdaMF-MAT | 63.00 (1 seed) | 62.22 (1 seed) |

Bảng đầy đủ: `results/tables/` (`make tables`), kiểm định ghép cặp: `results/adamf/vs_qagraph_paired.md`,
`results/psg_vs_adamf.md`; diễn giải: `results/JOURNAL.md`.

## Chống rò rỉ dữ liệu

Split cố định theo ảnh (seed 0); cạnh thống kê chỉ từ ảnh train; α và early stopping chọn trên val; test chỉ để báo cáo;
mọi phương pháp dùng chung `src/mmxai_eval.py`, cùng top-K CLIP và cùng gallery theo seed (có test trong `tests/`).
