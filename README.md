# MMXAI — Truy vấn văn bản → ảnh xếp hạng lại bằng đồ thị, có giải thích (Graph ML, Project 3)

Cho một câu mô tả, hệ thống lấy top-100 ảnh theo CLIP rồi **xếp hạng lại bằng đồ thị** ảnh–nhãn–khái niệm (scene graph +
ConceptNet, GNN GAT/GCN/GraphSAGE), và **giải thích** mỗi kết quả bằng chính các cạnh/đường đi đã đóng góp vào điểm, có đo
fidelity / sparsity / validity. Đánh giá trên hai tập công khai: **Visual Genome ∩ MS-COCO** và **PSG ∩ MS-COCO**, so với
CLIP đóng băng, adapter, khớp tên và **AdaMF-MAT** (LREC-COLING 2024) trên cùng giao thức. Đề bài: `docs/assignment/`.

## Chạy

```bash
make setup                       # .venv + torch CPU + gói mmxai
make help                        # mọi lệnh
make fetch-prebuilt              # đường tắt cho thành viên nhóm (Kaggle CLI, dataset riêng tư); người ngoài nhóm: xem mục Dữ liệu
make tables && make xai DS=psg   # bảng kết quả + giải thích trên mô hình thật
make ui                          # MMXAI Lab — http://127.0.0.1:8765
make web                         # MMXAI Observatory (trang trình bày kết quả)
```

Tái lập từ đầu (dữ liệu thô → bundle → huấn luyện 3 seed → AdaMF-MAT → bảng → giải thích; bước GPU chạy trên Kaggle qua
kaggle CLI): `make all-psg`, `make all-vg` — chi tiết từng bước, thời gian, đầu vào/ra: **`docs/PIPELINE.md`**.

## Dữ liệu

Bundle là dữ liệu đã tiền xử lý, đủ để huấn luyện, đánh giá và chạy giao diện mà không cần dựng lại: tập chia cố định
(`split/split.json`, seed 0), ảnh kèm đối tượng và quan hệ scene graph (`data/images.jsonl`), caption kèm khái niệm và nhóm
direct/indirect (`data/captions.jsonl`), đặc trưng CLIP, 100 ảnh ứng viên mỗi truy vấn (`candidates/`), vectơ nhãn và cạnh
ConceptNet (`graph_extra*/`), mã đánh giá chung (`mmxai_eval.py`). Tải về rồi giải nén đúng thư mục ở cột cuối.

| Dữ liệu | Link | Đặt vào |
|---|---|---|
| Bundle VG ∩ COCO | https://www.kaggle.com/datasets/ff8c01430ea44c27aa26b38c2481952172b7fc45f06d985e6333562a1542160e | `data/vg_bundle/` |
| Bundle PSG ∩ COCO | https://kaggle.com/datasets/fad5fd07c637ab23d1777e4724d1f37bc452ee813f639d5541fa60f725a0da8e | `data/psg_bundle/` |
| Dữ liệu thô: caption MS-COCO theo cách chia Karpathy (`dataset_coco.json`), siêu dữ liệu ảnh VG có `coco_id` (`image_data.json`) | https://drive.google.com/file/d/1kurWIVhvokkGMILC--zMZRcIquAnqVn3/view?usp=sharing | `data/raw/` |

Chỉ cần dữ liệu thô khi dựng lại bundle từ đầu; `make fetch` cũng tải được từ nguồn gốc. Các nguồn còn lại của bước dựng
lại: scene graph Visual Genome (dataset Kaggle `yenphamnhu/visual-genome`, đọc trong notebook `notebooks/kaggle/vg_data/`),
PSG (Hugging Face `HarborYuan/OpenPSG` và `maelic/PSG-coco-format`), ConceptNet (Hugging Face `peandrew/conceptnet_en_simple`).
Mã tạo tập chia: `notebooks/kaggle/vg_data/vg_data_clip_baseline.ipynb` (VG) và `tools/psg_to_bundle.py` (PSG).
Checkpoint, notebook và kết quả của phần đánh giá XAI nằm sẵn trong `src/mmxai/xai/xai-experiment/` (xem `docs/PIPELINE.md`).

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
| `queries/`, `docs/`, `report/`, `slides/` | Bộ 24 truy vấn đánh giá (`queries/eval_queries.json`), tài liệu, báo cáo (`report/report.pdf`), slide |

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
