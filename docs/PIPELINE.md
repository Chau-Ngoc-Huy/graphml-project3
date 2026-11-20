# Pipeline đầy đủ — tái lập mọi kết quả của báo cáo bằng dòng lệnh

Mọi bước chạy bằng `make` từ gốc repo. Bước cần GPU chạy trên **Kaggle qua kaggle CLI** (đẩy kernel → chờ → tải kết quả về),
không thao tác tay trên web. `DS=vg` (Visual Genome ∩ COCO) hoặc `DS=psg` (Panoptic Scene Graph ∩ COCO). `make help` liệt kê lệnh.

## 0. Chuẩn bị (một lần)

```bash
make setup                      # .venv + torch CPU + gói mmxai (phần phụ dev, clip, ui, viz)
# Kaggle: token ở ~/.kaggle/access_token (hoặc kaggle.json); ghi `KAGGLE_USER = <tên>` vào local.mk (xem local.mk.example)
```

## Các bước

| # | Lệnh | Chạy ở | Thời gian | Vào → Ra |
|---|---|---|---|---|
| 0 | `make fetch` | máy | vài phút | Internet → `data/raw/dataset_coco.json` (Karpathy), `data/conceptnet/hf_en_simple/train.parquet`, `data/psg_raw/` (psg.json + 8 parquet) |
| 1 | `make data DS=vg` | **Kaggle T4** | ~1 giờ | dataset `yenphamnhu/visual-genome` + Karpathy → notebook Nhiệm vụ 1 (Thắng): split, CLIP ViT-B/32, FAISS, baseline CLIP 3 seed → `data/vg_bundle/` |
| 1 | `make data DS=psg` | máy (CPU) | ~75 phút | `data/psg_raw` + Karpathy + ConceptNet → `data/psg_bundle/` (18 186 ảnh, 133 nhãn + 674 khái niệm) + đặc trưng CLIP |
| 2 | `make extras DS=..` | máy | vài phút | bundle → `graph_extra/` (vector nhãn), `graph_extra_cn/` (cạnh ConceptNet + vector) |
| 2b | `make publish DS=..` | máy → Kaggle | vài phút | bundle → Kaggle Dataset riêng tư `<user>/mmxai-task1-bundle` / `mmxai-psg-bundle` |
| 3 | `make train DS=vg` | **Kaggle T4** ×2 | ~3 giờ | 12 cấu hình × 3 seed (bảng `vg_official`) → `runs/`, `results/vg_*/` |
| 3 | `make train DS=psg` | máy (CPU, nhẹ) | ~2 giờ | 8 cấu hình × 3 seed → `runs/psg_*`, `results/psg_*/` |
| 4 | `make adamf DS=..` | **Kaggle T4** ×2 | ~5 giờ | AdaMF-MAT (Khánh) + ablation xáo/bỏ quan hệ → `results/adamf_<ds>/` |
| 5 | `make tables` | máy | < 1 phút | `results/tables/vg_official.md`, `psg_official.md`, `results/psg_vs_adamf.md`, `results/adamf/vs_qagraph_paired.md` |
| 6 | `make xai DS=..` | máy | ~1–2 phút / truy vấn | mô hình seed 0 + khung xAI (Dương) → `results/xai_bridge/<model>/` (fidelity, sparsity, validity) |
| 7 | `make web` / `make ui` | máy | — | Observatory (`apps/observatory`), Lab (`apps/lab`, cổng 8765; PSG: `MMXAI_BUNDLE=data/psg_bundle MMXAI_RUNS='psg_qa*'`) |
| 8 | `make report` | máy | — | `report/report.pdf` |

Chạy cả chuỗi một bộ dữ liệu: `make all-psg` hoặc `make all-vg`.
**Lượt thử** (rẻ, kiểm tra đường chạy): `make data DS=vg SMOKE=1` (vài trăm ảnh, ~5 phút), `make adamf DS=psg SMOKE=1` (2 epoch),
`make local EXP=psg_qa ARGS="--limit 3"`.

**Đường tắt** (không chạy lại bước 1–4, dùng bản dựng sẵn riêng tư của nhóm):
`make setup fetch-prebuilt` → bundle `data/{task1,psg}_bundle` + 4 checkpoint `models/*_seed0`, rồi `make tables xai web`.

## Kiểm chứng tái lập (đã làm trên nhánh final)

- Caption Karpathy tải bằng `make fetch` giống hệt từng byte bản đã dùng dựng bundle.
- `make extras`: dựng lại `conceptnet.tsv` (VG 16 413 cạnh, PSG 3 743), `concepts_added.json` giống hệt; `label_emb.npy` lệch 0.0.
- Observatory: `npm run data` từ `results/` cho số liệu giống hệt bản cũ (15/15 kiểm tra chéo MRR tính từ hạng = số báo cáo).
- Cầu nối xAI: điểm cuối khớp giao thức (lệch ≤ 2e-7).

## Chống rò rỉ (đề, mục 6)

Split cố định (seed 0); cạnh thống kê (sg_rel) chỉ từ ảnh train; α và early stopping chọn trên val; test chỉ để báo cáo;
mọi phương pháp dùng chung `src/mmxai_eval.py` (một bản), cùng top-K CLIP và cùng gallery theo seed.

## Nhánh và đồng bộ nhóm

Nhánh `final` là cấu trúc nộp. Thay đổi của thành viên trên `main` (cấu trúc cũ) đưa sang bằng `make sync`
(`docs/migration-map.tsv` ánh xạ đường dẫn cũ → mới).
