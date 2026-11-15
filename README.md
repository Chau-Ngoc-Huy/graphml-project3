# Project 3 — Truy vấn đa phương thức có giải thích bằng đồ thị (Graph ML)

Repo chung của nhóm. **Giai đoạn 1: mỗi người làm trong folder riêng**, push không ảnh hưởng người khác.
Giai đoạn 2 (tuần 3–4): gộp thành một pipeline chung (xem cuối file).

| Folder | Người | Nhiệm vụ (theo `shared/phan-cong.docx`) |
|---|---|---|
| `yen/` | Yến | Dữ liệu VG + FB15k-237-IMG, chuẩn hóa nhãn, **split** train/val/test, báo cáo, slide |
| `thang/` | Thắng | Nhiệm vụ 1: đặc trưng OpenCLIP, FAISS, metric, baseline CLIP 3 seed |
| `huy/` | Huy | Nhiệm vụ 2 cấp 1: dựng đồ thị, GNN (GCN/GAT/SAGE), xếp hạng lại, ablation |
| `khanh/` | Khánh | Nhiệm vụ 2 cấp 2: tái lập NativE/AdaMF-MAT, bộ ≥ 20 truy vấn trực tiếp/gián tiếp |
| `duong/` | Dương | Nhiệm vụ 3: giải thích (fidelity/validity/sparsity), MMKG tiếng Việt, demo |
| `shared/` | cả nhóm | Đề bài, phân công, **hợp đồng dữ liệu** (`shared/data-contract.md`) |

## Quy tắc (đọc 1 lần)

1. **Chỉ sửa file trong folder của mình.** Hai người không bao giờ sửa cùng file → không bao giờ conflict.
2. Muốn sửa `shared/` hoặc folder người khác: nhắn nhóm trước, hoặc tạo branch + Pull Request.
3. Trước khi push luôn kéo về trước:
   ```bash
   git pull --rebase
   git push
   ```
   Nếu push bị từ chối ("fetch first") chỉ là có người vừa push folder của họ → chạy lại 2 lệnh trên.
4. **Không commit dữ liệu, model, ảnh, `.npy`** (đã chặn trong `.gitignore`). Dữ liệu để trên Kaggle Dataset, trong code chỉ ghi
   tên dataset. File > 50 MB GitHub sẽ từ chối.
5. Không commit token/khóa: `kaggle.json`, `.env`, `local.mk`.
6. Commit nhỏ, message nói rõ làm gì (`huy: thêm ablation shuffle`). Lịch sử commit là minh chứng đóng góp khi vấn đáp.

## Bắt đầu (lần đầu)

```bash
git clone https://github.com/<owner>/graphml-project3.git
cd graphml-project3/<folder-của-bạn>
# làm việc, rồi:
git add .
git commit -m "<tên>: <làm gì>"
git pull --rebase && git push
```

## Dữ liệu dùng chung

Mọi người đọc/ghi dữ liệu theo **`shared/data-contract.md`** (tên file, định dạng, ai làm file nào) để khi gộp không phải
viết lại. Dataset Kaggle hiện có:

| Dataset | Có | Thiếu |
|---|---|---|
| `yenphamnhu/visual-genome` | `scene_graphs.json` (đối tượng + quan hệ) | ảnh, caption |
| `dannywu375/visualgenome` | ảnh VG (`images/VG_100K/`), `image_data.json` (có `coco_id`) | caption |
| `mathurinache/visual-genome` | `objects.json`, `relationships.json`, `object_alias.txt`, `relationship_alias.txt` | ảnh, caption |

**Cần chốt:** nguồn caption (đề xuất: caption COCO cho ~51k ảnh VG có `coco_id`) → rồi mới tạo split.

## Gộp lại (giai đoạn 2)

Khi các phần chạy được, chuyển dần vào một pipeline chung: `huy/` hiện đã có khung chạy thí nghiệm (config YAML, run_id,
Kaggle từ terminal, metric dùng chung trong `huy/src/mmxai/retrieval/core.py`). Phương án gộp chốt sau, ghi vào đây.
