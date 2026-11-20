# MMXAI — Nhiệm vụ 1: Truy vấn đa phương thức & đánh giá chuẩn

Project 3 *Truy vấn đa phương thức có giải thích bằng đồ thị* (môn Học máy với dữ liệu đồ thị):
trích đặc trưng OpenCLIP đóng băng + FAISS, cài metric (Recall@1/5/10, MRR, nDCG),
baseline CLIP thuần ≥3 seed (mean ± std), **module đánh giá dùng chung** cho mọi mô hình của nhóm,
mở rộng image→text và đánh giá theo nhóm truy vấn.

**Trang báo cáo kết quả:** https://minhthang03.github.io/mmxai-task1-clip-baseline/web/

## Nội dung

| File | Mô tả |
|---|---|
| `task1_clip_baseline_kaggle.ipynb` | Notebook chạy trên Kaggle (hoặc local), tạo toàn bộ đầu ra bên dưới |
| `mmxai_eval.py` | Module đánh giá dùng chung (metric, giao thức split/fold/seed, top-K, fuse α, kiểm định). Notebook nhúng đúng file này trong ô `%%writefile`; khi sửa module, sửa cả ô đó để hai bản giống nhau |
| `requirements.txt` | Thư viện cần thêm ngoài môi trường Kaggle |
| `kernel-metadata.json` | Cấu hình để đẩy notebook lên Kaggle bằng `kaggle kernels push` (sửa `KAGGLE_USERNAME`) |
| `kaggle_results/` | Kết quả bản full trên Kaggle: split cố định, bảng baseline, biểu đồ, phân tích lỗi (file hạng `ranks_clip_*.npz` không commit theo quy ước repo nhóm, lấy từ Output của notebook) |
| `web/` | Trang web tĩnh tổng hợp kết quả (mở `web/index.html`) |

## Chạy trên Kaggle

1. Tạo notebook mới → *File → Import Notebook* → chọn `task1_clip_baseline_kaggle.ipynb`.
2. *Add Input* → dataset `yenphamnhu/visual-genome` (chứa `scene_graphs.json`). Notebook tự tìm file trong `/kaggle/input`.
3. Settings: **Accelerator GPU T4 x1**, **Internet On** (tải `image_data.json` của VG, `dataset_coco.json` Karpathy và ảnh VG qua URL).
4. *Run All*. Thời gian ước tính trên T4: tải + mã hoá ~19K ảnh qua URL 10–20 phút, phần còn lại < 5 phút.
   Mọi bước có cache trong `/kaggle/working/task1`, chạy lại sẽ bỏ qua bước đã xong.
5. Kết quả gom trong `/kaggle/working/task1_bundle.zip` (tab *Output*).

Chạy local (CPU được, chậm hơn): `pip install -r requirements.txt` rồi mở notebook; đặt `MMXAI_SMOKE=1` để chạy thử nhanh với vài trăm ảnh.

## Dữ liệu và giao thức

- **Tập dữ liệu**: Visual Genome ∩ MS-COCO (ảnh VG có `coco_id`, 51 208 ảnh). Truy vấn = 5 caption COCO/ảnh (Karpathy);
  phía đồ thị = scene graph VG (objects + relationships, đã chuẩn hoá tên).
- **Split** (`split/split.json`): Karpathy `train+restval → train` (lấy mẫu 15 000 ảnh, seed 0), `val → val` (2 126), `test → test` (2 138).
  Nếu Yến đã chốt split khác: điền `CFG["split_file"]`.
- **Seed**: CLIP đóng băng là tất định; seed `0,1,2` định nghĩa 3 fold gallery 1 000 ảnh lấy từ pool test/val (giao thức test-1K của COCO).
  Báo cáo mean ± std trên 3 fold **và** số tất định trên toàn bộ pool. Mô hình học (GNN, phương pháp mới) dùng cùng các fold này + seed khởi tạo.
- **Top-K**: K = 100 ứng viên CLIP cho mỗi truy vấn; mọi phương pháp xếp lại trong cùng top-K (ảnh đúng ngoài top-K giữ hạng CLIP).
- **Chống rò rỉ**: từ vựng đối tượng / nhóm truy vấn chỉ tính từ train; caption val/test không tạo thống kê nào; α và siêu tham số chỉ chọn trên val
  (`tune_alpha`); bộ truy vấn tự xây của Khánh chỉ dùng để đánh giá.

## Cách cắm mô hình khác vào module

```python
from mmxai_eval import Protocol, fuse, tune_alpha, summary_table, paired_bootstrap
P = Protocol.load("/kaggle/working/task1")          # split + CLIP features + nhóm truy vấn

clip = lambda c, p: c["sim"]                          # baseline
base = P.evaluate(clip, "CLIP", split="test", seeds=[0, 1, 2])

def graph_score(c, p):                                # (Nq, K) điểm đồ thị cho từng ứng viên trong top-K
    ...                                               # c["q_cap"]: idx caption; c["cand"]: idx ảnh ứng viên; p.images[i]["objects"], p.captions[q]["concepts"]

c_val = P.candidates("val", None)
alpha, curve = tune_alpha(c_val, graph_score(c_val, P))            # α chỉ chọn trên val
ours = P.evaluate(lambda c, p: fuse(c["sim"], graph_score(c, p), alpha), "CLIP+GAT", "test", [0, 1, 2])
print(summary_table([base, ours]))                                 # mean ± std, bảng Markdown
print(ours["by_group"])                                            # direct / indirect / no_concept
print(paired_bootstrap(base["full"]["ranks"], ours["full"]["ranks"]))  # kiểm định ghép cặp trên toàn pool test
```

Hard negative cho huấn luyện GNN: `candidates/train_galleries.npz` (train chia thành gallery 1 000 ảnh, ảnh đúng luôn nằm trong top-K).
Bộ ≥20 truy vấn nhiều ảnh liên quan: dùng `multi_relevant_metrics(order, relevant_set)`.

## Xuất cho pipeline của nhóm (`shared/data-contract.md`)

Mục §9 của notebook ghi thư mục `/kaggle/working/mmxai_vg_contract/` mà `mmxai.data.contract.load()` (code của Huy) đọc được ngay:

```
features/image_emb.npy + image_ids.json       CLIP ảnh float32 [N_img, 512], chuẩn hoá L2, id = id ảnh Visual Genome
features/caption_emb.npy + caption_ids.json   CLIP caption float32 [N_cap, 512], caption_id = "<vg_id>_<k>"
features/label_emb.npy + labels.json          CLIP text "a photo of a {nhãn}" cho mọi tên đối tượng trong vg/objects.json
features/clip_model.json                      {model, pretrained, prompt}
captions.json                                 [{caption_id, image_id, text, group}]
splits.json, vg/objects.json, vg/relationships.json   bản tạm cho đúng tập ảnh này (Yến chốt bản chính thức)
```

Ô ngay sau đó tự kiểm tra thư mục theo đúng điều kiện của `load()` (dtype, chiều, chuẩn hoá, id khớp, split không giao nhau)
và dừng notebook nếu có lỗi. Notebook khác dùng thư mục này qua *Add Input → Notebook output*, hoặc tạo dataset bằng
`kaggle datasets create -p mmxai_vg_contract`.

## Đầu ra của notebook (`WORK/`)

```
split/split.json                     split cố định + meta
data/images.jsonl                    ảnh: id, vg_id, coco_id, url, split, objects, relations
data/captions.jsonl                  caption: img, text, group (direct/indirect/no_concept), concepts
data/object_vocab_train.json         từ vựng đối tượng (train)
features/clip_ViT-B-32_laion2b.npz   img (Ni,512) / cap (Nc,512) float16 đã chuẩn hoá, cap_img, ok
features/faiss_test.index            IndexFlatIP trên gallery test
candidates/{test,val}_{full,seed0..2}.npz, candidates/train_galleries.npz
results/clip_baseline.{json,md}      bảng baseline (fold + full, t2i + i2t, theo nhóm)
results/ranks_clip_*.npz             hạng từng truy vấn (cho kiểm định ghép cặp)
results/figures/*.png                biểu đồ cho báo cáo
results/clip_success_failure.{csv,html}  5 thành công / 5 thất bại
results/example_rerank_lexical.json  ví dụ cắm phương pháp + α trên val + bootstrap
```

## Kết quả bản full trên Kaggle (T4, commit 2026-11-01, 6,3 phút)

Notebook: https://www.kaggle.com/code/minhthang0369/notebook0c3ea0357c · Đầu ra đã tải về `kaggle_results/` (bảng, biểu đồ, split, phân tích lỗi).
Bản zip đầy đủ (đặc trưng CLIP, FAISS index, candidates, 155 MB) không đưa lên git vì vượt giới hạn 100 MB của GitHub:
lấy ở tab *Output* của notebook Kaggle (`task1_bundle.zip`) hoặc chạy lại notebook.

Split: train 15 000 / val 2 126 / test 2 138 ảnh; 96 381 caption; 19 264/19 264 ảnh mã hoá thành công.

| CLIP ViT-B/32 (laion2b), text → image | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| test, gallery 1K, mean ± std 3 fold | 57.36 ± 0.95 | 84.08 ± 0.24 | 91.76 ± 0.33 | 69.23 ± 0.68 | 74.32 ± 0.55 |
| test, toàn pool 2 138 ảnh | 48.31 | 75.35 | 84.08 | 60.40 | 65.51 |
| val, gallery 1K, mean ± std 3 fold | 58.17 ± 0.40 | 84.74 ± 0.68 | 92.29 ± 0.41 | 69.92 ± 0.40 | 75.00 ± 0.42 |

| image → text | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| test, gallery 1K, 3 fold | 74.10 ± 0.96 | 91.90 ± 0.54 | 96.53 ± 0.21 | 81.82 ± 0.84 |

Theo nhóm truy vấn (test, 1K): direct 56.9 / indirect 57.5 / no_concept 51.4 R@1 — CLIP không yếu hơn rõ rệt ở nhóm gián tiếp theo cách gán nhãn tự động này;
bộ truy vấn gián tiếp tự xây của Khánh mới là nơi kiểm tra vai trò đồ thị. Ví dụ rerank chỉ khớp tên đối tượng (α = 0.8 chọn trên val)
đã tăng MRR +1.06 (CI95 [0.74, 1.37], p < 0.001) trên toàn pool test → tín hiệu scene graph có ích, GNN của Huy nên vượt mức này.

## Trang web tổng hợp kết quả (`web/`)

Trang tĩnh HTML/CSS/JS thuần, không cần server hay thư viện ngoài (chỉ ảnh ví dụ tải từ máy chủ Visual Genome).

```bash
cd web
python build_site.py ../kaggle_results     # sinh data.js từ results/ + split/ (sau mỗi lần chạy lại notebook)
# cần results/ranks_clip_t2i_test.npz: giải nén từ task1_bundle.zip (Output Kaggle) vào kaggle_results/ trước khi chạy
xdg-open index.html                        # hoặc double-click index.html
```

Nội dung: 5 ô số chính, thông số split/giao thức, biểu đồ text→image và image→text theo fold, phân bố hạng, nhóm truy vấn,
bảng đầy đủ có bộ lọc chiều/split, độ nhạy theo α và so sánh CLIP vs CLIP + khớp tên, 10 ví dụ thành công/thất bại kèm ảnh, kết luận.
Tự chuyển sáng/tối theo hệ điều hành, hiển thị được trên điện thoại.

## Nguồn dữ liệu và mô hình

- **Visual Genome 1.4** (Krishna et al., IJCV 2017), giấy phép CC BY 4.0: `scene_graphs.json` (qua Kaggle `yenphamnhu/visual-genome`),
  `image_data.json` và ảnh tải từ máy chủ chính thức của Visual Genome.
- **MS-COCO captions** (Lin et al., ECCV 2014; Chen et al., 2015), chú thích theo giấy phép CC BY 4.0, với **Karpathy split** (Karpathy & Fei-Fei, CVPR 2015).
- **OpenCLIP** `ViT-B-32 / laion2b_s34b_b79k` (Cherti et al., CVPR 2023), dùng đóng băng, không fine-tune.
- **FAISS** (Johnson et al., 2019) cho tìm láng giềng gần.

Repo không chứa ảnh hay file dữ liệu gốc; `kaggle_results/` chỉ gồm số liệu tổng hợp, id ảnh, URL ảnh và một số caption COCO làm ví dụ.

## MMXAI Observatory (`observatory/`)

Trang web React trình bày kết quả của cả nhóm: pipeline, playground chạy truy vấn trên 14 cấu hình mô hình, so sánh mô hình, đồ thị truy vấn và kết quả đánh giá — mọi số liệu lấy từ file kết quả thật và được kiểm tra chéo. Cách chạy: `observatory/README.md`.
