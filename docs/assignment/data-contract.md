# Hợp đồng dữ liệu (data contract)

Pipeline của Nhiệm vụ 2 (Huy) đọc **một thư mục** có cấu trúc dưới đây. Dữ liệu giả (`mmxai.data.fake`) và dữ liệu thật
dùng **cùng** cấu trúc này, nên khi có dữ liệu thật chỉ cần đổi đường dẫn: `--set data=<thư mục>` (máy local) hoặc
`--set data=<tên dataset Kaggle>` (kernel tự tìm trong `/kaggle/input`).

| File | Ai làm | Định dạng | Bắt buộc |
|---|---|---|---|
| `vg/objects.json` | Yến | Visual Genome gốc: `[{"image_id", "objects": [{"object_id", "names": [...]}]}]` (có thể là tập con) | ✔ |
| `vg/relationships.json` | Yến | Visual Genome gốc: `[{"image_id", "relationships": [{"predicate", "subject": {"names"}, "object": {"names"}}]}]` | nên có |
| `synonyms.json` | Yến | `{"men": "man", "t-shirt": "shirt", ...}` chuẩn hóa nhãn | tùy chọn |
| `captions.json` | Yến/Thắng | `[{"caption_id", "image_id", "text", "group"?}]` — `group` = `direct`/`indirect` nếu có | ✔ |
| `splits.json` | Yến | `{"seed", "train": [image_id], "val": [...], "test": [...]}` — chia theo ẢNH | ✔ |
| `conceptnet.tsv` | Huy | `head<TAB>rel<TAB>tail`, đã lọc theo vocab (`tools/fetch_conceptnet.py`) | nên có |
| `features/image_emb.npy` + `image_ids.json` | Thắng | float32 `[N_img, D]`, đã L2-normalize; `image_ids[i]` ↔ dòng i | ✔ |
| `features/caption_emb.npy` + `caption_ids.json` | Thắng | float32 `[N_cap, D]`, L2-normalize | ✔ |
| `features/label_emb.npy` + `labels.json` | Huy/Thắng | CLIP text của `"a photo of a {nhãn}"` cho mọi nhãn + khái niệm (`tools/extract_clip.py --labels`) | ✔ |
| `features/clip_model.json` | Thắng | `{"model", "pretrained", "prompt"}` — **cả ba file emb phải cùng một model CLIP** | nên có |

## Quy tắc

- Cùng một model CLIP (vd. `ViT-B-32 / laion2b_s34b_b79k`) cho ảnh, caption và nhãn — khác model thì cosine vô nghĩa.
- Nhãn trong `labels.json` phải trùng với tên nhãn sau chuẩn hóa (`lower()`, gộp khoảng trắng, áp `synonyms.json`).
  Nhãn VG không có trong `labels.json` bị bỏ qua; số lượng in ra ở `graph.missing_labels` trong metrics.
- Split theo ảnh, cố định bằng seed; mọi thí nghiệm (CLIP, CLIP+Graph, phương pháp cấp 2) dùng chung một `splits.json`.
- Caption của ảnh test **không** được dùng huấn luyện; cạnh quan hệ scene graph chỉ gộp từ ảnh train (code tự lọc).

## Đưa lên Kaggle

```bash
# thư mục data đúng cấu trúc trên → dataset private
kaggle datasets init -p data/vg_real        # sửa dataset-metadata.json: id = huychau/mmxai-vg
kaggle datasets create -p data/vg_real --dir-mode zip
# local.mk: DATASET = huychau/mmxai-vg
make push EXP=clip_graph ARGS="--set data=mmxai-vg --limit 5 --tag thu"
```

## Kiểm tra nhanh một thư mục dữ liệu

```bash
.venv/bin/python -c "from mmxai.data.contract import load; b=load('<thư mục>'); print(len(b.image_ids), len(b.captions), len(b.labels), {k: len(v) for k, v in b.splits.items()})"
make local EXP=clip_baseline ARGS="--set data=<thư mục>"
```
