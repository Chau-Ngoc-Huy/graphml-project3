# Đầu vào đồ thị của QA-Graph (Huy) — bản chép từ `data/<bundle>/huy_extra*` để cả nhóm xem được

| Thư mục | Bundle | Nội dung |
|---|---|---|
| `vg/huy_extra/` | `data/task1_bundle` (VG ∩ COCO) | `labels.json` — 2 000 nút nhãn |
| `vg/huy_extra_cn/` | như trên | `labels.json` (2 000 nhãn + 1 000 khái niệm), `concepts_added.json`, `conceptnet.tsv` (cạnh ConceptNet đã lọc) |
| `psg/huy_extra/` | `data/psg_bundle` (PSG ∩ COCO) | `labels.json` — 807 nút (133 nhãn + 674 khái niệm mở rộng) |
| `psg/huy_extra_cn/` | như trên | `labels.json`, `concepts_added.json` (rỗng: khái niệm đã có trong vocab), `conceptnet.tsv` (3 743 cạnh) |

- `conceptnet.tsv`: `head<TAB>rel<TAB>tail`, sinh bởi `tools/fetch_conceptnet.py` (7 quan hệ IsA, PartOf, UsedFor, AtLocation,
  CapableOf, HasA, Synonym; trọng số ≥ 1; hai đầu đều trong vocab).
- `label_emb.npy` (vector CLIP chữ của từng nút, "a photo of a {}") **không** có ở đây vì `.gitignore` chặn `*.npy`;
  sinh lại bằng `tools/extract_label_emb.py --bundle <bundle> --extra <bundle>/huy_extra[_cn]`, hoặc lấy từ dataset Kaggle riêng tư
  `huychau/mmxai-task1-bundle` / `huychau/mmxai-psg-bundle`.
- Nguồn: ConceptNet 5 (https://conceptnet.io), giấy phép **CC BY-SA 4.0** — trích dẫn khi dùng lại; dữ liệu dẫn xuất giữ cùng giấy phép.
