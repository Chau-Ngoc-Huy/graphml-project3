# Đầu vào đồ thị của QA-Graph — bản chép từ `data/<bundle>/graph_extra*` để cả nhóm xem được

| Thư mục | Bundle | Nội dung |
|---|---|---|
| `vg/graph_extra/` | `data/vg_bundle` (VG ∩ COCO) | `labels.json` — 2 000 nút nhãn |
| `vg/graph_extra_cn/` | như trên | `labels.json` (2 000 nhãn + 1 000 khái niệm), `concepts_added.json`, `conceptnet.tsv` (cạnh ConceptNet đã lọc) |
| `psg/graph_extra/` | `data/psg_bundle` (PSG ∩ COCO) | `labels.json` — 807 nút (133 nhãn + 674 khái niệm mở rộng) |
| `psg/graph_extra_cn/` | như trên | `labels.json`, `concepts_added.json` (rỗng: khái niệm đã có trong vocab), `conceptnet.tsv` (3 743 cạnh) |

- `conceptnet.tsv`: `head<TAB>rel<TAB>tail`, sinh bởi `tools/fetch_conceptnet.py` (7 quan hệ IsA, PartOf, UsedFor, AtLocation,
  CapableOf, HasA, Synonym; trọng số ≥ 1; hai đầu đều trong vocab).
- `label_emb.npy` (vector CLIP chữ của từng nút, "a photo of a {}") **không** có ở đây vì `.gitignore` chặn `*.npy`;
  sinh lại bằng `tools/extract_label_emb.py --bundle <bundle> --extra <bundle>/graph_extra[_cn]`, hoặc lấy từ dataset Kaggle riêng tư
  `huychau/mmxai-vg-bundle` / `huychau/mmxai-psg-bundle`.
- Nguồn: ConceptNet 5 (https://conceptnet.io), giấy phép **CC BY-SA 4.0** — trích dẫn khi dùng lại; dữ liệu dẫn xuất giữ cùng giấy phép.
