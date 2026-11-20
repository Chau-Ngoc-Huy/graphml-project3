"""Lọc ConceptNet 5.7 (conceptnet-assertions-5.7.0.csv.gz, ~500 MB) thành conceptnet.tsv cho vocab nhãn VG.

Giữ cạnh tiếng Anh, quan hệ trong --relations, có ít nhất một đầu thuộc vocab nhãn; với --add-concepts, đầu còn lại
(khái niệm mới như "child", "sport") được thêm vào labels để có đỉnh khái niệm — sau đó phải chạy lại
extract_clip.py --labels để có label_emb cho khái niệm mới.

Nguồn dự phòng (khi S3 lỗi): huggingface.co/datasets/peandrew/conceptnet_en_simple (.parquet).
Tải: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
Dùng: python tools/fetch_conceptnet.py --assertions conceptnet-assertions-5.7.0.csv.gz --data <data_dir> \
        [--relations IsA PartOf UsedFor AtLocation CapableOf HasA Synonym] [--min-weight 1.0] [--add-concepts 2000]
Bundle Nhiệm vụ 1 (Thắng): --bundle data/vg_bundle  → vocab = data/object_vocab_train.json, ghi vào <bundle>/graph_extra/,
        sau đó chạy tools/extract_label_emb.py --bundle data/vg_bundle (tự gộp concepts_added.json).
"""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--assertions", required=True)
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--data", help="thư mục dữ liệu theo hợp đồng (features/labels.json)")
g.add_argument("--bundle", help="bundle Nhiệm vụ 1 của Thắng (đọc data/object_vocab_train.json, ghi graph_extra/)")
p.add_argument("--relations", nargs="+", default=["IsA", "PartOf", "UsedFor", "AtLocation", "CapableOf", "HasA", "Synonym"])
p.add_argument("--min-weight", type=float, default=1.0)
p.add_argument("--out", help="thư mục ghi (mặc định <bundle>/graph_extra hoặc <data>)")
p.add_argument("--add-concepts", type=int, default=0, help="thêm tối đa N khái niệm ngoài vocab (theo số cạnh)")
a = p.parse_args()

if a.bundle:
    root = Path(a.out) if a.out else Path(a.bundle) / "graph_extra"
    root.mkdir(parents=True, exist_ok=True)
    labels_path = Path(a.bundle) / "data" / "object_vocab_train.json"  # vocab GỐC, không lẫn khái niệm đã thêm lần trước
else:
    root = Path(a.data)
    labels_path = root / "features" / "labels.json"
if labels_path.exists():
    vocab = set(json.loads(labels_path.read_text()))
else:  # chưa có features: lấy vocab từ objects.json
    vocab = {o["names"][0].strip().lower() for r in json.loads((root / "vg" / "objects.json").read_text())
             for o in r.get("objects", []) if o.get("names")}
rels = set(a.relations)


def term(uri: str) -> str | None:
    m = re.match(r"^/c/en/([^/]+)", uri)
    return m.group(1).replace("_", " ") if m else None


def rows():
    """(rel_uri, arg1_uri, arg2_uri, weight) từ file gốc .csv.gz hoặc bản parquet tiếng Anh trên Hugging Face
    (peandrew/conceptnet_en_simple: cột rel, arg1, arg2, weight — dùng khi máy chủ S3 của ConceptNet không tải được)."""
    if str(a.assertions).endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(a.assertions).iter_batches(columns=["rel", "arg1", "arg2", "weight"]):
            yield from zip(*(batch.column(i).to_pylist() for i in range(4)))
        return
    with gzip.open(a.assertions, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                yield parts[1], parts[2], parts[3], json.loads(parts[4]).get("weight", 1.0)


edges, outside = [], Counter()
for rel_uri, a1, a2, w in rows():
    rel = rel_uri.strip("/").split("/")[-1]
    if rel not in rels:
        continue
    h, t = term(a1), term(a2)
    if not h or not t or h == t:
        continue
    w = 1.0 if w is None else float(w)
    if w < a.min_weight or (h not in vocab and t not in vocab):
        continue
    edges.append((h, rel, t))
    for x in (h, t):
        if x not in vocab:
            outside[x] += 1

new = {c for c, _ in outside.most_common(a.add_concepts)}
keep = [(h, r, t) for h, r, t in edges if (h in vocab or h in new) and (t in vocab or t in new)]
(root / "conceptnet.tsv").write_text("".join(f"{h}\t{r}\t{t}\n" for h, r, t in sorted(set(keep))), encoding="utf-8")
(root / "concepts_added.json").write_text(json.dumps(sorted(new), ensure_ascii=False))
rc = Counter(r for _, r, _ in set(keep))
print(f"{len(set(keep))} cạnh {dict(rc)}, {len(new)} khái niệm mới → {root / 'conceptnet.tsv'}")
print("Tiếp: " + ("tools/extract_label_emb.py --bundle " + a.bundle if a.bundle else "extract_clip.py --labels"))
