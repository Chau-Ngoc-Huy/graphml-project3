"""Lọc ConceptNet 5.7 (conceptnet-assertions-5.7.0.csv.gz, ~500 MB) thành conceptnet.tsv cho vocab nhãn VG.

Giữ cạnh tiếng Anh, quan hệ trong --relations, có ít nhất một đầu thuộc vocab nhãn; với --add-concepts, đầu còn lại
(khái niệm mới như "child", "sport") được thêm vào labels để có đỉnh khái niệm — sau đó phải chạy lại
extract_clip.py --labels để có label_emb cho khái niệm mới.

Tải: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
Dùng: python tools/fetch_conceptnet.py --assertions conceptnet-assertions-5.7.0.csv.gz --data <data_dir> \
        [--relations IsA PartOf UsedFor AtLocation CapableOf HasA] [--min-weight 1.0] [--add-concepts 2000]
"""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--assertions", required=True)
p.add_argument("--data", required=True)
p.add_argument("--relations", nargs="+", default=["IsA", "PartOf", "UsedFor", "AtLocation", "CapableOf", "HasA"])
p.add_argument("--min-weight", type=float, default=1.0)
p.add_argument("--add-concepts", type=int, default=0, help="thêm tối đa N khái niệm ngoài vocab (theo số cạnh)")
a = p.parse_args()

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


edges, outside = [], Counter()
with gzip.open(a.assertions, "rt", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        rel = parts[1].strip("/").split("/")[-1]
        if rel not in rels:
            continue
        h, t = term(parts[2]), term(parts[3])
        if not h or not t or h == t:
            continue
        w = json.loads(parts[4]).get("weight", 1.0)
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
print(f"{len(set(keep))} cạnh, {len(new)} khái niệm mới → {root / 'conceptnet.tsv'}; nhớ chạy extract_clip.py --labels")
