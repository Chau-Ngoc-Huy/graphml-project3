"""Tạo splits.json cố định theo seed, chia theo ẢNH (mọi caption của một ảnh nằm cùng split → không rò rỉ).

Phần chính thức là; file này để có split tạm khi chưa nhận được, cùng định dạng.
Dùng: python tools/make_split.py --data <data_dir> [--ratios 0.8 0.1 0.1] [--seed 0]
"""

import argparse
import json
import random
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", required=True)
p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
root = Path(a.data)
ids = sorted({int(c["image_id"]) for c in json.loads((root / "captions.json").read_text())})
random.Random(a.seed).shuffle(ids)
n1, n2 = int(a.ratios[0] * len(ids)), int((a.ratios[0] + a.ratios[1]) * len(ids))
out = {"seed": a.seed, "ratios": a.ratios, "train": sorted(ids[:n1]), "val": sorted(ids[n1:n2]), "test": sorted(ids[n2:])}
(root / "splits.json").write_text(json.dumps(out))
print({k: len(out[k]) for k in ("train", "val", "test")}, "→", root / "splits.json")
