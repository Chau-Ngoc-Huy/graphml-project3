"""Nạp dữ liệu theo hợp đồng (docs/data-contract.md). Dữ liệu thật và dữ liệu giả có CÙNG cấu trúc thư mục:

    <data_dir>/vg/objects.json          định dạng Visual Genome gốc (có thể là tập con)
    <data_dir>/vg/relationships.json    định dạng Visual Genome gốc
    <data_dir>/captions.json            [{"caption_id", "image_id", "text", "group"?: "direct"|"indirect"}]
    <data_dir>/splits.json              {"train": [image_id], "val": [...], "test": [...]}  (Yến)
    <data_dir>/conceptnet.tsv           head<TAB>rel<TAB>tail  (tùy chọn, đã lọc theo vocab; tools/fetch_conceptnet.py)
    <data_dir>/synonyms.json            {"men": "man", ...}  (tùy chọn, chuẩn hóa nhãn; Yến)
    <data_dir>/features/image_emb.npy + image_ids.json       CLIP ảnh, đã L2-normalize (Thắng)
    <data_dir>/features/caption_emb.npy + caption_ids.json   CLIP caption (Thắng)
    <data_dir>/features/label_emb.npy + labels.json          CLIP text của tên nhãn/khái niệm (tools/extract_clip.py --labels)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_label(name: str, synonyms: dict[str, str]) -> str:
    s = re.sub(r"\s+", " ", name.strip().lower())
    return synonyms.get(s, s)


def _names(obj: dict) -> list[str]:
    if "names" in obj and obj["names"]:
        return obj["names"]
    return [obj["name"]] if obj.get("name") else []


@dataclass
class Bundle:
    image_ids: list[int]                     # thứ tự dòng của image_emb
    image_emb: np.ndarray                    # [N_img, D]
    image_labels: dict[int, list[str]]       # image_id -> nhãn đối tượng (đã chuẩn hóa, có lặp)
    image_rels: dict[int, list[tuple[str, str, str]]]  # image_id -> (subj, predicate, obj)
    captions: list[dict]                     # theo thứ tự caption_emb
    caption_emb: np.ndarray                  # [N_cap, D]
    splits: dict[str, list[int]]
    labels: list[str]
    label_emb: np.ndarray                    # [N_lab, D]
    concept_edges: list[tuple[str, str, str]] = field(default_factory=list)
    root: Path | None = None

    def captions_of(self, split: str, limit: int | None = None) -> list[int]:
        """Chỉ số caption (dòng trong caption_emb) có ảnh thuộc split."""
        keep = set(self.splits[split])
        idx = [i for i, c in enumerate(self.captions) if c["image_id"] in keep]
        return idx[:limit] if limit else idx


def load(data_dir: Path | str, max_images: int | None = None) -> Bundle:
    root = Path(data_dir)
    syn_path = root / "synonyms.json"
    synonyms = {k.lower(): v.lower() for k, v in _read_json(syn_path).items()} if syn_path.exists() else {}

    feat = root / "features"
    image_ids = [int(x) for x in _read_json(feat / "image_ids.json")]
    image_emb = np.load(feat / "image_emb.npy").astype(np.float32)
    splits = {k: [int(x) for x in v] for k, v in _read_json(root / "splits.json").items() if k in ("train", "val", "test")}
    if max_images:  # lượt thử: giữ tỷ lệ split
        keep = set()
        for k, v in splits.items():
            keep |= set(v[: max(2, int(max_images * len(v) / max(1, len(image_ids))))])
        rows = [i for i, x in enumerate(image_ids) if x in keep]
        image_ids, image_emb = [image_ids[i] for i in rows], image_emb[rows]
        splits = {k: [x for x in v if x in keep] for k, v in splits.items()}
    have = set(image_ids)

    image_labels: dict[int, list[str]] = {i: [] for i in image_ids}
    for rec in _read_json(root / "vg" / "objects.json"):
        iid = int(rec["image_id"])
        if iid in have:
            image_labels[iid] = [normalize_label(n, synonyms) for o in rec.get("objects", []) for n in _names(o)[:1]]
    image_rels: dict[int, list[tuple[str, str, str]]] = {i: [] for i in image_ids}
    rel_path = root / "vg" / "relationships.json"
    if rel_path.exists():
        for rec in _read_json(rel_path):
            iid = int(rec["image_id"])
            if iid not in have:
                continue
            for r in rec.get("relationships", []):
                s, o = _names(r["subject"]), _names(r["object"])
                if s and o:
                    image_rels[iid].append((normalize_label(s[0], synonyms), r["predicate"].strip().lower(),
                                            normalize_label(o[0], synonyms)))

    cap_ids = _read_json(feat / "caption_ids.json")
    cap_emb = np.load(feat / "caption_emb.npy").astype(np.float32)
    cap_by_id = {str(c["caption_id"]): c for c in _read_json(root / "captions.json")}
    rows = [i for i, cid in enumerate(cap_ids) if str(cid) in cap_by_id and int(cap_by_id[str(cid)]["image_id"]) in have]
    captions = [dict(cap_by_id[str(cap_ids[i])], image_id=int(cap_by_id[str(cap_ids[i])]["image_id"])) for i in rows]
    cap_emb = cap_emb[rows]

    labels = [normalize_label(l, synonyms) for l in _read_json(feat / "labels.json")]
    label_emb = np.load(feat / "label_emb.npy").astype(np.float32)

    concept_edges = []
    cn = root / "conceptnet.tsv"
    if cn.exists():
        for line in cn.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                concept_edges.append((normalize_label(parts[0], synonyms), parts[1], normalize_label(parts[2], synonyms)))

    return Bundle(image_ids, image_emb, image_labels, image_rels, captions, cap_emb, splits, labels, label_emb,
                  concept_edges, root)


def resolve_data(cfg: dict, cache_dir: Path) -> Path:
    """cfg.data = đường dẫn dữ liệu thật; null → sinh dữ liệu giả vào cache (tham số ở cfg.fake)."""
    if cfg.get("data"):
        return Path(cfg["data"])
    from .fake import generate

    fk = dict(cfg.get("fake") or {})
    out = cache_dir / ("fake_vg_" + "_".join(f"{k}{v}" for k, v in sorted(fk.items())) if fk else "fake_vg")
    if not (out / "splits.json").exists():
        generate(out, **fk)
    return out
