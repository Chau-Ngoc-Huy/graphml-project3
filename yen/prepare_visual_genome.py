"""Prepare Visual Genome and COCO data for the shared MMXAI bundle contract.

The committed Task 1 split uses COCO IDs; this script maps it to Visual Genome
IDs. For a new experiment, --random-split creates a deterministic image-level
split and records it in splits.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "thang" / "kaggle_results" / "split" / "split.json"
SPLITS = ("train", "val", "test")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)


def object_names(value: Any) -> list[str]:
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, dict):
        names = value.get("names") or ([value["name"]] if value.get("name") else [])
    else:
        names = []
    return list(dict.fromkeys(
        str(name).strip() for name in names if name is not None and str(name).strip()
    ))


def repair_scene_graph(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Restore named relationship endpoints omitted from objects; report unknown endpoints."""
    objects: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []
    for obj in record.get("objects", []):
        if obj.get("object_id") is None:
            continue
        object_id = int(obj["object_id"])
        if object_id not in objects:
            objects[object_id] = dict(obj)
            ordered_ids.append(object_id)

    relationships: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    recovered: set[int] = set()

    for rel in record.get("relationships", []):
        endpoints: dict[str, dict[str, Any] | None] = {}
        for role in ("subject", "object"):
            endpoint = rel.get(role)
            object_id = rel.get(f"{role}_id")
            if object_id is None and isinstance(endpoint, dict):
                object_id = endpoint.get("object_id")
            if object_id is None:
                endpoints[role] = None
                continue

            object_id = int(object_id)
            names = object_names(endpoint)
            existing = objects.get(object_id)
            if existing is None and names:
                existing = {"object_id": object_id, "names": names}
                objects[object_id] = existing
                ordered_ids.append(object_id)
                recovered.add(object_id)
            elif existing is not None and not object_names(existing) and names:
                existing["names"] = names

            names = object_names(existing) if existing else names
            endpoints[role] = {"object_id": object_id, "names": names} if names else None

        subject, obj = endpoints["subject"], endpoints["object"]
        if subject is None or obj is None:
            unresolved.append({
                "relationship_id": rel.get("relationship_id"),
                "subject_id": rel.get("subject_id"),
                "object_id": rel.get("object_id"),
                "missing_names": [
                    role for role, endpoint in endpoints.items() if endpoint is None
                ],
            })
            continue

        normalized = {
            "predicate": str(rel.get("predicate", "")).strip(),
            "subject": subject,
            "object": obj,
        }
        if rel.get("relationship_id") is not None:
            normalized["relationship_id"] = rel["relationship_id"]
        relationships.append(normalized)

    repaired = {
        "image_id": int(record["image_id"]),
        "objects": [objects[object_id] for object_id in ordered_ids],
    }
    return {
        "objects": repaired,
        "relationships": {
            "image_id": int(record["image_id"]),
            "relationships": relationships,
        },
        "recovered_object_ids": sorted(recovered),
        "unresolved_relationships": unresolved,
    }, unresolved


def make_splits(
    vg_by_coco: dict[int, list[int]],
    available_coco_ids: set[int],
    split_file: Path | None,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"split ratios must be non-negative and sum to 1: {ratios}")

    if split_file is not None:
        reference = read_json(split_file)
        coco_splits = {
            name: {int(coco_id) for coco_id in reference[name]}
            for name in SPLITS
        }
        for index, name in enumerate(SPLITS):
            for other in SPLITS[index + 1:]:
                overlap = coco_splits[name] & coco_splits[other]
                if overlap:
                    raise ValueError(f"COCO IDs overlap between {name} and {other}: {sorted(overlap)[:10]}")
        requested = set().union(*coco_splits.values())
        missing = requested - available_coco_ids
        if missing:
            raise ValueError(
                f"fixed split contains {len(missing)} COCO IDs without a VG image and caption; "
                f"examples: {sorted(missing)[:10]}"
            )
        source = f"fixed COCO-ID split: {split_file}"
        split_seed = int(reference.get("meta", {}).get("seed_data", reference.get("seed", seed)))
    else:
        ids = sorted(available_coco_ids)
        random.Random(seed).shuffle(ids)
        train_end = int(ratios[0] * len(ids))
        val_end = int((ratios[0] + ratios[1]) * len(ids))
        coco_splits = {
            "train": set(ids[:train_end]),
            "val": set(ids[train_end:val_end]),
            "test": set(ids[val_end:]),
        }
        source = "deterministic COCO-ID shuffle; all mapped VG IDs stay together"
        split_seed = seed

    output = {
        name: sorted(
            vg_id
            for coco_id in coco_splits[name]
            for vg_id in vg_by_coco[coco_id]
        )
        for name in SPLITS
    }
    flat = [vg_id for name in SPLITS for vg_id in output[name]]
    if len(flat) != len(set(flat)):
        raise ValueError("Visual Genome image IDs overlap between splits")
    meta = {
        "source": source,
        "seed": split_seed,
        "ratios": list(ratios),
        "grouped_by": "coco_id",
        "counts": {name: len(output[name]) for name in SPLITS},
    }
    return {"seed": split_seed, **output, "meta": meta}, meta


def prepare(
    scene_graphs_path: Path,
    image_data_path: Path,
    coco_captions_path: Path,
    output: Path,
    split_file: Path | None,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    image_data = read_json(image_data_path)
    coco_images = read_json(coco_captions_path)["images"]
    scenes = read_json(scene_graphs_path)

    vg_by_coco: dict[int, list[int]] = {}
    for metadata in image_data:
        coco_id = metadata.get("coco_id")
        if coco_id is None:
            continue
        coco_id = int(coco_id)
        vg_id = int(metadata["image_id"])
        vg_by_coco.setdefault(coco_id, []).append(vg_id)

    coco_captions = {
        int(image["cocoid"]): image
        for image in coco_images
        if image.get("cocoid") is not None
    }
    scene_by_id = {int(record["image_id"]): record for record in scenes}
    available_coco_ids = {
        coco_id
        for coco_id, vg_ids in vg_by_coco.items()
        if coco_id in coco_captions and all(vg_id in scene_by_id for vg_id in vg_ids)
    }

    if split_file is not None:
        fixed_ids = set().union(*(set(read_json(split_file)[name]) for name in SPLITS))
        unavailable = {int(value) for value in fixed_ids} - available_coco_ids
        if unavailable:
            raise ValueError(
                f"{len(unavailable)} COCO IDs from the fixed split are missing captions, "
                f"VG metadata, or scene graphs; examples: {sorted(unavailable)[:10]}"
            )

    split_data, split_meta = make_splits(
        vg_by_coco, available_coco_ids, split_file, seed, ratios
    )
    selected_vg_ids = set().union(*(set(split_data[name]) for name in SPLITS))
    selected_coco_ids = {
        coco_id for coco_id, vg_ids in vg_by_coco.items()
        if selected_vg_ids.intersection(vg_ids)
    }

    object_records: list[dict[str, Any]] = []
    relationship_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "split": split_meta,
        "selected_images": len(selected_vg_ids),
        "recovered_object_ids": {},
        "unresolved_relationships": {},
    }
    for vg_id in sorted(selected_vg_ids):
        result, unresolved = repair_scene_graph(scene_by_id[vg_id])
        object_records.append(result["objects"])
        relationship_records.append(result["relationships"])
        if result["recovered_object_ids"]:
            report["recovered_object_ids"][str(vg_id)] = result["recovered_object_ids"]
        if unresolved:
            report["unresolved_relationships"][str(vg_id)] = unresolved

    captions: list[dict[str, Any]] = []
    for coco_id in sorted(selected_coco_ids):
        image = coco_captions[coco_id]
        vg_ids = sorted(set(vg_by_coco[coco_id]) & selected_vg_ids)
        for vg_id in vg_ids:
            for index, sentence in enumerate(image.get("sentences", [])):
                text = sentence.get("raw") or sentence.get("sent")
                if not text:
                    continue
                sentence_id = sentence.get("sentid", index)
                captions.append({
                    "caption_id": f"{coco_id}_{vg_id}_{sentence_id}",
                    "image_id": vg_id,
                    "text": str(text).strip(),
                })

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "captions.json", captions)
    write_json(output / "splits.json", split_data)
    write_json(output / "vg" / "objects.json", object_records)
    write_json(output / "vg" / "relationships.json", relationship_records)
    write_json(output / "preprocess_report.json", report)

    return {
        "images": len(selected_vg_ids),
        "captions": len(captions),
        "objects_recovered": sum(len(ids) for ids in report["recovered_object_ids"].values()),
        "unresolved_relationships": sum(
            len(items) for items in report["unresolved_relationships"].values()
        ),
        "split": split_meta["counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-graphs", required=True, type=Path)
    parser.add_argument("--image-data", required=True, type=Path)
    parser.add_argument("--coco-captions", required=True, type=Path,
                        help="Karpathy dataset_coco.json")
    parser.add_argument("--output", type=Path, default=ROOT / "yen" / "data" / "visual_genome_mmxai")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT,
                        help="COCO-ID split JSON; defaults to the committed MMXAI benchmark split")
    parser.add_argument("--random-split", action="store_true",
                        help="create a deterministic split instead of using --split-file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                        metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    split_file = None if args.random_split else args.split_file
    result = prepare(
        args.scene_graphs, args.image_data, args.coco_captions, args.output,
        split_file, args.seed, tuple(args.ratios),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
