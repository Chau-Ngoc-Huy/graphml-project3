"""Preserve human review and export provisional VLM assessments separately."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .image_download import _is_jpeg
from .query_registry import read_case_review_workbook
from .review_workbook import create_review_workbook


HUMAN_FIELDS = (
    "verified_group", "claim_subject", "claim_predicate", "claim_object",
    "gt_relevance", "retrieved_relevance", "hard_negative_same_objects",
    "relation_differs", "visual_reason", "reviewer",
)
VALIDATION_FIELDS = HUMAN_FIELDS[:-1]


def image_paths(image_dir: Path, row: dict) -> tuple[Path, Path]:
    stem = f"{row['dataset']}_{row['query_id']}"
    root = Path(image_dir).resolve()
    return root / f"{stem}_gt.jpg", root / f"{stem}_negative.jpg"


def load_queue(registry_path: Path, workbook_path: Path, image_dir: Path):
    """Return registry, workbook rows, untouched rows eligible for VLM review."""
    with Path(registry_path).open(newline="", encoding="utf-8") as stream:
        registry = list(csv.DictReader(stream))
    reviews = read_case_review_workbook(workbook_path)
    review_by_key = {(row["dataset"], row["query_id"]): row for row in reviews}
    registry_keys = {(row["dataset"], row["query_id"]) for row in registry}
    if len(registry) != len(reviews) or len(review_by_key) != len(reviews) or set(review_by_key) != registry_keys:
        raise ValueError("workbook rows do not match the frozen query registry")
    human_keys = {
        key for key, row in review_by_key.items()
        if any(row.get(field, "").strip() for field in HUMAN_FIELDS)
        or row.get("review_status") == "approved"
    }
    queue = [row for row in registry if (row["dataset"], row["query_id"]) not in human_keys]
    for row in queue:
        for path in image_paths(image_dir, row):
            if not path.is_file() or not _is_jpeg(path.read_bytes()):
                raise FileNotFoundError(f"missing or invalid review JPEG: {path}")
    return registry, review_by_key, human_keys, queue


def completed_records(log_path: Path) -> dict[tuple[str, str], dict]:
    if not Path(log_path).exists():
        return {}
    records = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return {(str(row["dataset"]), str(row["query_id"])): row
            for row in records if row.get("status") == "ok"}


def matching_record(record: dict | None, row: dict, image_dir: Path, model_id: str) -> bool:
    """A cached result is usable only for the same model and image bytes."""
    if not record or record.get("status") != "ok" or record.get("model") != model_id:
        return False
    gt_path, neg_path = image_paths(image_dir, row)
    return (record.get("query_sha256") == hashlib.sha256(row["query"].encode("utf-8")).hexdigest()
            and record.get("gt_sha256") == hashlib.sha256(gt_path.read_bytes()).hexdigest()
            and record.get("negative_sha256") == hashlib.sha256(neg_path.read_bytes()).hexdigest())


def export_draft(registry: list[dict], reviews: dict[tuple[str, str], dict],
                 human_keys: set[tuple[str, str]], completed: dict[tuple[str, str], dict],
                 image_dir: Path, output_dir: Path, model_id: str) -> tuple[Path, Path, dict]:
    """Export new CSV/XLSX without editing source registry, workbook or images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows, cases = [], []
    counts = {"human_partial": 0, "vlm_draft": 0, "pending_or_error": 0}
    for source in registry:
        key = (source["dataset"], source["query_id"])
        manual = reviews[key]
        record = completed.get(key) if key not in human_keys else None
        draft = record.get("result") if matching_record(record, source, image_dir, model_id) else None
        row = dict(source)
        gt_path, neg_path = image_paths(image_dir, source)
        case = dict(manual)
        case.update({
            "dataset": source["dataset"], "query_id": source["query_id"],
            "query": source["query"], "source_group": source["source_group"],
            "gt_image_id": source["gt_image_id"],
            "retrieved_image_id": source["hard_negative_candidate_id"],
            "gt_source_url": source["gt_source_url"],
            "retrieved_source_url": source["hard_negative_source_url"],
            "gt_image_file": str(gt_path), "retrieved_image_file": str(neg_path),
            "validation_origin": "human_partial" if key in human_keys else
                                 "vlm_draft" if draft else "pending_or_error",
            "vlm_confidence": draft.get("confidence", "") if draft else "",
            "vlm_uncertainty": draft.get("uncertainty_notes", "") if draft else "",
            "vlm_image_a_observation": draft.get("image_a_observation", "") if draft else "",
            "vlm_image_b_observation": draft.get("image_b_observation", "") if draft else "",
            "review_timestamp": record.get("timestamp_utc", "") if draft else manual.get("review_timestamp", ""),
        })
        if key in human_keys:
            chosen = {field: manual.get(field, "") for field in HUMAN_FIELDS}
            chosen["review_status"] = manual.get("review_status", "pending")
            origin = "human_partial"
        elif draft:
            chosen = {field: draft[field] for field in VALIDATION_FIELDS}
            chosen.update(reviewer=model_id, review_status="vlm_draft")
            case.update(chosen)
            origin = "vlm_draft"
        else:
            chosen = {}
            origin = "pending_or_error"
        counts[origin] += 1
        for field in ("verified_group", "claim_subject", "claim_predicate", "claim_object",
                      "gt_relevance", "visual_reason", "reviewer", "review_status"):
            if field in chosen:
                row[field] = chosen[field]
        if "retrieved_relevance" in chosen:
            row["negative_relevance"] = chosen["retrieved_relevance"]
        objects = chosen.get("hard_negative_same_objects", "")
        relation = chosen.get("relation_differs", "")
        if objects and relation:
            row["same_objects_different_relation"] = (
                "supported" if objects == relation == "supported"
                else "contradicted" if "contradicted" in (objects, relation)
                else "uncertain"
            )
        row.update({
            "validation_origin": origin,
            "hard_negative_same_objects": objects,
            "relation_differs": relation,
            "vlm_confidence": draft.get("confidence", "") if draft else "",
            "vlm_uncertainty": draft.get("uncertainty_notes", "") if draft else "",
            "vlm_image_a_observation": draft.get("image_a_observation", "") if draft else "",
            "vlm_image_b_observation": draft.get("image_b_observation", "") if draft else "",
        })
        output_rows.append(row)
        cases.append(case)
    csv_path = output_dir / "queries_validation_vlm_draft.csv"
    fieldnames = list(dict.fromkeys(field for row in output_rows for field in row))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    xlsx_path = output_dir / "queries_validation_vlm_draft.xlsx"
    create_review_workbook(cases, [], xlsx_path, Path(image_dir))
    return csv_path, xlsx_path, counts
