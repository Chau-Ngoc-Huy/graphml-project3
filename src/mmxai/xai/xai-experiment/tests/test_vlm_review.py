import csv
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from xai_experiment.query_registry import read_case_review_workbook
from xai_experiment.review_workbook import create_review_workbook
from xai_experiment.vlm_review import export_draft, load_queue, matching_record


class VlmReviewTests(unittest.TestCase):
    def test_export_keeps_human_work_and_marks_model_labels_as_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            jpeg = b"\xff\xd8" + b"0" * 120 + b"\xff\xd9"
            for query_id in ("1", "2"):
                for role in ("gt", "negative"):
                    (images / f"vg_{query_id}_{role}.jpg").write_bytes(jpeg)

            registry = []
            cases = []
            for query_id in ("1", "2"):
                row = {
                    "dataset": "vg", "query_id": query_id, "query": "truck near hay",
                    "source_group": "direct", "gt_image_id": "10",
                    "hard_negative_candidate_id": "20", "gt_source_url": "",
                    "hard_negative_source_url": "", "review_status": "pending",
                }
                registry.append(row)
                cases.append({**row, "retrieved_image_id": "20",
                              "gt_image_file": str(images / f"vg_{query_id}_gt.jpg"),
                              "retrieved_image_file": str(images / f"vg_{query_id}_negative.jpg")})
            cases[0].update(claim_subject="truck", visual_reason="Human observation")
            registry_path = root / "registry.csv"
            with registry_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(registry[0]))
                writer.writeheader()
                writer.writerows(registry)
            workbook = create_review_workbook(cases, [], root / "input.xlsx", images)

            rows, reviews, human_keys, queue = load_queue(registry_path, workbook, images)
            self.assertEqual(human_keys, {("vg", "1")})
            self.assertEqual([row["query_id"] for row in queue], ["2"])
            result = {
                "verified_group": "direct", "claim_subject": "truck",
                "claim_predicate": "near", "claim_object": "hay",
                "gt_relevance": "supported", "retrieved_relevance": "contradicted",
                "hard_negative_same_objects": "contradicted", "relation_differs": "uncertain",
                "visual_reason": "Model observation", "confidence": "medium",
                "uncertainty_notes": "truck partially hidden",
                "image_a_observation": "truck and hay", "image_b_observation": "hay only",
            }
            record = {
                "status": "ok", "model": "gemini-3.8-flash", "result": result,
                "timestamp_utc": "2026-11-22T00:00:00+00:00",
                "query_sha256": hashlib.sha256(b"truck near hay").hexdigest(),
                "gt_sha256": hashlib.sha256(jpeg).hexdigest(),
                "negative_sha256": hashlib.sha256(jpeg).hexdigest(),
            }
            self.assertTrue(matching_record(record, rows[1], images, "gemini-3.8-flash"))
            csv_path, xlsx_path, counts = export_draft(
                rows, reviews, human_keys, {("vg", "1"): record, ("vg", "2"): record},
                images, root / "output", "gemini-3.8-flash")
            self.assertEqual(counts, {"human_partial": 1, "vlm_draft": 1,
                                      "pending_or_error": 0})
            with csv_path.open(newline="", encoding="utf-8") as stream:
                output = list(csv.DictReader(stream))
            self.assertEqual(output[0]["visual_reason"], "Human observation")
            self.assertEqual(output[1]["review_status"], "vlm_draft")
            self.assertEqual(output[1]["same_objects_different_relation"], "contradicted")
            exported = read_case_review_workbook(xlsx_path)
            self.assertEqual(exported[0]["claim_subject"], "truck")
            self.assertEqual(exported[0]["visual_reason"], "Human observation")
            self.assertEqual(exported[0]["vlm_image_b_observation"], "")
            self.assertEqual(exported[1]["vlm_image_b_observation"], "hay only")
            self.assertEqual(exported[1]["review_status"], "vlm_draft")
            with zipfile.ZipFile(xlsx_path) as archive:
                self.assertEqual(len([name for name in archive.namelist()
                                      if name.startswith("xl/media/")]), 4)
            self.assertFalse(matching_record({**record, "negative_sha256": "old"},
                                             rows[1], images, "gemini-3.8-flash"))


if __name__ == "__main__":
    unittest.main()
