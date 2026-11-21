import csv
import tempfile
import unittest
from pathlib import Path

from xai.query_registry import validated_registry_rows


class DirectRegistryTests(unittest.TestCase):
    def test_source_indirect_and_missing_negative_do_not_block_direct_query(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.csv"
            rows = []
            for dataset in ("vg", "psg"):
                for index in range(13):
                    rows.append({
                        "dataset": dataset, "query_id": str(index),
                        "source_group": "indirect", "verified_group": "direct" if index < 12 else "relational",
                        "review_status": "approved", "gt_relevance": "supported",
                        "negative_relevance": "", "same_objects_different_relation": "",
                        "claim_subject": "truck", "claim_predicate": "near",
                        "claim_object": "hay", "visual_reason": "Visible truck and hay",
                        "reviewer": "human_1",
                    })
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            selected = validated_registry_rows(path)
            self.assertEqual(len(selected), 24)
            self.assertTrue(all(row["source_group"] == "indirect" for row in selected))
            rows[0]["reviewer"] = "gemini-3.8-flash"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(validated_registry_rows(path)), 23)
            rows[1]["gt_relevance"] = "contradicted"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(validated_registry_rows(path)), 22)


if __name__ == "__main__":
    unittest.main()
