import unittest

from mmxai.xai.demo_data import demo_graph, demo_schema
from mmxai.xai.experiment import QueryCase, run_matrix, select_balanced_case_studies
from mmxai.xai.scorer_adapter import MockPathScorer


class ExperimentTests(unittest.TestCase):
    def test_matrix_covers_sizes_and_seeds(self):
        case = QueryCase(
            "q1", "person riding horse", "indirect", "1", ("1", "2"), frozenset({"1"})
        )
        rows = run_matrix(
            [case], demo_graph(), demo_schema(),
            lambda seed: MockPathScorer({"1": 0.7, "2": 0.6}),
            explanation_sizes=(1, 3), seeds=(0, 1), random_repeats=2,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual({row.seed for row in rows}, {0, 1})
        self.assertEqual({row.k for row in rows}, {1, 3})
        self.assertEqual({row.retrieval_outcome for row in rows}, {"success"})
        self.assertEqual(len(select_balanced_case_studies(rows, per_outcome=1)), 1)

