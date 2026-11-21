import unittest

from xai.explain import EdgeContributionExplainer
from _fixtures import tiny_graph, tiny_scorer


class ExplainTests(unittest.TestCase):
    def test_explainer_selects_positive_edges_and_records_loo_scores(self):
        graph = tiny_graph()
        scorer = tiny_scorer()
        explanation = EdgeContributionExplainer(top_k=2).explain(
            "person riding horse", "1", ["1", "2"], graph, scorer
        )
        self.assertEqual(len(explanation.edge_ids), 2)
        self.assertLessEqual(set(explanation.edge_ids), {"e1", "e2", "e3"})
        self.assertTrue(all(row.final_contribution > 0 for row in explanation.contributions))
        self.assertTrue(all(row.baseline.final_score > row.perturbed.final_score for row in explanation.contributions))

