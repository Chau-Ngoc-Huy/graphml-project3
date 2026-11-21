import unittest

from xai.evaluate_xai import XAIEvaluator
from xai.explain import EdgeContributionExplainer
from xai.perturbation import random_edge_sets
from _fixtures import tiny_graph, tiny_schema, tiny_scorer


class MetricTests(unittest.TestCase):
    def test_metrics_cover_fidelity_sparsity_rank_validity_and_evidence(self):
        graph = tiny_graph()
        scorer = tiny_scorer()
        explanation = EdgeContributionExplainer(top_k=2).explain(
            "person riding horse", "1", ["1", "2"], graph, scorer
        )
        result = XAIEvaluator(tiny_schema(), random_repeats=20, seed=3).evaluate(
            "person riding horse", "1", ["1", "2"], graph, scorer, explanation
        )
        self.assertGreater(result.graph_score_drop, 0)
        self.assertGreater(result.final_score_drop, 0)
        self.assertGreaterEqual(result.rank_drop, 0)
        self.assertAlmostEqual(result.sparsity, 1 - 2 / 3)
        self.assertEqual(result.structural_validity, 1.0)
        self.assertEqual(result.path_validity, 1.0)
        self.assertEqual(result.evidence_support_rate, 1.0)
        self.assertGreaterEqual(result.random_final_score_drop_std, 0)


    def test_random_baseline_is_deterministic_for_seed(self):
        graph = tiny_graph()
        scorer = tiny_scorer()
        explanation = EdgeContributionExplainer(top_k=1).explain("horse", "1", ["1", "2"], graph, scorer)
        evaluator = XAIEvaluator(tiny_schema(), random_repeats=8, seed=9)
        a = evaluator.evaluate("horse", "1", ["1", "2"], graph, scorer, explanation)
        b = evaluator.evaluate("horse", "1", ["1", "2"], graph, scorer, explanation)
        self.assertEqual(a, b)

    def test_random_sets_match_selected_edge_types(self):
        strata = {"a": "contains", "b": "contains", "c": "concept", "d": "concept"}
        sets = random_edge_sets(tuple(strata), 2, 20, 7, matched_to=("a", "c"), strata_by_id=strata)
        self.assertEqual(len(sets), 20)
        self.assertTrue(all(sorted(strata[e] for e in row) == ["concept", "contains"] for row in sets))
        self.assertEqual(sets, random_edge_sets(tuple(strata), 2, 20, 7,
                                                matched_to=("a", "c"), strata_by_id=strata))

