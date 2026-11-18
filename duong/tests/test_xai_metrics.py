import unittest

from xai.demo_data import demo_graph, demo_schema
from xai.evaluate_xai import XAIEvaluator
from xai.explain import EdgeContributionExplainer
from xai.scorer_adapter import MockPathScorer


class MetricTests(unittest.TestCase):
    def test_metrics_cover_fidelity_sparsity_rank_validity_and_evidence(self):
        graph = demo_graph()
        scorer = MockPathScorer({"1": 0.60, "2": 0.69}, alpha=0.5, max_hops=2)
        explanation = EdgeContributionExplainer(top_k=2).explain(
            "person riding horse", "1", ["1", "2"], graph, scorer
        )
        result = XAIEvaluator(demo_schema(), random_repeats=20, seed=3).evaluate(
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
        graph = demo_graph()
        scorer = MockPathScorer({"1": 0.6, "2": 0.5})
        explanation = EdgeContributionExplainer(top_k=1).explain("horse", "1", ["1", "2"], graph, scorer)
        evaluator = XAIEvaluator(demo_schema(), random_repeats=8, seed=9)
        a = evaluator.evaluate("horse", "1", ["1", "2"], graph, scorer, explanation)
        b = evaluator.evaluate("horse", "1", ["1", "2"], graph, scorer, explanation)
        self.assertEqual(a, b)

