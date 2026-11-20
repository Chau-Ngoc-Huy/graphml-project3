import unittest

from mmxai.xai.demo_data import demo_graph
from mmxai.xai.scorer_adapter import CallableScorerAdapter, MockPathScorer, ScoreComponents


class ScorerTests(unittest.TestCase):
    def test_mock_scorer_recomputes_graph_branch_but_keeps_clip_fixed(self):
        graph = demo_graph()
        scorer = MockPathScorer({"1": 0.7, "2": 0.6}, alpha=0.5, max_hops=2)
        before = scorer.score("person riding horse", ["1", "2"], graph)["1"]
        after = scorer.score("person riding horse", ["1", "2"], graph.without_edges(["e3"]))["1"]
        self.assertEqual(before.clip_score, after.clip_score)
        self.assertGreater(before.graph_score, after.graph_score)
        self.assertGreater(before.final_score, after.final_score)


    def test_local_edges_are_actual_scoring_paths(self):
        scorer = MockPathScorer({"1": 0.7}, max_hops=2)
        self.assertEqual(
            set(scorer.local_edge_ids("person riding horse", "1", demo_graph())),
            {"e1", "e2", "e3"},
        )

    def test_callable_adapter_passes_perturbed_graph_to_real_callback(self):
        seen_edge_counts = []

        def score_fn(query, candidates, graph):
            seen_edge_counts.append(len(graph.edges))
            return {image: ScoreComponents(0.2, len(graph.edges), 0.2 + len(graph.edges))
                    for image in candidates}

        adapter = CallableScorerAdapter(score_fn, lambda query, image, graph: tuple(graph.edges))
        graph = demo_graph()
        adapter.score("horse", ["1"], graph)
        adapter.score("horse", ["1"], graph.without_edges(["e3"]))
        self.assertEqual(seen_edge_counts, [4, 3])

