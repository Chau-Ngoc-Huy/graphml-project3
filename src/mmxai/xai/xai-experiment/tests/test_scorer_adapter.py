import unittest

from xai_experiment.scorer_adapter import CallableScorerAdapter, ScoreComponents
from _fixtures import tiny_graph, tiny_scorer


class ScorerTests(unittest.TestCase):
    def test_adapter_recomputes_graph_branch_but_keeps_clip_fixed(self):
        graph = tiny_graph()
        scorer = tiny_scorer()
        before = scorer.score("person riding horse", ["1", "2"], graph)["1"]
        after = scorer.score("person riding horse", ["1", "2"], graph.without_edges(["e3"]))["1"]
        self.assertEqual(before.clip_score, after.clip_score)
        self.assertGreater(before.graph_score, after.graph_score)
        self.assertGreater(before.final_score, after.final_score)


    def test_local_edges_are_actual_scoring_edges(self):
        scorer = tiny_scorer()
        self.assertEqual(
            set(scorer.local_edge_ids("person riding horse", "1", tiny_graph())),
            {"e1", "e2", "e3"},
        )

    def test_callable_adapter_passes_perturbed_graph_to_real_callback(self):
        seen_edge_counts = []

        def score_fn(query, candidates, graph):
            seen_edge_counts.append(len(graph.edges))
            return {image: ScoreComponents(0.2, len(graph.edges), 0.2 + len(graph.edges))
                    for image in candidates}

        adapter = CallableScorerAdapter(score_fn, lambda query, image, graph: tuple(graph.edges))
        graph = tiny_graph()
        adapter.score("horse", ["1"], graph)
        adapter.score("horse", ["1"], graph.without_edges(["e3"]))
        self.assertEqual(seen_edge_counts, [4, 3])
