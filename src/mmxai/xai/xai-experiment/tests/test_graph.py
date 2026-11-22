import unittest

from xai_experiment.graph import Edge, KnowledgeGraph, Node


class GraphTests(unittest.TestCase):
    def test_without_edges_is_non_mutating(self):
        graph = KnowledgeGraph.from_records(
            [Node("c", "concept"), Node("i", "image", external_id="7")],
            [Edge("e", "c", "i", "contains")],
        )
        changed = graph.without_edges(["e"])
        self.assertEqual(set(graph.edges), {"e"})
        self.assertFalse(changed.edges)
        self.assertEqual(graph.image_node_id("7"), "i")


    def test_rejects_dangling_and_duplicate_records(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            KnowledgeGraph.from_records([Node("x", "concept"), Node("x", "concept")], [])
        with self.assertRaisesRegex(ValueError, "dangling"):
            KnowledgeGraph.from_records([Node("x", "concept")], [Edge("e", "x", "missing", "r")])
