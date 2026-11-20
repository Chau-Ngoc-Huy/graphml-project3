import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from xai.demo_data import demo_graph
from xai.io import load_graph


class IOTests(unittest.TestCase):
    def test_json_graph_loader(self):
        graph = demo_graph()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps({
                "nodes": [{"id": n.id, "type": n.type, "label": n.label, "external_id": n.external_id}
                          for n in graph.nodes.values()],
                "edges": [{"id": e.id, "source": e.source, "target": e.target, "relation": e.relation,
                           "weight": e.weight, "provenance": e.provenance,
                           "support_status": e.support_status}
                          for e in graph.edges.values()],
            }))
            loaded = load_graph(path)
            self.assertEqual(loaded, graph)

