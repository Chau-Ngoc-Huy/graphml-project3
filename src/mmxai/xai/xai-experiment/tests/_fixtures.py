"""Small scorer and graph for testing the XAI contracts."""

from xai_experiment.graph import Edge, GraphSchema, KnowledgeGraph, Node, RelationRule
from xai_experiment.scorer_adapter import CallableScorerAdapter, ScoreComponents


def tiny_graph():
    return KnowledgeGraph.from_records(
        [Node("c1", "label", "person"), Node("c2", "label", "horse"),
         Node("img1", "image", external_id="1"), Node("img2", "image", external_id="2")],
        [Edge("e1", "c1", "img1", "contains", provenance="image", support_status="supported"),
         Edge("e2", "c2", "img1", "contains", provenance="image", support_status="supported"),
         Edge("e3", "c1", "c2", "concept", provenance="ConceptNet", support_status="supported"),
         Edge("e4", "c2", "img2", "contains", provenance="image", support_status="supported")],
    )


def tiny_schema():
    return GraphSchema(frozenset({"label", "image"}), {
        "contains": RelationRule.make({"label"}, {"image"}),
        "concept": RelationRule.make({"label"}, {"label"}),
    })


def tiny_scorer():
    def score_fn(query, candidate_ids, graph):
        active = set(graph.edges)
        graph_scores = {
            "1": 0.7 * ("e1" in active) + 0.5 * ("e2" in active) + 0.1 * ("e3" in active),
            "2": 0.2 * ("e4" in active),
        }
        clips = {"1": 0.6, "2": 0.5}
        return {str(i): ScoreComponents(clips[str(i)], graph_scores[str(i)],
                                        clips[str(i)] + graph_scores[str(i)]) for i in candidate_ids}

    def local_edges_fn(query, image_id, graph):
        eligible = ("e1", "e2", "e3") if str(image_id) == "1" else ("e4",)
        return tuple(e for e in eligible if e in graph.edges)

    return CallableScorerAdapter(score_fn, local_edges_fn)
