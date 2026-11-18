"""Deterministic fixture used only for smoke tests and API demonstration."""

from __future__ import annotations

from .graph import Edge, GraphSchema, KnowledgeGraph, Node, RelationRule


def demo_graph() -> KnowledgeGraph:
    nodes = [
        Node("c_person", "concept", "person"),
        Node("c_horse", "concept", "horse"),
        Node("c_saddle", "concept", "saddle"),
        Node("img_1", "image", "horse image", external_id="1"),
        Node("img_2", "image", "street image", external_id="2"),
        Node("c_street", "concept", "street"),
    ]
    edges = [
        Edge("e1", "c_person", "c_saddle", "related_to", 0.9, "train_scene_graph", "supported"),
        Edge("e2", "c_saddle", "img_1", "contains", 1.0, "image_metadata", "supported"),
        Edge("e3", "c_horse", "img_1", "contains", 1.0, "image_metadata", "supported"),
        Edge("e4", "c_street", "img_2", "contains", 1.0, "image_metadata", "supported"),
    ]
    return KnowledgeGraph({node.id: node for node in nodes}, {edge.id: edge for edge in edges})


def demo_schema() -> GraphSchema:
    return GraphSchema(
        node_types=frozenset({"concept", "image"}),
        relations={
            "related_to": RelationRule.make(["concept"], ["concept"]),
            "contains": RelationRule.make(["concept"], ["image"]),
        },
    )

