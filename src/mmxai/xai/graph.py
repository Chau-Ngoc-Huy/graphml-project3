"""Small, dependency-free graph model with stable node/edge identifiers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    type: str
    label: str = ""
    external_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0
    provenance: str | None = None
    support_status: str = "uncertain"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.support_status not in {"supported", "unsupported", "uncertain"}:
            raise ValueError(f"invalid support_status for edge {self.id!r}")


@dataclass(frozen=True, slots=True)
class RelationRule:
    source_types: frozenset[str]
    target_types: frozenset[str]

    @classmethod
    def make(cls, source_types: Iterable[str], target_types: Iterable[str]) -> "RelationRule":
        return cls(frozenset(source_types), frozenset(target_types))


@dataclass(frozen=True, slots=True)
class GraphSchema:
    node_types: frozenset[str] = frozenset()
    relations: Mapping[str, RelationRule] = field(default_factory=dict)

    def validate_edge(self, edge: Edge, graph: "KnowledgeGraph") -> bool:
        if edge.source not in graph.nodes or edge.target not in graph.nodes:
            return False
        source, target = graph.nodes[edge.source], graph.nodes[edge.target]
        if self.node_types and (source.type not in self.node_types or target.type not in self.node_types):
            return False
        rule = self.relations.get(edge.relation)
        return bool(
            rule
            and source.type in rule.source_types
            and target.type in rule.target_types
        )


@dataclass(frozen=True)
class KnowledgeGraph:
    nodes: Mapping[str, Node]
    edges: Mapping[str, Edge]

    def __post_init__(self) -> None:
        nodes, edges = dict(self.nodes), dict(self.edges)
        if len(nodes) != len(self.nodes) or any(k != v.id for k, v in nodes.items()):
            raise ValueError("node mapping keys must equal unique node IDs")
        if len(edges) != len(self.edges) or any(k != v.id for k, v in edges.items()):
            raise ValueError("edge mapping keys must equal unique edge IDs")
        dangling = [e.id for e in edges.values() if e.source not in nodes or e.target not in nodes]
        if dangling:
            raise ValueError(f"dangling edges: {dangling}")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def from_records(cls, nodes: Iterable[Node], edges: Iterable[Edge]) -> "KnowledgeGraph":
        node_rows, edge_rows = list(nodes), list(edges)
        node_map = {n.id: n for n in node_rows}
        edge_map = {e.id: e for e in edge_rows}
        if len(node_map) != len(node_rows):
            raise ValueError("duplicate node IDs")
        if len(edge_map) != len(edge_rows):
            raise ValueError("duplicate edge IDs")
        return cls(node_map, edge_map)

    def without_edges(self, edge_ids: Iterable[str]) -> "KnowledgeGraph":
        removed = set(edge_ids)
        return replace(self, edges={k: v for k, v in self.edges.items() if k not in removed})

    def only_edges(self, edge_ids: Iterable[str]) -> "KnowledgeGraph":
        kept = set(edge_ids)
        return replace(self, edges={k: v for k, v in self.edges.items() if k in kept})

    def incident_edge_ids(self, node_id: str) -> tuple[str, ...]:
        return tuple(e.id for e in self.edges.values() if node_id in (e.source, e.target))

    def image_node_id(self, image_id: str) -> str:
        image_id = str(image_id)
        matches = [
            n.id for n in self.nodes.values()
            if n.type == "image" and (n.id == image_id or str(n.external_id) == image_id)
        ]
        if len(matches) != 1:
            raise KeyError(f"expected one graph node for image_id={image_id!r}, found {len(matches)}")
        return matches[0]

    def adjacency(self, *, undirected: bool = False) -> dict[str, list[tuple[str, str]]]:
        out = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            out[edge.source].append((edge.target, edge.id))
            if undirected:
                out[edge.target].append((edge.source, edge.id))
        return out

