"""Scorer contract, callback adapter, and an auditable mock path scorer."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

from .graph import KnowledgeGraph


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    clip_score: float
    graph_score: float
    final_score: float


ScoreMap = dict[str, ScoreComponents]


@runtime_checkable
class Scorer(Protocol):
    def score(
        self, query: str, candidate_image_ids: Sequence[str], graph: KnowledgeGraph
    ) -> ScoreMap: ...

    def local_edge_ids(self, query: str, image_id: str, graph: KnowledgeGraph) -> Sequence[str]: ...


class CallableScorerAdapter:
    """Connect any real GNN/reranker without coupling the evaluator to its framework.

    ``score_fn`` must perform a fresh forward pass using the supplied graph. It must
    return all candidates because final-score normalisation may depend on the pool.
    """

    def __init__(
        self,
        score_fn: Callable[[str, Sequence[str], KnowledgeGraph], Mapping[str, ScoreComponents | Mapping[str, float]]],
        local_edges_fn: Callable[[str, str, KnowledgeGraph], Sequence[str]],
    ) -> None:
        self._score_fn = score_fn
        self._local_edges_fn = local_edges_fn

    def score(self, query: str, candidate_image_ids: Sequence[str], graph: KnowledgeGraph) -> ScoreMap:
        raw = self._score_fn(query, candidate_image_ids, graph)
        normalized_raw = {str(image_id): value for image_id, value in raw.items()}
        missing = set(map(str, candidate_image_ids)) - set(normalized_raw)
        if missing:
            raise ValueError(f"scorer omitted candidates: {sorted(missing)}")
        out: ScoreMap = {}
        for image_id in map(str, candidate_image_ids):
            value = normalized_raw[image_id]
            out[image_id] = value if isinstance(value, ScoreComponents) else ScoreComponents(
                float(value["clip_score"]), float(value["graph_score"]), float(value["final_score"])
            )
        return out

    def local_edge_ids(self, query: str, image_id: str, graph: KnowledgeGraph) -> Sequence[str]:
        edge_ids = tuple(dict.fromkeys(self._local_edges_fn(query, str(image_id), graph)))
        unknown = set(edge_ids) - set(graph.edges)
        if unknown:
            raise ValueError(f"local edge selector returned unknown IDs: {sorted(unknown)}")
        return edge_ids


class MockPathScorer:
    """Transparent development scorer; never a substitute for the trained model.

    Query terms link to concept labels. Graph score is the saturating sum of
    products of edge weights along simple undirected paths to an image node.
    CLIP scores remain fixed under graph perturbation.
    """

    def __init__(
        self,
        clip_scores: Mapping[str, float] | Mapping[str, Mapping[str, float]],
        *,
        alpha: float = 0.6,
        max_hops: int = 2,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if max_hops < 1:
            raise ValueError("max_hops must be positive")
        self.clip_scores = clip_scores
        self.alpha = alpha
        self.max_hops = max_hops

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _query_nodes(self, query: str, graph: KnowledgeGraph) -> list[str]:
        qt = self._tokens(query)
        return [
            n.id for n in graph.nodes.values()
            if n.type != "image" and self._tokens(n.label) and self._tokens(n.label) <= qt
        ]

    def paths(self, query: str, image_id: str, graph: KnowledgeGraph) -> list[tuple[tuple[str, ...], float]]:
        target = graph.image_node_id(str(image_id))
        adjacency = graph.adjacency(undirected=True)
        found: list[tuple[tuple[str, ...], float]] = []
        for start in self._query_nodes(query, graph):
            stack = [(start, (start,), tuple(), 1.0)]
            while stack:
                node, seen_nodes, edge_ids, weight = stack.pop()
                if node == target and edge_ids:
                    found.append((edge_ids, weight / len(edge_ids)))
                    continue
                if len(edge_ids) >= self.max_hops:
                    continue
                for nxt, edge_id in adjacency[node]:
                    if nxt in seen_nodes:
                        continue
                    if graph.nodes[nxt].type == "image" and nxt != target:
                        continue
                    edge = graph.edges[edge_id]
                    stack.append((nxt, seen_nodes + (nxt,), edge_ids + (edge_id,), weight * max(0.0, edge.weight)))
        unique: dict[tuple[str, ...], float] = {}
        for edge_ids, weight in found:
            unique[edge_ids] = max(weight, unique.get(edge_ids, 0.0))
        return sorted(unique.items())

    def _clip(self, query: str, image_id: str) -> float:
        nested = self.clip_scores.get(query)
        if isinstance(nested, Mapping):
            return float(nested.get(image_id, 0.0))
        return float(self.clip_scores.get(image_id, 0.0))

    def score(self, query: str, candidate_image_ids: Sequence[str], graph: KnowledgeGraph) -> ScoreMap:
        out = {}
        for raw_id in candidate_image_ids:
            image_id = str(raw_id)
            raw_graph = sum(weight for _, weight in self.paths(query, image_id, graph))
            graph_score = 1.0 - math.exp(-raw_graph)
            clip_score = self._clip(query, image_id)
            out[image_id] = ScoreComponents(
                clip_score=clip_score,
                graph_score=graph_score,
                final_score=self.alpha * clip_score + (1.0 - self.alpha) * graph_score,
            )
        return out

    def local_edge_ids(self, query: str, image_id: str, graph: KnowledgeGraph) -> Sequence[str]:
        return tuple(dict.fromkeys(e for path, _ in self.paths(query, str(image_id), graph) for e in path))

