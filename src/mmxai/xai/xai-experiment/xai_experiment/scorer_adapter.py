"""Scorer contract and callback adapter for trained retrieval models."""

from __future__ import annotations

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
