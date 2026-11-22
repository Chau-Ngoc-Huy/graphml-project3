"""Leave-one-edge-out explanations tied directly to the supplied scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .graph import KnowledgeGraph
from .perturbation import score_without_edges
from .scorer_adapter import ScoreComponents, Scorer


@dataclass(frozen=True, slots=True)
class EdgeContribution:
    edge_id: str
    graph_contribution: float
    final_contribution: float
    baseline: ScoreComponents
    perturbed: ScoreComponents


@dataclass(frozen=True, slots=True)
class ExplanationPath:
    edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class Explanation:
    query: str
    image_id: str
    edge_ids: tuple[str, ...]
    local_edge_ids: tuple[str, ...]
    contributions: tuple[EdgeContribution, ...]
    paths: tuple[ExplanationPath, ...]
    baseline: ScoreComponents
    method: str = "leave_one_edge_out"


class EdgeContributionExplainer:
    def __init__(self, *, top_k: int = 3, positive_only: bool = True) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.positive_only = positive_only

    def explain(
        self,
        query: str,
        image_id: str,
        candidate_image_ids: Sequence[str],
        graph: KnowledgeGraph,
        scorer: Scorer,
    ) -> Explanation:
        image_id = str(image_id)
        candidate_ids = tuple(map(str, candidate_image_ids))
        if image_id not in candidate_ids:
            raise ValueError("image being explained must belong to candidate_image_ids")
        baseline = scorer.score(query, candidate_ids, graph)[image_id]
        local = tuple(dict.fromkeys(scorer.local_edge_ids(query, image_id, graph)))
        unknown = set(local) - set(graph.edges)
        if unknown:
            raise ValueError(f"scorer returned unknown local edges: {sorted(unknown)}")
        rows = []
        for edge_id in local:
            perturbed = score_without_edges(scorer, query, candidate_ids, graph, [edge_id]).scores[image_id]
            rows.append(EdgeContribution(
                edge_id=edge_id,
                graph_contribution=baseline.graph_score - perturbed.graph_score,
                final_contribution=baseline.final_score - perturbed.final_score,
                baseline=baseline,
                perturbed=perturbed,
            ))
        if self.positive_only:
            rows = [row for row in rows if row.final_contribution > 0]
        rows.sort(key=lambda row: (-row.final_contribution, -row.graph_contribution, row.edge_id))
        selected = tuple(rows[: self.top_k])
        edge_ids = tuple(row.edge_id for row in selected)
        paths = _paths_from_selected_edges(graph, image_id, edge_ids)
        return Explanation(query, image_id, edge_ids, local, selected, paths, baseline)


def _paths_from_selected_edges(
    graph: KnowledgeGraph, image_id: str, selected_edge_ids: Sequence[str]
) -> tuple[ExplanationPath, ...]:
    """Report only genuine connected paths; disconnected edges remain top edges."""
    selected = set(selected_edge_ids)
    if not selected:
        return ()
    target = graph.image_node_id(image_id)
    adjacency: dict[str, list[tuple[str, str]]] = {node: [] for node in graph.nodes}
    for edge_id in selected:
        edge = graph.edges[edge_id]
        adjacency[edge.source].append((edge.target, edge_id))
        adjacency[edge.target].append((edge.source, edge_id))
    paths: list[ExplanationPath] = []
    starts = [node for node, links in adjacency.items() if links and node != target and len(links) == 1]
    for start in starts:
        stack = [(start, (start,), tuple())]
        while stack:
            node, nodes, edges = stack.pop()
            if node == target:
                paths.append(ExplanationPath(edges, nodes))
                continue
            for nxt, edge_id in adjacency[node]:
                if nxt not in nodes:
                    stack.append((nxt, nodes + (nxt,), edges + (edge_id,)))
    unique = {(path.edge_ids, path.node_ids): path for path in paths}
    return tuple(sorted(unique.values(), key=lambda path: (len(path.edge_ids), path.edge_ids)))
