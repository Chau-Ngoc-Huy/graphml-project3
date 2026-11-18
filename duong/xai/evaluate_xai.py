"""Fidelity, sparsity, structural validity, evidence, and random baseline."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Sequence

from .explain import Explanation
from .graph import GraphSchema, KnowledgeGraph
from .perturbation import random_edge_sets, rank_of, score_drop, score_without_edges
from .scorer_adapter import Scorer


@dataclass(frozen=True)
class EvaluationResult:
    query: str
    image_id: str
    method: str
    seed: int
    k: int
    n_local_edges: int
    n_explained_edges: int
    n_explained_paths: int
    average_path_length: float
    sparsity: float
    graph_score_drop: float
    final_score_drop: float
    rank_before: int
    rank_after: int
    rank_drop: int
    random_graph_score_drop_mean: float
    random_graph_score_drop_std: float
    random_final_score_drop_mean: float
    random_final_score_drop_std: float
    random_rank_drop_mean: float
    random_rank_drop_std: float
    fidelity_advantage: float
    structural_validity: float
    path_validity: float
    evidence_support_rate: float
    evidence_known_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


class XAIEvaluator:
    def __init__(
        self,
        schema: GraphSchema,
        *,
        random_repeats: int = 30,
        seed: int = 0,
    ) -> None:
        self.schema = schema
        self.random_repeats = random_repeats
        self.seed = seed

    def evaluate(
        self,
        query: str,
        image_id: str,
        candidate_image_ids: Sequence[str],
        graph: KnowledgeGraph,
        scorer: Scorer,
        explanation: Explanation,
    ) -> EvaluationResult:
        image_id = str(image_id)
        candidate_ids = tuple(map(str, candidate_image_ids))
        if explanation.query != query or explanation.image_id != image_id:
            raise ValueError("explanation does not match the requested query-image pair")
        if not set(explanation.edge_ids) <= set(explanation.local_edge_ids):
            raise ValueError("explained edges must be a subset of local candidate edges")

        before_scores = scorer.score(query, candidate_ids, graph)
        after_scores = score_without_edges(
            scorer, query, candidate_ids, graph, explanation.edge_ids
        ).scores
        before, after = before_scores[image_id], after_scores[image_id]
        drops = score_drop(before, after)
        rank_before = rank_of(before_scores, image_id, candidate_ids)
        rank_after = rank_of(after_scores, image_id, candidate_ids)

        random_graph, random_final, random_rank = [], [], []
        for edge_set in random_edge_sets(
            explanation.local_edge_ids,
            len(explanation.edge_ids),
            self.random_repeats,
            self.seed,
        ):
            scores = score_without_edges(scorer, query, candidate_ids, graph, edge_set).scores
            rd = score_drop(before, scores[image_id])
            random_graph.append(rd["graph_score_drop"])
            random_final.append(rd["final_score_drop"])
            random_rank.append(rank_of(scores, image_id, candidate_ids) - rank_before)

        valid = [self.schema.validate_edge(graph.edges[e], graph) for e in explanation.edge_ids]
        valid_paths = [_path_is_valid(path.edge_ids, path.node_ids, graph, image_id)
                       for path in explanation.paths]
        statuses = [graph.edges[e].support_status for e in explanation.edge_ids]
        supported = sum(status == "supported" for status in statuses)
        known = sum(status != "uncertain" for status in statuses)
        n_exp, n_local = len(explanation.edge_ids), len(explanation.local_edge_ids)
        path_lengths = [len(path.edge_ids) for path in explanation.paths]
        random_final_mean = statistics.fmean(random_final)
        return EvaluationResult(
            query=query,
            image_id=image_id,
            method=explanation.method,
            seed=self.seed,
            k=n_exp,
            n_local_edges=n_local,
            n_explained_edges=n_exp,
            n_explained_paths=len(explanation.paths),
            average_path_length=statistics.fmean(path_lengths) if path_lengths else 0.0,
            sparsity=1.0 - n_exp / n_local if n_local else 1.0,
            graph_score_drop=drops["graph_score_drop"],
            final_score_drop=drops["final_score_drop"],
            rank_before=rank_before,
            rank_after=rank_after,
            rank_drop=rank_after - rank_before,
            random_graph_score_drop_mean=statistics.fmean(random_graph),
            random_graph_score_drop_std=_pstdev(random_graph),
            random_final_score_drop_mean=random_final_mean,
            random_final_score_drop_std=_pstdev(random_final),
            random_rank_drop_mean=statistics.fmean(random_rank),
            random_rank_drop_std=_pstdev(random_rank),
            fidelity_advantage=drops["final_score_drop"] - random_final_mean,
            structural_validity=sum(valid) / n_exp if n_exp else 1.0,
            path_validity=sum(valid_paths) / len(valid_paths) if valid_paths else 1.0,
            evidence_support_rate=supported / n_exp if n_exp else 1.0,
            evidence_known_rate=known / n_exp if n_exp else 1.0,
        )


def _pstdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _path_is_valid(
    edge_ids: Sequence[str], node_ids: Sequence[str], graph: KnowledgeGraph, image_id: str
) -> bool:
    if len(node_ids) != len(edge_ids) + 1 or not edge_ids:
        return False
    if node_ids[-1] != graph.image_node_id(image_id):
        return False
    for edge_id, left, right in zip(edge_ids, node_ids, node_ids[1:]):
        if edge_id not in graph.edges:
            return False
        edge = graph.edges[edge_id]
        if {edge.source, edge.target} != {left, right}:
            return False
    return True

