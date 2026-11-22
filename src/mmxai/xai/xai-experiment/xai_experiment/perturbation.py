"""Graph interventions that always trigger a fresh scorer call."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .graph import KnowledgeGraph
from .scorer_adapter import ScoreComponents, ScoreMap, Scorer


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    removed_edge_ids: tuple[str, ...]
    scores: ScoreMap


def score_without_edges(
    scorer: Scorer,
    query: str,
    candidate_image_ids: Sequence[str],
    graph: KnowledgeGraph,
    edge_ids: Iterable[str],
) -> PerturbationResult:
    removed = tuple(dict.fromkeys(edge_ids))
    unknown = set(removed) - set(graph.edges)
    if unknown:
        raise KeyError(f"cannot remove unknown edge IDs: {sorted(unknown)}")
    perturbed_graph = graph.without_edges(removed)
    return PerturbationResult(removed, scorer.score(query, candidate_image_ids, perturbed_graph))


def score_with_only_edges(
    scorer: Scorer,
    query: str,
    candidate_image_ids: Sequence[str],
    graph: KnowledgeGraph,
    edge_ids: Iterable[str],
) -> PerturbationResult:
    kept = tuple(dict.fromkeys(edge_ids))
    unknown = set(kept) - set(graph.edges)
    if unknown:
        raise KeyError(f"cannot retain unknown edge IDs: {sorted(unknown)}")
    return PerturbationResult(tuple(e for e in graph.edges if e not in kept), scorer.score(
        query, candidate_image_ids, graph.only_edges(kept)
    ))


def random_edge_sets(
    local_edge_ids: Sequence[str], size: int, repeats: int, seed: int,
    *, matched_to: Sequence[str] | None = None, strata_by_id: Mapping[str, str] | None = None,
) -> list[tuple[str, ...]]:
    pool = tuple(dict.fromkeys(local_edge_ids))
    if size < 0 or size > len(pool):
        raise ValueError("random deletion size must be between 0 and local edge count")
    if repeats < 1:
        raise ValueError("random_repeats must be positive")
    rng = random.Random(seed)
    if matched_to is not None:
        if strata_by_id is None or len(matched_to) != size or not set(matched_to) <= set(pool):
            raise ValueError("matched_to must be local edges and requires strata_by_id")
        if not set(pool) <= set(strata_by_id):
            raise ValueError("strata_by_id omitted local edges")
        counts: dict[str, int] = {}
        for edge_id in matched_to:
            stratum = strata_by_id[edge_id]
            counts[stratum] = counts.get(stratum, 0) + 1
        pools = {stratum: tuple(e for e in pool if strata_by_id[e] == stratum) for stratum in counts}
        return [tuple(e for stratum, count in counts.items() for e in rng.sample(pools[stratum], count))
                for _ in range(repeats)]
    return [tuple(rng.sample(pool, size)) for _ in range(repeats)]


def rank_of(scores: ScoreMap, image_id: str, candidate_order: Sequence[str]) -> int:
    """Stable, 1-based rank; ties follow the fixed candidate order."""
    candidates = tuple(map(str, candidate_order))
    missing = set(candidates) - set(scores)
    if missing:
        raise ValueError(f"scores omitted candidates: {sorted(missing)}")
    order_index = {image: i for i, image in enumerate(candidates)}
    ranked = sorted(candidates, key=lambda image: (-scores[image].final_score, order_index[image]))
    return ranked.index(str(image_id)) + 1


def score_drop(before: ScoreComponents, after: ScoreComponents) -> dict[str, float]:
    return {
        "clip_score_drop": before.clip_score - after.clip_score,
        "graph_score_drop": before.graph_score - after.graph_score,
        "final_score_drop": before.final_score - after.final_score,
    }
