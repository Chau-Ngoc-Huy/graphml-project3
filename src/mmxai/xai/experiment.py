"""Batch experiment matrix for queries, k values, and random seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .evaluate_xai import EvaluationResult, XAIEvaluator
from .explain import EdgeContributionExplainer, Explanation
from .graph import GraphSchema, KnowledgeGraph
from .perturbation import rank_of
from .scorer_adapter import Scorer


@dataclass(frozen=True, slots=True)
class QueryCase:
    query_id: str
    query_text: str
    query_type: str
    image_id: str
    candidate_image_ids: tuple[str, ...]
    relevant_image_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CaseStudy:
    query_id: str
    query_type: str
    image_id: str
    retrieval_outcome: str
    k: int
    seed: int
    explanation: Explanation
    metrics: EvaluationResult


def run_matrix(
    cases: Iterable[QueryCase],
    graph: KnowledgeGraph,
    schema: GraphSchema,
    scorer_factory: Callable[[int], Scorer],
    *,
    explanation_sizes: Sequence[int] = (1, 3, 5),
    seeds: Sequence[int] = (0, 1, 2),
    random_repeats: int = 30,
    retrieval_k: int = 10,
) -> list[CaseStudy]:
    """Run a reproducible matrix; the caller writes rows/case-study artifacts."""
    case_rows = list(cases)
    if not case_rows:
        raise ValueError("at least one query case is required")
    output: list[CaseStudy] = []
    for seed in seeds:
        scorer = scorer_factory(seed)
        for case in case_rows:
            candidates = tuple(map(str, case.candidate_image_ids))
            scores = scorer.score(case.query_text, candidates, graph)
            target_rank = rank_of(scores, case.image_id, candidates)
            is_relevant = not case.relevant_image_ids or case.image_id in case.relevant_image_ids
            outcome = "success" if is_relevant and target_rank <= retrieval_k else "failure"
            for size in explanation_sizes:
                explanation = EdgeContributionExplainer(top_k=size).explain(
                    case.query_text, case.image_id, candidates, graph, scorer
                )
                metrics = XAIEvaluator(schema, random_repeats=random_repeats, seed=seed).evaluate(
                    case.query_text, case.image_id, candidates, graph, scorer, explanation
                )
                output.append(CaseStudy(
                    case.query_id, case.query_type, case.image_id, outcome,
                    size, seed, explanation, metrics,
                ))
    return output


def select_balanced_case_studies(
    rows: Sequence[CaseStudy], per_outcome: int = 5
) -> list[CaseStudy]:
    """Select deterministic success/failure examples without redefining relevance."""
    chosen = []
    for outcome in ("success", "failure"):
        matches = [row for row in rows if row.retrieval_outcome == outcome]
        unique = {}
        for row in matches:
            unique.setdefault((row.query_id, row.image_id), row)
        chosen.extend(list(unique.values())[:per_outcome])
    return chosen
