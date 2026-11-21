"""Faithful XAI for graph-enhanced text-to-image retrieval."""

from .evaluate_xai import EvaluationResult, XAIEvaluator
from .explain import EdgeContributionExplainer, Explanation
from .graph import Edge, GraphSchema, KnowledgeGraph, Node, RelationRule
from .scorer_adapter import CallableScorerAdapter, ScoreComponents, Scorer

__all__ = [
    "CallableScorerAdapter",
    "Edge",
    "EdgeContributionExplainer",
    "EvaluationResult",
    "Explanation",
    "GraphSchema",
    "KnowledgeGraph",
    "Node",
    "RelationRule",
    "ScoreComponents",
    "Scorer",
    "XAIEvaluator",
]

