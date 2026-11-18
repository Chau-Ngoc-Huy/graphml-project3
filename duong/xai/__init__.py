"""Faithful XAI for graph-enhanced text-to-image retrieval."""

from .evaluate_xai import EvaluationResult, XAIEvaluator
from .explain import EdgeContributionExplainer, Explanation
from .experiment import CaseStudy, QueryCase, run_matrix, select_balanced_case_studies
from .graph import Edge, GraphSchema, KnowledgeGraph, Node, RelationRule
from .scorer_adapter import CallableScorerAdapter, MockPathScorer, ScoreComponents, Scorer

__all__ = [
    "CallableScorerAdapter",
    "CaseStudy",
    "Edge",
    "EdgeContributionExplainer",
    "EvaluationResult",
    "Explanation",
    "GraphSchema",
    "KnowledgeGraph",
    "MockPathScorer",
    "Node",
    "QueryCase",
    "RelationRule",
    "ScoreComponents",
    "Scorer",
    "XAIEvaluator",
    "run_matrix",
    "select_balanced_case_studies",
]

