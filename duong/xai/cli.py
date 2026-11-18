"""Reproducible smoke demo for the complete explanation/evaluation path."""

from __future__ import annotations

import argparse

from .demo_data import demo_graph, demo_schema
from .evaluate_xai import XAIEvaluator
from .explain import EdgeContributionExplainer
from .io import explanation_to_dict, write_json, write_results_csv
from .scorer_adapter import MockPathScorer
from .visualize import format_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/demo")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    graph, schema = demo_graph(), demo_schema()
    query, candidates, image_id = "person riding horse", ("1", "2"), "1"
    scorer = MockPathScorer({"1": 0.72, "2": 0.69}, alpha=0.6, max_hops=2)
    explanation = EdgeContributionExplainer(top_k=args.top_k).explain(
        query, image_id, candidates, graph, scorer
    )
    metrics = XAIEvaluator(schema, random_repeats=30, seed=args.seed).evaluate(
        query, image_id, candidates, graph, scorer, explanation
    )
    write_json(explanation_to_dict(explanation, graph), f"{args.output_dir}/explanation.json")
    write_results_csv([metrics], f"{args.output_dir}/xai_results.csv")
    print(format_summary(explanation, metrics, graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

