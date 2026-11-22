"""JSON/CSV serialization at the XAI boundary."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .evaluate_xai import EvaluationResult
from .explain import Explanation
from .graph import Edge, GraphSchema, KnowledgeGraph, Node, RelationRule


def load_graph(path: str | Path) -> KnowledgeGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KnowledgeGraph(
        nodes={str(row["id"]): Node(
            id=str(row["id"]), type=row["type"], label=row.get("label", ""),
            external_id=str(row["external_id"]) if row.get("external_id") is not None else None,
            metadata=row.get("metadata", {}),
        ) for row in data["nodes"]},
        edges={str(row["id"]): Edge(
            id=str(row["id"]), source=str(row["source"]), target=str(row["target"]),
            relation=row["relation"], weight=float(row.get("weight", 1.0)),
            provenance=row.get("provenance"), support_status=row.get("support_status", "uncertain"),
            metadata=row.get("metadata", {}),
        ) for row in data["edges"]},
    )


def load_schema(path: str | Path) -> GraphSchema:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GraphSchema(
        node_types=frozenset(data.get("node_types", [])),
        relations={name: RelationRule.make(rule["source_types"], rule["target_types"])
                   for name, rule in data["relations"].items()},
    )


def explanation_to_dict(explanation: Explanation, graph: KnowledgeGraph) -> dict:
    return {
        "query": explanation.query,
        "image_id": explanation.image_id,
        "method": explanation.method,
        "baseline": asdict(explanation.baseline),
        "edge_ids": list(explanation.edge_ids),
        "local_edge_ids": list(explanation.local_edge_ids),
        "contributions": [asdict(row) for row in explanation.contributions],
        "paths": [asdict(path) for path in explanation.paths],
        "edges": [asdict(graph.edges[edge_id]) for edge_id in explanation.edge_ids],
    }


def write_json(data: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_results_csv(results: Iterable[EvaluationResult], path: str | Path) -> None:
    rows = [result.to_dict() for result in results]
    if not rows:
        raise ValueError("cannot write an empty result set")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
