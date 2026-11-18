"""Optional NetworkX/Matplotlib rendering and plain-text fallback."""

from __future__ import annotations

from pathlib import Path

from .evaluate_xai import EvaluationResult
from .explain import Explanation
from .graph import KnowledgeGraph


def format_summary(explanation: Explanation, metrics: EvaluationResult, graph: KnowledgeGraph) -> str:
    lines = [
        f"query={explanation.query!r} image={explanation.image_id}",
        f"final score drop={metrics.final_score_drop:.4f}; graph score drop={metrics.graph_score_drop:.4f}; "
        f"rank {metrics.rank_before} -> {metrics.rank_after}",
        f"sparsity={metrics.sparsity:.3f}; structural validity={metrics.structural_validity:.3f}; "
        f"evidence support={metrics.evidence_support_rate:.3f}",
        "top edges:",
    ]
    for row in explanation.contributions:
        edge = graph.edges[row.edge_id]
        lines.append(
            f"  {edge.id}: {graph.nodes[edge.source].label} -[{edge.relation}]-> "
            f"{graph.nodes[edge.target].label} (final contribution={row.final_contribution:.4f})"
        )
    return "\n".join(lines)


def plot_explanation(
    explanation: Explanation,
    graph: KnowledgeGraph,
    output_path: str | Path | None = None,
):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("plotting requires: pip install -e '.[viz]'") from exc

    selected = set(explanation.edge_ids)
    nx_graph = nx.DiGraph()
    for edge_id in selected:
        edge = graph.edges[edge_id]
        nx_graph.add_node(edge.source, label=graph.nodes[edge.source].label or edge.source)
        nx_graph.add_node(edge.target, label=graph.nodes[edge.target].label or edge.target)
        nx_graph.add_edge(edge.source, edge.target, label=edge.relation)
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(nx_graph, seed=7)
    labels = nx.get_node_attributes(nx_graph, "label")
    nx.draw_networkx(nx_graph, pos, labels=labels, node_color="#dcecff", node_size=1800, ax=ax)
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=nx.get_edge_attributes(nx_graph, "label"), ax=ax)
    ax.set_title(f"{explanation.query} → image {explanation.image_id}")
    ax.axis("off")
    fig.tight_layout()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160, bbox_inches="tight")
    return fig

