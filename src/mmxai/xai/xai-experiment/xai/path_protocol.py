"""Predeclared, target-specific path perturbation protocol for the QA scorer.

All score comparisons use the same candidate list and the same frozen model.
Predicate names are display metadata: the current scorer consumes edge_index only.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Any

from .perturbation import rank_of, score_without_edges


def _edge_kind(graph, edge_id: str) -> str:
    return graph.edges[edge_id].relation


def _path_record(case, edge_numbers, node_numbers, query_id: str) -> dict[str, Any]:
    graph = case.intervention_graph
    edge_ids = tuple(map(str, edge_numbers))
    edges = [graph.edges[e] for e in edge_ids]
    node_ids = tuple([edges[0].source] + [edge.target for edge in edges])
    if len(node_ids) != len(edge_ids) + 1 or node_ids[-1] != graph.image_node_id(str(case.top_id)):
        raise ValueError("candidate path is not directed to the target image")
    if any(left != edge.source or right != edge.target
           for left, right, edge in zip(node_ids, node_ids[1:], edges)):
        raise ValueError("candidate path is discontinuous")
    if len(edge_ids) not in (1, 2, 3) or edges[-1].relation != "contains":
        raise ValueError("candidate path violates length or terminal edge rule")
    if any(edge.relation == "contains" for edge in edges[:-1]):
        raise ValueError("nonterminal contains edge")
    if any(graph.nodes[node].type != 'label' for node in node_ids[:-1]) or graph.nodes[node_ids[-1]].type != 'image':
        raise ValueError("candidate path violates node types")
    return {
        "query_id": str(query_id), "target_image_id": str(case.top_id),
        "path_id": "|".join(edge_ids), "edge_ids": list(edge_ids), "node_ids": list(node_ids),
        "length": len(edge_ids), "level": "object" if len(edge_ids) == 1 else "relational_connectivity",
        "edge_types": [edge.relation for edge in edges],
        "predicates": [edge.metadata.get("pair_relation_hint", "") for edge in edges],
        "sources": [edge.provenance for edge in edges],
        "source_splits": [edge.metadata.get("source_split", "unknown") for edge in edges],
        "edge_details": [{"id": edge.id, "source": edge.source, "target": edge.target,
                          "type": edge.relation,
                          "predicate_hint": edge.metadata.get("pair_relation_hint", ""),
                          "provenance": edge.provenance,
                          "split": edge.metadata.get("source_split", "unknown")}
                         for edge in edges],
        "structural_validity": True,
        "inference_direction": "graph_edge_direction; predicate_inverse_unverified",
        "query_concept": graph.nodes[node_ids[0]].label,
        "terminal_image_label": graph.nodes[node_ids[-2]].label,
    }


def evaluate_case(case, query: str, ground_truth_image_id: str, *, dataset: str,
                  model_seed: int, query_id: str, group: str, budgets=(2, 3, 5),
                  random_repeats: int = 20, random_seed: int = 0,
                  hard_negative_image_id: str | None = None,
                  explanation_mode: str = "relational") -> tuple[list[dict], list[dict]]:
    """Score every screened candidate path, select positive paths and run controls.

    Returns flat result rows and JSONL-ready path rows. A missing relational
    path is an explicit result. Random controls are N/A when the eligible path
    pool cannot supply an independent set with identical path signatures and
    unique-edge budget.
    """
    if random_repeats < 1:
        raise ValueError("random_repeats must be positive")
    if explanation_mode not in ("relational", "object"):
        raise ValueError("explanation_mode must be relational or object")
    graph, scorer = case.intervention_graph, case.scorer
    ids = tuple(map(str, case.ids))
    target, gt = str(case.top_id), str(ground_truth_image_id)
    negative = str(hard_negative_image_id) if hard_negative_image_id is not None else None
    if gt not in ids:
        gt_rank_before = None
    else:
        gt_rank_before = rank_of(scorer.score(query, ids, graph), gt, ids)
    before = scorer.score(query, ids, graph)
    base_qa = case.score_removed()["qa"][list(ids).index(target)]
    target_rank_before = rank_of(before, target, ids)
    raw_paths = [_path_record(case, e, n, query_id) for e, n in case.path_candidates]
    scored_paths = []
    for path in raw_paths:
        removed = tuple(path["edge_ids"])
        after = score_without_edges(scorer, query, ids, graph, removed).scores
        qa_after = case.score_removed(tuple(map(int, removed)))["qa"][list(ids).index(target)]
        path.update({
            "graph_score_drop": before[target].graph_score - after[target].graph_score,
            "qa_score_drop": float(base_qa - qa_after),
            "final_score_drop": before[target].final_score - after[target].final_score,
            "target_rank_after": rank_of(after, target, ids),
            "selected_at_budgets": [], "selection_reason": "not_selected",
        })
        scored_paths.append(path)
    relational = [p for p in scored_paths if p["level"] == "relational_connectivity"]
    positive_relational = [p for p in relational if p["final_score_drop"] > 0]
    object_paths = [p for p in scored_paths if p["level"] == "object"]
    candidate_paths = relational if explanation_mode == "relational" else object_paths
    positive_paths = [p for p in candidate_paths if p["final_score_drop"] > 0]
    contains = [e for e in graph.edges if _edge_kind(graph, e) == "contains"]
    rng = random.Random(random_seed)

    def measure(removed, intervention, budget, *, random_mean=None, random_n=0,
                random_unique_n=0,
                selected_paths=(), status="ok"):
        removed = tuple(dict.fromkeys(removed))
        after = score_without_edges(scorer, query, ids, graph, removed).scores
        qa_after = case.score_removed(tuple(map(int, removed)))["qa"][list(ids).index(target)]
        kinds = Counter(_edge_kind(graph, e) for e in removed)
        eligible = case.eligible_type_counts
        e_total = len(case.eligible_edge_ids)
        e_rel = eligible.get("concept", 0) + eligible.get("sg_rel", 0)
        e_explanation = eligible.get("contains", 0) if explanation_mode == "object" else e_total
        screened_explanation = (case.screened_type_counts.get("contains", 0)
                                if explanation_mode == "object" else len(case.screened))
        n_rel = kinds.get("concept", 0) + kinds.get("sg_rel", 0)
        return {
            "dataset": dataset, "query_id": str(query_id), "query": query,
            "group_source": group, "model_seed": model_seed,
            "explanation_mode": explanation_mode,
            "target_image_id": target,
            "target_role": ("top1_and_ground_truth" if target == gt else
                            "top1_and_verified_hard_negative" if target == negative else
                            "predicted_top1" if target == str(case.predicted_top_id) else
                            "other_candidate"),
            "ground_truth_image_id": gt, "ground_truth_in_pool": gt in ids,
            "hard_negative_image_id": negative,
            "candidate_pool_size": len(ids), "budget": budget, "intervention": intervention,
            "status": status, "n_candidate_paths": case.all_path_count,
            "n_candidate_relational_paths_full": getattr(case, "all_relational_path_count", len(relational)),
            "n_screened_paths": len(case.path_candidates),
            "n_screened_out_paths": case.all_path_count - len(case.path_candidates),
            "n_candidate_relational_paths": len(relational),
            "has_positive_relational_path": bool(positive_relational),
            "has_positive_object_path": any(p["final_score_drop"] > 0 for p in object_paths),
            "n_eligible": e_total, "n_eligible_relation": e_rel,
            "n_eligible_explanation": e_explanation,
            "n_screened": len(case.screened),
            "n_eligible_contains": eligible.get("contains", 0),
            "n_eligible_concept": eligible.get("concept", 0),
            "n_eligible_sg_rel": eligible.get("sg_rel", 0),
            "n_screened_contains": case.screened_type_counts.get("contains", 0),
            "n_screened_concept": case.screened_type_counts.get("concept", 0),
            "n_screened_sg_rel": case.screened_type_counts.get("sg_rel", 0),
            "n_removed_unique": len(removed), "n_removed_relation": n_rel,
            "n_removed_contains": kinds.get("contains", 0),
            "n_removed_concept": kinds.get("concept", 0),
            "n_removed_sg_rel": kinds.get("sg_rel", 0),
            "n_paths": len(selected_paths), "selected_path_ids": "|".join(p["path_id"] for p in selected_paths),
            "structural_validity": all(p["structural_validity"] for p in selected_paths) if selected_paths else None,
            "removed_edge_ids": "|".join(removed),
            "sparsity": (1 - len(removed) / e_explanation
                         if e_explanation and removed and
                         (explanation_mode == "relational" or n_rel == 0) else None),
            "relation_selected_fraction": n_rel / e_rel if e_rel else None,
            "screen_coverage": screened_explanation / e_explanation if e_explanation else None,
            "graph_score_before": before[target].graph_score,
            "graph_score_after": after[target].graph_score,
            "graph_score_drop": before[target].graph_score - after[target].graph_score,
            "qa_score_before": float(base_qa), "qa_score_after": float(qa_after),
            "qa_score_drop": float(base_qa - qa_after),
            "final_score_before": before[target].final_score,
            "final_score_after": after[target].final_score,
            "final_score_drop": before[target].final_score - after[target].final_score,
            "target_rank_before": target_rank_before,
            "target_rank_after": rank_of(after, target, ids),
            "gt_rank_before": gt_rank_before,
            "gt_rank_after": rank_of(after, gt, ids) if gt in ids else None,
            "hard_negative_rank_before": rank_of(before, negative, ids) if negative in ids else None,
            "hard_negative_rank_after": rank_of(after, negative, ids) if negative in ids else None,
            "random_final_drop_mean": random_mean, "random_n": random_n,
            "random_unique_n": random_unique_n,
            "random_seed": random_seed,
        }

    rows = []
    for budget in budgets:
        if budget < 1:
            raise ValueError("budgets must be positive")
        selected, selected_edges = [], set()
        while True:
            options = []
            for path in positive_paths:
                extra = set(path["edge_ids"]) - selected_edges
                if not extra or len(selected_edges) + len(extra) > budget:
                    continue
                union = selected_edges | set(path["edge_ids"])
                score = score_without_edges(scorer, query, ids, graph, union).scores[target].final_score
                current = score_without_edges(scorer, query, ids, graph, selected_edges).scores[target].final_score
                marginal = current - score
                if marginal > 0:
                    options.append((marginal / len(extra), marginal, path["path_id"], path, union))
            if not options:
                break
            _, _, _, chosen, selected_edges = max(options, key=lambda v: (v[0], v[1], v[2]))
            selected.append(chosen)
        for path in selected:
            path["selected_at_budgets"].append(budget)
            path["selection_reason"] = "positive_marginal_path"
        if selected:
            status = "ok"
        elif not candidate_paths:
            status = f"no_{explanation_mode}_path"
        elif not positive_paths:
            status = f"no_positive_{explanation_mode}_path"
        elif not any(len(p['edge_ids']) <= budget for p in positive_paths):
            status = "budget_too_small"
        else:
            status = f"no_{explanation_mode}_explanation"
        matched = []
        if selected:
            signatures = [tuple(p["edge_types"]) for p in selected]
            random_pool = (candidate_paths if explanation_mode == "relational" else
                           [{"path_id": str(e), "edge_ids": [str(e)], "edge_types": ["contains"]}
                            for e in contains])
            alternatives = [[p for p in random_pool if tuple(p["edge_types"]) == sig
                             and p["path_id"] not in {s["path_id"] for s in selected}]
                            for sig in signatures]
            if all(alternatives):
                attempts = 0
                while len(matched) < random_repeats and attempts < random_repeats * 100:
                    attempts += 1
                    draw = [rng.choice(pool) for pool in alternatives]
                    edge_set = {e for p in draw for e in p["edge_ids"]}
                    if len(edge_set) == len(selected_edges) and edge_set != selected_edges:
                        matched.append(tuple(sorted(edge_set)))
        random_drops = []
        for edge_set in matched:
            after = score_without_edges(scorer, query, ids, graph, edge_set).scores
            random_drops.append(before[target].final_score - after[target].final_score)
        random_mean = statistics.fmean(random_drops) if random_drops else None
        rows.append(measure(sorted(selected_edges),
                            "relational_path" if explanation_mode == "relational" else "object_path", budget,
                            random_mean=random_mean, random_n=len(random_drops),
                            random_unique_n=len(set(matched)),
                            selected_paths=selected, status=status))
        if matched:
            for idx, edge_set in enumerate(matched):
                row = measure(edge_set, "random_path", budget, status="matched")
                row["random_repeat"] = idx
                rows.append(row)
        elif selected:
            rows[-1]["random_status"] = "no_independent_matched_path"
        if explanation_mode == "relational":
            # Contains-only control uses the same number of unique edges when possible.
            if len(contains) >= len(selected_edges) and selected_edges:
                ranked_contains = sorted(contains, key=lambda e: (
                    -(before[target].final_score - score_without_edges(
                        scorer, query, ids, graph, (e,)).scores[target].final_score), e))
                rows.append(measure(ranked_contains[:len(selected_edges)], "contains_only", budget))
            else:
                rows.append(measure((), "contains_only", budget, status="budget_unavailable"))
    ablation_kinds = ("contains",) if explanation_mode == "object" else ("contains", "concept", "sg_rel")
    for kind in ablation_kinds:
        edges = [e for e in graph.edges if _edge_kind(graph, e) == kind]
        rows.append(measure(edges, f"ablate_{kind}", len(edges), status="ok" if edges else "no_edges"))
    if not raw_paths:
        scored_paths.append({"query_id": str(query_id), "target_image_id": target,
                             "path_id": "no_path", "edge_ids": [], "node_ids": [],
                             "selection_reason": "no_path_in_screened_graph"})
    elif explanation_mode == "relational" and not relational:
        scored_paths.append({"query_id": str(query_id), "target_image_id": target,
                             "path_id": "no_relational_path", "edge_ids": [], "node_ids": [],
                             "selection_reason": "no_relational_path_in_screened_graph"})
    elif explanation_mode == "object" and not object_paths:
        scored_paths.append({"query_id": str(query_id), "target_image_id": target,
                             "path_id": "no_object_path", "edge_ids": [], "node_ids": [],
                             "selection_reason": "no_object_path_in_screened_graph"})
    if case.all_path_count > len(case.path_candidates):
        scored_paths.append({"query_id": str(query_id), "target_image_id": target,
                             "path_id": "screen_budget_excluded", "edge_ids": [], "node_ids": [],
                             "n_excluded": case.all_path_count - len(case.path_candidates),
                             "selection_reason": "deterministic_path_screen_limit"})
    return rows, scored_paths
