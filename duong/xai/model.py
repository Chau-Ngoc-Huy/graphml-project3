"""Adapter from Huy's trained QA scorer to the graph intervention XAI API.

The intervention graph contains the screened local edges. Edges outside that
screen remain in Huy's original graph during every forward pass. This keeps
the comparison faithful to the trained model while making each intervention
cheap enough to run on Colab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graph import Edge, GraphSchema, KnowledgeGraph, Node, RelationRule
from .scorer_adapter import CallableScorerAdapter, ScoreComponents


@dataclass(frozen=True)
class ModelSpec:
    name: str
    experiment: str
    bundle_dir: str
    graph_variant: str
    gnn_kind: str
    expected_labels: int
    expected_concept_edges: int


MODEL_SPECS = {
    "vg": ModelSpec("vg", "t1_qa_cn_no_sg_sage", "mmxai_vg_bundle", "no_sg", "sage", 3000, 32826),
    "psg": ModelSpec("psg", "psg_qa_cn", "mmxai_psg_bundle", "full", "gat", 807, 7486),
}


class HuyQACaseModel:
    """One fixed query and top-100 pool, with fresh graph scoring after deletion."""

    def __init__(self, model, graph, qd, bundle, protocol, candidates, eval_module,
                 candidate_row: int, alpha: float, device, max_rel_edges: int = 40,
                 image_tensor=None):
        self.model, self.graph, self.qd, self.bundle = model, graph, qd, bundle
        self.protocol, self.candidates, self.ev = protocol, candidates, eval_module
        self.j, self.alpha, self.device = int(candidate_row), float(alpha), device
        self.cap_id = int(candidates["q_cap"][self.j])
        self.ids = np.asarray(candidates["cand"][self.j], dtype=np.int64)
        self.row_of = {int(iid): row for row, iid in enumerate(bundle.image_ids)}
        self.rows_np = np.asarray([self.row_of[int(i)] for i in self.ids], dtype=np.int64)
        self.rows = torch.as_tensor(self.rows_np, device=device)
        self.cc = qd.cap_concepts[self.cap_id:self.cap_id + 1]
        self.query = torch.as_tensor(protocol.cap[self.cap_id:self.cap_id + 1], device=device)
        self.clip = np.asarray(candidates["sim"][self.j], dtype=np.float64)
        self.ei, self.et = graph.data.edge_index, graph.data.edge_type
        self.rel_global = torch.nonzero((self.ei[0] >= graph.n_img) &
                                        (self.ei[1] >= graph.n_img)).flatten().tolist()
        self.rel_position = {int(e): p for p, e in enumerate(self.rel_global)}
        self.x_img = image_tensor if image_tensor is not None else torch.from_numpy(bundle.image_emb).to(device)
        model.eval()
        with torch.no_grad():
            self.base_h = model.label_h(qd.x_lab, qd.ei_lab)
            self.adapter_t = model.adapter_cos(self.query, self.x_img[self.rows])
        self._cache = {}
        self.local_all, self.screened = self._screen_edges(max_rel_edges)
        self._screened_set = set(self.screened)
        self.intervention_graph, self.schema = self._make_intervention_graph()
        self.scorer = CallableScorerAdapter(self._score_fn, self._local_edges_fn)

    @torch.no_grad()
    def score_removed(self, removed=()):
        key = tuple(sorted({int(e) for e in removed}))
        if key in self._cache:
            return self._cache[key]
        relation = [e for e in key if e in self.rel_position]
        contains = [e for e in key if int(self.et[e]) == 0]
        if relation:
            keep = torch.ones(self.qd.ei_lab.shape[1], dtype=torch.bool, device=self.device)
            keep[[self.rel_position[e] for e in relation]] = False
            h = self.model.label_h(self.qd.x_lab, self.qd.ei_lab[:, keep])
        else:
            h = self.base_h
        labels = self.qd.img_labels[self.rows].clone()
        for e in contains:
            label = int(self.ei[0, e]) - self.graph.n_img
            image_row = int(self.ei[1, e])
            candidate_pos = np.flatnonzero(self.rows_np == image_row)
            if len(candidate_pos):
                pos = int(candidate_pos[0])
                labels[pos, labels[pos] == label] = -1
        graph_t = self.model.graph_matrix(h, self.cc, labels)
        qa_t = self.model.total(self.adapter_t, graph_t)
        graph_scores = graph_t.cpu().numpy()[0]
        qa_scores = qa_t.cpu().numpy()[0]
        final_scores = self.ev.fuse(self.clip[None], qa_scores[None], self.alpha)[0]
        scores = {"clip": self.clip, "graph": graph_scores, "qa": qa_scores, "final": final_scores}
        self._cache[key] = scores
        return scores

    def _screen_edges(self, max_rel_edges: int):
        baseline = self.score_removed()
        top_pos = int(np.argsort(-baseline["final"], kind="stable")[0])
        self.top_id = int(self.ids[top_pos])
        target_row = int(self.rows_np[top_pos])
        ei, et, n = self.ei.cpu().numpy(), self.et.cpu().numpy(), self.graph.n_img
        contains = np.flatnonzero((et == 0) & (ei[1] == target_row)).tolist()
        start = {n + int(v) for v in self.cc[0].tolist() if v >= 0}
        start |= {n + int(v) for v in self.qd.img_labels[target_row].tolist() if v >= 0}
        incoming = {}
        for e in self.rel_global:
            incoming.setdefault(int(ei[1, e]), []).append(int(e))
        found, frontier = {}, start
        for depth in (1, 2):
            next_frontier = set()
            for node in frontier:
                for e in incoming.get(node, ()):
                    found.setdefault(e, depth)
                    next_frontier.add(int(ei[0, e]))
            frontier = next_frontier
        relation_ids = list(found)
        h = self.base_h.cpu().numpy()
        src = np.array([int(ei[0, e]) - n for e in relation_ids], dtype=np.int64)
        dst = np.array([int(ei[1, e]) - n for e in relation_ids], dtype=np.int64)
        query_labels = self.cc[0][self.cc[0] >= 0].cpu().numpy()
        image_labels = self.qd.img_labels[target_row]
        image_labels = image_labels[image_labels >= 0].cpu().numpy()

        def similarities(label_ids):
            if len(label_ids) == 0 or len(relation_ids) == 0:
                return np.zeros(len(relation_ids)), np.zeros(len(relation_ids))
            return (h[src] @ h[label_ids].T).max(1), (h[dst] @ h[label_ids].T).max(1)

        sq, dq = similarities(query_labels)
        si, di = similarities(image_labels)
        relevance = np.maximum(sq + di, dq + si)
        priority = {e: float(v) for e, v in zip(relation_ids, relevance)}
        relations = sorted(relation_ids, key=lambda e: (found[e], -priority[e], e))
        return tuple(contains + relations), tuple(contains + relations[:max_rel_edges])

    def _make_intervention_graph(self):
        nodes = {f"image:{self.top_id}": Node(f"image:{self.top_id}", "image",
                                               external_id=str(self.top_id))}
        edges = []
        n = self.graph.n_img
        for edge_id in self.screened:
            src, dst = self.ei[:, edge_id].tolist()
            kind = {0: "contains", 1: "sg_rel", 2: "concept"}[int(self.et[edge_id])]
            for node in (src, dst):
                if node >= n:
                    node_id = f"label:{node - n}"
                    nodes.setdefault(node_id, Node(node_id, "label", self.graph.labels[node - n]))
            source = f"label:{src - n}"
            target = f"image:{int(self.bundle.image_ids[dst])}" if kind == "contains" else f"label:{dst - n}"
            edges.append(Edge(str(edge_id), source, target, kind,
                              provenance={"contains": "VG/PSG objects", "sg_rel": "PSG train scene graph",
                                          "concept": "ConceptNet"}[kind],
                              metadata={"edge_type": kind,
                                        "pair_relation_hint": self.graph.rel_name.get((src, dst), kind),
                                        "predicate_status": "display_only_not_model_input",
                                        "message_direction": "source_to_target",
                                        "source_split": (self.protocol.images[dst].get("split", "unknown")
                                                         if kind == "contains" else
                                                         "train" if kind == "sg_rel" else "external")}))
        graph = KnowledgeGraph.from_records(nodes.values(), edges)
        schema = GraphSchema(frozenset({"image", "label"}), {
            "contains": RelationRule.make({"label"}, {"image"}),
            "sg_rel": RelationRule.make({"label"}, {"label"}),
            "concept": RelationRule.make({"label"}, {"label"}),
        })
        return graph, schema

    def _score_fn(self, query, candidate_ids, local_graph):
        if tuple(map(int, candidate_ids)) != tuple(self.ids):
            raise ValueError("candidate pool/order changed during perturbation")
        removed = self._screened_set - {int(e) for e in local_graph.edges}
        values = self.score_removed(removed)
        return {str(iid): ScoreComponents(float(values["clip"][p]), float(values["graph"][p]),
                                          float(values["final"][p]))
                for p, iid in enumerate(self.ids)}

    def _local_edges_fn(self, query, image_id, local_graph):
        if int(image_id) != self.top_id:
            raise ValueError("this case model explains its original top image")
        return tuple(str(e) for e in self.screened if str(e) in local_graph.edges)


class HuyQAPathCaseModel(HuyQACaseModel):
    """A target-specific path screen with an exact two-layer eligibility audit.

    The screen uses graph topology and a fixed path limit only. It never uses
    perturbation scores or test relevance. The original scorer still retains
    every edge outside the intervention graph.
    """

    def __init__(self, *args, target_image_id=None, max_paths=200, **kwargs):
        self.target_image_id = target_image_id
        self.max_paths = int(max_paths)
        if self.max_paths < 1:
            raise ValueError("max_paths must be positive")
        super().__init__(*args, **kwargs)

    def _screen_edges(self, max_rel_edges):
        base = self.score_removed()
        self.predicted_top_id = int(self.ids[int(np.argsort(-base["final"], kind="stable")[0])])
        self.top_id = int(self.target_image_id) if self.target_image_id is not None else self.predicted_top_id
        if self.top_id not in set(map(int, self.ids)):
            raise ValueError("target image must be in the fixed candidate pool")
        target_row = self.row_of[self.top_id]
        ei, et, n = self.ei.cpu().numpy(), self.et.cpu().numpy(), self.graph.n_img
        contains = tuple(int(e) for e in np.flatnonzero((et == 0) & (ei[1] == target_row)))
        image_labels = {int(ei[0, e]) for e in contains}
        query_labels = {n + int(v) for v in self.cc[0].tolist() if v >= 0}
        incoming, outgoing = {}, {}
        for e in self.rel_global:
            a, b = int(ei[0, e]), int(ei[1, e])
            incoming.setdefault(b, []).append(int(e))
            outgoing.setdefault(a, []).append(int(e))
        # Each endpoint embedding H depends on incoming messages at most two hops away.
        frontier, eligible_rel = query_labels | image_labels, set()
        for _ in range(len(self.model.convs)):
            frontier_next = set()
            for node in frontier:
                for e in incoming.get(node, ()):
                    eligible_rel.add(e)
                    frontier_next.add(int(ei[0, e]))
            frontier = frontier_next
        self.eligible_edge_ids = tuple(contains) + tuple(sorted(eligible_rel))
        self.eligible_type_counts = {
            kind: sum(int(et[e]) == code for e in self.eligible_edge_ids)
            for kind, code in (("contains", 0), ("sg_rel", 1), ("concept", 2))
        }
        contains_by_label = {}
        for e in contains:
            contains_by_label.setdefault(int(ei[0, e]), []).append(e)
        candidates = []
        for q in sorted(query_labels):
            for c in contains_by_label.get(q, ()):
                candidates.append(((c,), (q, target_row)))
            for e1 in outgoing.get(q, ()):
                v = int(ei[1, e1])
                if e1 not in eligible_rel or v == q:
                    continue
                for c in contains_by_label.get(v, ()):
                    candidates.append(((e1, c), (q, v, target_row)))
                for e2 in outgoing.get(v, ()):
                    w = int(ei[1, e2])
                    if e2 not in eligible_rel or w in (q, v):
                        continue
                    for c in contains_by_label.get(w, ()):
                        candidates.append(((e1, e2, c), (q, v, w, target_row)))
        # Score relational paths first. Contains-only paths are covered by their
        # own control and must not consume the relational screen budget.
        candidates = sorted(set(candidates), key=lambda p: (len(p[0]) == 1, len(p[0]), p[0]))
        self.all_path_count = len(candidates)
        self.all_relational_path_count = sum(len(edges) > 1 for edges, _ in candidates)
        self.path_candidates = tuple(candidates[:self.max_paths])
        screened = set(contains)
        for edge_ids, _ in self.path_candidates:
            screened.update(edge_ids)
        self.screened_type_counts = {
            kind: sum(int(et[e]) == code for e in screened)
            for kind, code in (("contains", 0), ("sg_rel", 1), ("concept", 2))
        }
        return self.eligible_edge_ids, tuple(sorted(screened))
