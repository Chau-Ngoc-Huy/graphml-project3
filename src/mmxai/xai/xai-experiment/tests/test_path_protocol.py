import unittest

from xai_experiment.graph import Edge, KnowledgeGraph, Node
from xai_experiment.path_protocol import evaluate_case
from xai_experiment.scorer_adapter import CallableScorerAdapter, ScoreComponents


class FakeCase:
    def __init__(self, *, paths=True):
        self.ids = (1, 2)
        self.top_id = self.predicted_top_id = 1
        self.intervention_graph = KnowledgeGraph.from_records(
            [Node('q', 'label', 'person'), Node('a', 'label', 'horse'),
             Node('b', 'label', 'bike'), Node('image:1', 'image', external_id='1')],
            [Edge('1', 'q', 'a', 'concept'), Edge('2', 'a', 'image:1', 'contains'),
             Edge('3', 'q', 'b', 'concept'), Edge('4', 'b', 'image:1', 'contains'),
             Edge('5', 'q', 'image:1', 'contains')],
        )
        self.path_candidates = (((1, 2), (0, 1, 3)), ((3, 4), (0, 2, 3))) if paths else ()
        self.all_path_count = len(self.path_candidates)
        self.eligible_edge_ids = tuple(range(1, 6))
        self.eligible_type_counts = {'concept': 2, 'contains': 3, 'sg_rel': 0}
        self.screened = self.eligible_edge_ids
        self.screened_type_counts = dict(self.eligible_type_counts)

        def score_fn(query, ids, graph):
            value = sum({1: .3, 2: .4, 3: .1, 4: .2, 5: .05}[int(e)] for e in graph.edges)
            return {'1': ScoreComponents(.6, value, .6 + value),
                    '2': ScoreComponents(.5, .1, .6)}

        self.scorer = CallableScorerAdapter(score_fn, lambda q, i, g: tuple(g.edges))

    def score_removed(self, removed=()):
        value = sum({1: .3, 2: .4, 3: .1, 4: .2, 5: .05}[i]
                    for i in range(1, 6) if i not in removed)
        return {'qa': [value + .6, .6]}


class PathProtocolTests(unittest.TestCase):
    def test_direct_object_path_uses_same_target_contains_control(self):
        case = FakeCase()
        case.path_candidates = (((5,), (0, 3)),)
        case.all_path_count = 1
        rows, paths = evaluate_case(case, 'person on horse', '1',
                                    dataset='vg', model_seed=0, query_id='q0', group='direct',
                                    budgets=(1,), random_repeats=5, random_seed=7,
                                    explanation_mode='object')
        selected = next(r for r in rows if r['intervention'] == 'object_path')
        self.assertEqual(selected['status'], 'ok')
        self.assertEqual(selected['n_removed_unique'], 1)
        self.assertEqual(selected['n_removed_relation'], 0)
        self.assertEqual(selected['n_eligible_explanation'], 3)
        self.assertAlmostEqual(selected['sparsity'], 1 - 1 / 3)
        self.assertEqual(selected['random_n'], 5)
        self.assertEqual(selected['explanation_mode'], 'object')
        self.assertEqual(paths[0]['selected_at_budgets'], [1])

    def test_deletes_whole_path_and_matches_random_signature(self):
        rows, paths = evaluate_case(FakeCase(), 'person on horse', '1',
                                    dataset='vg', model_seed=0, query_id='q1', group='direct',
                                    budgets=(2,), random_repeats=5, random_seed=7)
        selected = next(r for r in rows if r['intervention'] == 'relational_path')
        self.assertEqual(selected['n_removed_unique'], 2)
        self.assertAlmostEqual(selected['final_score_drop'], .7)
        self.assertAlmostEqual(selected['sparsity'], 1 - 2 / 5)
        self.assertEqual(selected['random_n'], 5)
        self.assertEqual(selected['target_rank_before'], 1)
        self.assertEqual(selected['gt_rank_before'], 1)
        self.assertTrue(any(p['selected_at_budgets'] == [2] for p in paths))

    def test_no_path_and_gt_outside_pool_remain_explicit(self):
        rows, paths = evaluate_case(FakeCase(paths=False), 'person on horse', '999',
                                    dataset='vg', model_seed=0, query_id='q2', group='direct',
                                    budgets=(2,), random_repeats=2)
        selected = next(r for r in rows if r['intervention'] == 'relational_path')
        self.assertEqual(selected['status'], 'no_relational_path')
        self.assertIsNone(selected['gt_rank_before'])
        self.assertEqual(selected['n_removed_unique'], 0)
        self.assertIsNone(selected['sparsity'])
        self.assertEqual(paths[0]['path_id'], 'no_path')
