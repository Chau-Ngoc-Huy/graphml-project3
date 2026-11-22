# Direct object XAI report — path-20261121-184857

Protocol: verified_direct; fixed top-100 candidates, frozen QA checkpoints and alpha.
The main explanation is a query-matched object-to-image contains edge. It does not establish the full caption relation. A top-1 image with a different GT ID is an exact-ID mismatch, not automatically a semantic retrieval error.

## Exact-ID benchmark (not semantic relevance)

```
dataset  n_rows  n_queries  exact_id_match_rate
    psg      48         16             0.562500
     vg      54         18             0.648148
```

The 10-case review deliberately samples GT-ID matches and mismatches; its semantic relevance rate cannot be extrapolated to the full query set.

## Coverage and fidelity by dataset, screen and unique-edge budget

```
dataset  screen_path_limit  budget  n_cases  n_queries  positive_object_rate  selected_path_rate  no_explanation_rate  structural_no_path_rate  mean_eligible  mean_screen_coverage  mean_sparsity  matched_cases  unique_random_sets  mean_fidelity_advantage
    psg                 40       1       48         16              1.000000            1.000000             0.000000                 0.000000       7.354167                   1.0       0.823911             48                 279                 0.583110
    psg                 40       2       48         16              1.000000            1.000000             0.000000                 0.000000       7.354167                   1.0       0.688660             48                 353                 1.069860
    psg                 40       3       48         16              1.000000            1.000000             0.000000                 0.000000       7.354167                   1.0       0.664106             45                 337                 1.093735
     vg                 40       1       54         18              0.944444            0.944444             0.055556                 0.055556      16.388889                   1.0       0.934857             51                 585                 0.348234
     vg                 40       2       54         18              0.944444            0.944444             0.055556                 0.055556      16.388889                   1.0       0.886521             51                 846                 0.538252
     vg                 40       3       54         18              0.944444            0.944444             0.055556                 0.055556      16.388889                   1.0       0.856710             51                 865                 0.646362
```

Matched fidelity is conditional on independent contains edges of the same target image and unique-edge budget. The `matched_cases` column gives its denominator. `mean_sparsity` is conditional on an explanation existing; no-path cases remain in yield/coverage denominators. VG and PSG use different graph/model configurations, so their fuse-score magnitudes are not directly compared.

## Query-cluster bootstrap 95% CI

```
dataset  screen_path_limit  budget  n_queries  n_matched_queries  coverage_mean  coverage_ci_low  coverage_ci_high  advantage_mean  advantage_ci_low  advantage_ci_high
    psg                 40       1         16                 16       1.000000         1.000000               1.0        0.583110          0.500324           0.659645
    psg                 40       2         16                 16       1.000000         1.000000               1.0        1.069860          0.818370           1.342423
    psg                 40       3         16                 15       1.000000         1.000000               1.0        1.093735          0.831323           1.388473
     vg                 40       1         18                 17       0.944444         0.833333               1.0        0.348234          0.263848           0.435963
     vg                 40       2         18                 17       0.944444         0.833333               1.0        0.538252          0.404431           0.666962
     vg                 40       3         18                 17       0.944444         0.833333               1.0        0.646362          0.480796           0.818361
```

Top-1 semantic relevance and object/visual validity remain pending independent review. See `semantic_case_review.csv` after the Case review sheet is completed. Structural validity confirms graph continuity only. `no_object_explanation` is a result, not an exclusion.
