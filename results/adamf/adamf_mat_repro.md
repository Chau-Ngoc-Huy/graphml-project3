# AdaMF-MAT reproduction results (Kaggle, 2x T4, run finished)

Source: results/extracted/report.txt (+ MKG-W-full.log, MKG-Y-full.log). Notebook: src/adamf_mat_kaggle.ipynb.
Kaggle run to cite in the report (version 6, the MKG-W / MKG-Y reproduction): https://www.kaggle.com/code/khnhnguyninhquang/adamf-mat-kaggle?scriptVersionId=369425081
Setup: authors' scripts (dim 200, batch 1024, neg 128, lr 2e-5, 1000 epochs, seed 42, margin 12 for MKG-W / 4 for MKG-Y). Single seed. Filtered link prediction, averaged over head and tail. torch 2.11 on Python 3.13 (repo's pinned torch 1.9.1 not needed).

| Dataset | MRR | MR | Hits@1 | Hits@3 | Hits@10 | Train time |
|---|---|---|---|---|---|---|
| MKG-W | 0.3568 | 600.0 | 0.2898 | 0.3868 | 0.4813 | 3h32m (12.7 s/epoch) |
| MKG-Y | 0.3910 | 1106.1 | 0.3459 | 0.4123 | 0.4668 | 2h14m (8.1 s/epoch) |

Per direction (filtered):
- MKG-W: head-pred MRR 0.1856 / Hits@10 0.2857; tail-pred MRR 0.5280 / Hits@10 0.6769
- MKG-Y: head-pred MRR 0.3948 / Hits@10 0.4765; tail-pred MRR 0.3873 / Hits@10 0.4570

Smoke test (3 epochs, MKG-W): MRR 0.0007 (untrained, as expected). Full training reached MRR 0.357.

## Comparison with the paper (arXiv 2402.15444, Table 2; papers/2402.15444v1.pdf), in %
| Dataset | | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| MKG-W | paper | 35.85 | 29.04 | 39.01 | 48.42 |
| MKG-W | ours | 35.68 | 28.98 | 38.68 | 48.13 |
| MKG-W | diff | -0.17 | -0.06 | -0.33 | -0.29 |
| MKG-Y | paper | 38.57 | 34.34 | 40.59 | 45.76 |
| MKG-Y | ours | 39.10 | 34.59 | 41.23 | 46.68 |
| MKG-Y | diff | +0.53 | +0.25 | +0.64 | +0.92 |
(DB15K paper: 35.14 / 25.30 / 41.11 / 52.92; not run by us.)
All differences are under 1 point, so the reproduction matches the paper on both datasets.

NOT done / not verified
- One seed only (42); the spec asks for >= 3 seeds on main experiments. Paper does not state seed count; small gaps could be seed noise.
- Plain KG completion on MKG-W/MKG-Y; not yet connected to the Visual Genome retrieval pipeline (design plan A/B in DESIGN_TWO_LEVEL.md).
