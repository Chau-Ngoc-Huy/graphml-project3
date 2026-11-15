# Baseline CLIP thuần — VG ∩ COCO (split seed 0, gallery 1000, K=100)

**Mean ± std trên 3 fold gallery theo seed [0, 1, 2]**

| phương pháp | split | n truy vấn | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B-32 (laion2b) | test | 5001 | 57.36 ± 0.95 | 84.08 ± 0.24 | 91.76 ± 0.33 | 69.23 ± 0.68 | 74.32 ± 0.55 |
| CLIP ViT-B-32 (laion2b) | val | 5001 | 58.17 ± 0.40 | 84.74 ± 0.68 | 92.29 ± 0.41 | 69.92 ± 0.40 | 75.00 ± 0.42 |
| CLIP ViT-B-32 (laion2b) (image→text) | test | 1000 | 74.10 ± 0.96 | 91.90 ± 0.54 | 96.53 ± 0.21 | 81.82 ± 0.84 | 85.24 ± 0.70 |
| CLIP ViT-B-32 (laion2b) (image→text) | val | 1000 | 73.73 ± 1.87 | 92.97 ± 0.47 | 97.40 ± 0.43 | 82.03 ± 1.35 | 85.65 ± 1.13 |

**Toàn bộ pool (tất định)**

| phương pháp | split | n truy vấn | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B-32 (laion2b) | test | 10696 | 48.31 | 75.35 | 84.08 | 60.40 | 65.51 |
| CLIP ViT-B-32 (laion2b) | val | 10633 | 48.55 | 75.39 | 84.36 | 60.61 | 65.73 |
| CLIP ViT-B-32 (laion2b) (image→text) | test | 2138 | 65.53 | 86.01 | 91.44 | 74.65 | 78.38 |
| CLIP ViT-B-32 (laion2b) (image→text) | val | 2126 | 65.19 | 87.30 | 92.99 | 74.95 | 79.03 |

**Text → image theo nhóm truy vấn (mean ± std theo seed)**

| phương pháp | split | n truy vấn | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B-32 (laion2b) — direct | test | 933 | 56.93 ± 0.79 | 84.69 ± 0.83 | 93.15 ± 0.40 | 69.33 ± 0.89 | 74.78 ± 0.79 |
| CLIP ViT-B-32 (laion2b) — indirect | test | 4054 | 57.48 ± 1.02 | 83.99 ± 0.11 | 91.50 ± 0.38 | 69.24 ± 0.65 | 74.25 ± 0.52 |
| CLIP ViT-B-32 (laion2b) — no_concept | test | 13 | 51.35 ± 7.36 | 71.88 ± 9.64 | 73.84 ± 7.03 | 60.87 ± 8.27 | 64.08 ± 7.98 |
| CLIP ViT-B-32 (laion2b) — direct | val | 1003 | 63.70 ± 0.79 | 87.69 ± 1.18 | 93.39 ± 0.25 | 74.05 ± 0.71 | 78.43 ± 0.51 |
| CLIP ViT-B-32 (laion2b) — indirect | val | 3989 | 56.80 ± 0.68 | 84.06 ± 0.60 | 92.08 ± 0.51 | 68.93 ± 0.56 | 74.18 ± 0.53 |
| CLIP ViT-B-32 (laion2b) — no_concept | val | 8 | 47.20 ± 7.26 | 54.23 ± 3.06 | 61.27 ± 3.99 | 51.87 ± 4.28 | 54.06 ± 3.07 |