# QA-Graph trên VG theo nhóm truy vấn — toàn pool test, seed 0, MRR %

10696 truy vấn (1971 direct, 8695 indirect). Δ = (hàng) − CLIP + adapter (không cạnh), CI95 bootstrap ghép cặp theo truy vấn (2000 lần, `paired_bootstrap`).

| Cấu hình | MRR direct | Δ direct [CI95] | MRR indirect | Δ indirect [CI95] |
|---|---|---|---|---|
| CLIP + adapter (không cạnh) | 63.80 | – | 63.06 | – |
| chỉ contains | 71.22 | +7.42 [+6.53, +8.29] | 63.22 | +0.16 [-0.28, +0.64] |
| contains + sg_rel | 70.21 | +6.41 [+5.56, +7.23] | 63.76 | +0.71 [+0.31, +1.12] |
| contains + ConceptNet (SAGE) | 71.10 | +7.30 [+6.37, +8.23] | 63.78 | +0.72 [+0.25, +1.18] |
| contains + ConceptNet, xáo cạnh | 64.03 | +0.23 [-0.37, +0.83] | 62.50 | -0.56 [-0.91, -0.21] |

Cạnh quan hệ (contains + ConceptNet (SAGE) − chỉ contains): tổng +0.43 [+0.16, +0.69], direct -0.12 [-0.69, +0.46], indirect +0.56 [+0.25, +0.85].

Cùng GNN (GAT), Δ = (cấu hình) − chỉ contains:

| Cấu hình (GAT) | Δ tổng [CI95] | Δ direct [CI95] | Δ indirect [CI95] |
|---|---|---|---|
| contains + sg_rel | +0.26 [+0.00, +0.51] | -1.01 [-1.60, -0.42] | +0.54 [+0.24, +0.83] |
| contains + ConceptNet | +0.15 [-0.09, +0.41] | -0.47 [-1.06, +0.17] | +0.30 [+0.03, +0.59] |
| contains + sg_rel + ConceptNet | +0.25 [-0.02, +0.52] | -1.34 [-1.93, -0.77] | +0.60 [+0.29, +0.90] |
