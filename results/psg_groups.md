# QA-Graph trên PSG theo nhóm truy vấn — toàn pool test, seed 0, MRR %

10933 truy vấn (1199 direct, 9645 indirect). Δ = (hàng) − CLIP + adapter (không cạnh), CI95 bootstrap ghép cặp theo truy vấn (2000 lần, `paired_bootstrap`).

| Cấu hình | MRR direct | Δ direct [CI95] | MRR indirect | Δ indirect [CI95] |
|---|---|---|---|---|
| CLIP + adapter (không cạnh) | 65.21 | – | 62.49 | – |
| chỉ contains | 68.27 | +3.06 [+2.03, +4.07] | 63.92 | +1.42 [+1.04, +1.81] |
| contains + sg_rel | 67.72 | +2.51 [+1.67, +3.33] | 63.97 | +1.48 [+1.16, +1.80] |
| contains + sg_rel + ConceptNet | 68.01 | +2.80 [+1.82, +3.80] | 63.92 | +1.42 [+1.05, +1.79] |
| contains + sg_rel + ConceptNet, xáo cạnh | 64.95 | -0.27 [-1.05, +0.52] | 61.74 | -0.75 [-1.07, -0.45] |

Cạnh quan hệ (contains + sg_rel + ConceptNet − chỉ contains): tổng -0.03 [-0.31, +0.23], direct -0.26 [-1.10, +0.52], indirect -0.00 [-0.29, +0.29].

Cùng GNN (GAT), Δ = (cấu hình) − chỉ contains:

| Cấu hình (GAT) | Δ tổng [CI95] | Δ direct [CI95] | Δ indirect [CI95] |
|---|---|---|---|
| contains + sg_rel | -0.01 [-0.29, +0.29] | -0.55 [-1.45, +0.31] | +0.06 [-0.25, +0.37] |
| contains + ConceptNet | -0.05 [-0.34, +0.23] | +0.13 [-0.68, +0.98] | -0.06 [-0.36, +0.22] |
| contains + sg_rel + ConceptNet | -0.03 [-0.31, +0.23] | -0.26 [-1.10, +0.52] | -0.00 [-0.29, +0.29] |
