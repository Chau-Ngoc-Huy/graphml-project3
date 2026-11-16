# Level 2 on Visual Genome: CLIP vs CLIP + AdaMF-MAT (first run)

Run: src/level2_vg_kaggle.ipynb on Kaggle (notebook output: results/results_level2.zip; add the Kaggle version link here).
Training stage worked (export 8,324 images, 11,848 entities, 31 relations, 162,342 train triples; 207 epochs, 69 s/epoch, ~4h).
The Kaggle evaluation stage crashed on a key bug (val queries use "pos", eval queries "positives"); fixed in src/run_level2.py and
the evaluation was re-run locally on CPU with the saved checkpoint (results/results_level2.json, ~35 s).
Setup: gallery = 1,133 test images; 24 eval queries (12 direct, 12 indirect); alpha chosen on 51 val queries; single seed (42).

## Results on the 24 eval queries (test gallery)
| Query group | Method | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| direct (12) | CLIP | .417 | .833 | .833 | .559 | .499 |
| | AdaMF-MAT only | .417 | .917 | .917 | .615 | .609 |
| | CLIP + AdaMF-MAT (a=0.1) | .417 | .917 | 1.0 | .615 | .620 |
| | **CLIP + presence-only (no relations)** | **.917** | .917 | 1.0 | **.929** | .742 |
| indirect (12) | CLIP | .500 | .917 | .917 | .700 | .523 |
| | AdaMF-MAT only | .500 | 1.0 | 1.0 | .722 | .672 |
| | CLIP + AdaMF-MAT | .583 | 1.0 | 1.0 | .739 | .685 |
| | **CLIP + presence-only** | **.667** | 1.0 | 1.0 | **.833** | .727 |
| all (24) | CLIP | .458 | .875 | .875 | .630 | .511 |
| | CLIP + AdaMF-MAT | .500 | .958 | 1.0 | .677 | .653 |
| | **CLIP + presence-only** | **.792** | .958 | 1.0 | **.881** | .735 |

presence-only = count of the query's concepts in the image's annotated object list (no relational knowledge at all).

## What this does and does not show
- CLIP + AdaMF-MAT beats CLIP, but a presence-only baseline that uses NO relations beats AdaMF-MAT clearly (MRR .881 vs .677).
  So the gain over CLIP comes from knowing which objects are in the image (annotated labels), not from learned relational structure.
  The claim "the graph helps" is NOT supported by this experiment.
- Why the test cannot separate the two: the positives are defined from the same annotations (direct: triple annotated; indirect: both
  objects present, linked via a third object), and the graph scorer sees the annotated object list. Both objects are present in every
  positive, so presence alone nearly solves it. The scorer also never sees an image's own relations (hidden by design), so it cannot
  tell "s p o" from "s next to o" for images containing both.
- Only 12 queries per group: one query = 8% of R@1, differences of 1-2 queries are not significant. One seed (spec asks >= 3).
- No ablation of the relation structure yet (drop/shuffle/random edges), which is required for the claim.

## What a valid indirect test needs (next)
Simulate the spec's case "the image only shows related objects": for indirect eval positives, remove the query's object from the image's
object list (also from its contains edges in training/export), so presence-only fails and only relational knowledge can infer it.
Then compare AdaMF-MAT vs presence-only vs shuffled-edge AdaMF-MAT. Needs one re-export + retrain (~4h on Kaggle).
