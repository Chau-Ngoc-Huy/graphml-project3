# Level 2 evaluation query set: how it was built and checked

Files: `eval_queries.json` (full), `eval_queries.csv` (review sheet). Code: `src/build_eval_queries.py` (seed 0, reproducible).
Source: Visual Genome `scene_graphs.json` (108,077 images). Images and GPU are not needed to rebuild it.

## Size
24 queries: 12 direct, 12 indirect. Query text is "subject predicate object" (e.g. "woman feeding giraffe").

## Construction
Common to both groups
- Only action predicates (holding, riding, playing with, eating, carrying, flying, pulling, throwing, hitting, catching, kicking, watching, reading, driving, pushing, feeding, petting, sitting on, standing on, walking on).
- Subject and object names normalised (lower-case, leading/trailing articles removed); both must occur in at least 200 images.
- At most 3 queries per predicate, no repeated (subject, object) pair, vague objects (hand, game, food, ...) excluded.

Direct group: positives = images whose annotation contains the triple s-p-o (5 to 60 allowed; observed 5 to 15).

Indirect group: positives = images where s and o are connected only through a third object x (s -r1-> x -r2-> o)
and have no annotated relation between s and o in either direction. The relation must be inferred, not read off
the image graph. 5 to 60 positives allowed (observed 5 to 56). Example objects x per query are listed in the CSV.

## Automatic checks (run, all passed)
- No duplicate query text; no duplicate (subject, object) pair; at most 3 queries per predicate.
- Direct positives contain the triple and indirect positives have no s-o relation: guaranteed by construction in the code.
- The direct and indirect positive image sets do not overlap (0 shared images).

## Not checked (limitations, say this in the report)
- Nobody looked at the images. The indirect positives are candidates: a 2-hop path through a third object does not
  prove the image shows "s p o" (e.g. "cat sitting on bed" via laptop could be wrong). Indirect results are therefore
  noisy and likely an underestimate of true performance.
- Visual Genome annotations are sparse: images that really show a direct query but lack the annotation are missing from
  its positives, which also lowers recall.
- Both limits apply equally to every method compared, but they should be stated.

## Use rules (spec section 6)
- Evaluation only. Never tune hyperparameters, alpha, or train on these queries.
- When training anything on Visual Genome, exclude the positive image ids listed in `eval_queries.json`.
- Optional spot check if time allows: open ~5 indirect examples per query on Kaggle and mark keep/drop in the CSV.
