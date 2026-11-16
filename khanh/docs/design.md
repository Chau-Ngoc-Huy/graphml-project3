# Design: two-level multimodal retrieval with graph explanations (Khanh, Week 1)

## 1. Idea in one paragraph
Query text -> **Tier 1** fast retrieval with a frozen CLIP encoder gives the top-k candidate images by cosine -> **Tier 2** a graph model scores each
candidate using relational knowledge (concept relations learned from scene graphs / an MMKG) -> candidates are reranked by a fused score ->
the relations/paths that produced the graph score are the explanation. "Two levels" = the two project levels that plug into the same
Tier 2 slot: **Level 1** graph scorer = GraphSAGE over a concept/image graph; **Level 2** graph scorer = a 2024+ multimodal KG-completion
model (AdaMF-MAT). Same pipeline, split, metrics; only the scorer changes, so the comparison is fair.

## 2. Pipeline (shared by all methods)
1. **Encode (frozen):** OpenCLIP ViT-B-32 image and text features, L2-normalised. Never fine-tuned.
2. **Tier 1:** for query q, score s_clip(q, i) = cos(t_q, v_i) over the gallery; keep top-k (k=100).
3. **Tier 2:** graph score s_graph(q, i) from a pluggable scorer (section 3).
4. **Fusion:** score = a * z(s_clip) + (1-a) * z(s_graph), z = per-query z-score over the gallery. Rank the top-k by this score; items outside top-k keep CLIP order below them. `a` and all hyperparameters are chosen on validation only.
5. **Explain:** return the concept relations/paths that contributed to s_graph for each returned image.

## 3. The two scorers (the only thing that differs)
**Level 1 - GraphSAGE scorer (implemented, smoke-tested on fake features).**
- Graph: concept nodes (object names) + image nodes. Edges: image--concept "contains" (the image's object list only, no relations) and concept--concept relation edges counted from TRAIN images only.
- Model: 2-layer GraphSAGE-mean; node features = CLIP text embedding (concepts), CLIP image embedding (images). Query projected to the same space; s_graph = cos(proj(t_q), h_i).
- Training: InfoNCE; positives = train images that contain the query triple; other queries' positives masked from the negatives.

**Level 2 - AdaMF-MAT scorer (reproduction running on MKG-W / MKG-Y; adapter not written).**
- AdaMF-MAT is a multimodal KG-completion model: structure embeddings + visual + textual embeddings fused with adaptive modality weights, trained with modality-adversarial samples (zjukg/AdaMF-MAT, LREC-COLING 2024).
- Plan A (faithful to "same pipeline"): export the same Visual Genome graph as an MMKG in the repo's format (entities = concepts and images, relations = contains + predicates, visual/textual embeddings from CLIP), train AdaMF-MAT on train triples, and score an image for a query as the model's plausibility of (image, contains, c) summed over the query concepts c in {s, o}. Relation knowledge reaches the image through the learned concept relations, which is what the indirect queries need.
- Plan B (fallback if Plan A is too slow): report AdaMF-MAT as a separate KGC reproduction on MKG-W and MKG-Y, and keep the VG pipeline on Level 1 only. Needs the lecturer's OK.
- Status: reproduction run in progress. Plan A needs the repo's benchmark file format (entity2id, relation2id, train2id/valid2id/test2id) confirmed before export code is written.

## 4. Data, split and leakage rules (spec section 6)
- Data: Visual Genome scene graphs (108,077 images) + images (dannywu375/visualgenome; ids match 108,077/108,077).
- One fixed image-level split (train/val/test, seed 0) shared by every method; replace with Yến's split file when it exists.
- Concept relation edges and all training use TRAIN images only. A val/test image exposes only its object list.
- `a` chosen on validation. The 20+ query set (`data/eval_queries.json`) is evaluation only; its positive images are excluded from any training.
- 3 seeds for every main experiment, report mean +- std.

## 5. Evaluation
- Metrics: Recall@1/5/10, MRR, nDCG@10 (hit-rate Recall). Same `evaluate` for every method (replace with Thang's module when available).
- Query groups reported separately: direct (triple annotated) vs indirect (s and o linked only through a third object, relation must be inferred).
- Ablations on the concept--concept edges: drop, shuffle, random graph. Expected: gains vanish with shuffled/random edges, proving the relational structure matters.

## 6. Interfaces with teammates
| From | Need | Plugs in at |
|---|---|---|
| Yen | split file + normalised labels | step 2 / data loading |
| Thang | evaluate.py, CLIP features, baseline numbers | metrics + Tier 1 |
| Huy | graph builder + GNN numbers | Level 1 scorer, fair comparison |
| Duong | takes my best scorer for the Vietnamese MMKG | Tier 2 |

## 7. Risks
- KGC model vs retrieval: Plan A needs a mapping from VG to the repo's format; may take longer than the reproduction itself.
- Indirect ground truth is unverified (see data/EVAL_QUERIES_README.md): results are noisy.
- The Level 1 numbers are from a stand-in implementation until Huy/Thang's pieces replace it.
