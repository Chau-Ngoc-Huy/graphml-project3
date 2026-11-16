# Level 2 notes: which 2024+ method to reproduce

Status: research only (web search, nothing downloaded or run). Facts below are from search snippets, verify in the repos.

## Candidates (both from zjukg, Zhejiang Univ.)
| | AdaMF-MAT | NativE |
|---|---|---|
| Venue | LREC-COLING 2024 (arXiv 2402.15444) | SIGIR 2024 (arXiv 2406.17605) |
| Code | https://github.com/zjukg/AdaMF-MAT | https://github.com/zjukg/NATIVE |
| Idea | adaptive modality weights for fusion + modality-adversarial training | relation-guided dual adaptive fusion for any modalities + collaborative modality adversarial training |
| Data | MMRNS datasets (DB15K, MKG-W, MKG-Y per README; verify) + embeddings from Google Drive | WildKGC benchmark (5 datasets) + embeddings from Google Drive |
| Setup | shell scripts in scripts/ | Python 3.8, torch 1.9.1, built on OpenKE |

Spec (section 10) lists MKG-W / MKG-Y for Level 2 -> AdaMF-MAT matches the spec's datasets most directly.
Recommendation: **AdaMF-MAT first** (simpler, datasets match spec); NativE only if time.

## AdaMF-MAT repo facts (read from README/scripts via a subagent, not yet run)
- Needs: numpy 1.23.3, scikit_learn 1.1.2, torch 1.9.1+cu111, tqdm. No Python version stated. Old torch likely won't install on Kaggle's Python; try the preinstalled torch first.
- Data: MMKG datasets from MMRNS (github.com/quqxui/MMRNS); processed multimodal embeddings on the authors' Google Drive folder, put in `embeddings/`. README does not say where triple files live (probably `benchmarks/`; unverified).
- Run: `python run_adamf_mat.py -dataset=<DB15K|MKG-W|MKG-Y> ...` via scripts/run_*.sh. MKG-W: dim 200, margin 12, lr 2e-5, 1000 epochs. MKG-Y: dim 200, margin 4. DB15K: dim 250, margin 12.
- README's DB15K example (lr 1e-4, -lrg) differs from scripts/run_db15k.sh (lr 2e-5); check the paper.
- README shows no result numbers: reproduction targets must come from the paper.
- Notebook: src/adamf_mat_kaggle.ipynb

## Key risk: these are KG-completion methods
They predict (head, relation, ?) over entities with image/text embeddings. Retrieval pipeline is query text -> images. Two ways to connect:
A. Use the trained multimodal entity/relation scoring as the graph_score in CLIP+Graph on Visual Genome concepts. Needs mapping VG concepts to KG entities. Most faithful to the "same pipeline" rule, most work.
B. Run the method on MKG-W/MKG-Y as KGC (Hits@k, MRR) and report separately, then run CLIP+Graph on VG. Least work, may not satisfy "same pipeline and split". Ask the lecturer.
Decision needed from team/lecturer before heavy work.

## Independent next steps (no teammate needed)
1. Clone AdaMF-MAT on Kaggle, download its data+embeddings, reproduce the paper numbers on one dataset.
2. Hand-check the query set (src outputs) and fix the indirect-query definition.
3. Write method design page and report sections.
