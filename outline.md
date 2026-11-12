# MERIT v2.0 — Architecture & Plan

## Repository Structure

- **MERIT2** (https://github.com/mbanf/MERIT2)
  - `main` branch — v2.0 with GNN/CNN/RL extensions
  - `release-1.0` branch — clean Python port of R v1.0 (validated)
- **differential-gene-expression** (https://github.com/mbanf/differential-gene-expression)
  - Data curation: GSE69995 → differential expression matrices
  - He et al. (2016) Plant J, 86: 472-480. doi:10.1111/tpj.13175

## v2.0 Pipeline Architecture

### Step 0: Data Curation
- Raw expression (GSE69995) + annotation → Welch t-test per comparison
- Output: fold-change + p-value matrices (21,678 genes x 435 comparisons)
- Stored in differential-gene-expression repo

### Step 1: GRN Inference (keep continuous scores)
- **RF scores** (continuous importance, not thresholded)
- **LASSO scores** (continuous stability frequency)
- **CLR scores** (continuous MI z-scores)
- **Monte Carlo confidence** — instead of hard 95th percentile cutoff,
  compute per-edge confidence: fraction of background runs exceeded
- All precomputed and stored as .rds in Release 1.0 tmp/

### Step 2: TFBS Evidence (continuous, not binary filter)
Two approaches (both produce continuous scores):

**Level 1 — Continuous PWM features** (interpretable, no training needed):
- best_score: max log-odds score across all motif hits
- n_hits: number of hits above lenient threshold
- best_position: relative to TSS (proximal vs distal)
- score_percentile: rank among all genes for this TF

**Level 3 — CNN with PWM-initialized filters** (learned):
- Conv1d filters initialized from CIS-BP PWMs (1,786 motifs)
- Additional random filters learn novel motifs/grammar
- Trained on DAP-seq (1.79M edges, 349 TFs)
- Outputs continuous binding probability per TF-gene pair

### Step 3: GAT Integration (replaces lead-support + hard thresholding)
- **Node features**: gene expression profiles (435-dim)
- **Edge features** (per TF-gene pair):
  - RF score, LASSO score, CLR score (3)
  - RF/LR/CLR Monte Carlo confidence (3)
  - PWM continuous features (4) or CNN binding probability (1)
  - = 7-10 dimensional edge feature vector
- **Graph Attention Network** learns to integrate all evidence
- **Training**: gold-standard edges from ConnecTF + ATRM + DAP-seq
- **Output**: edge probabilities (continuous)

### Step 4: RL-Based Network Refinement (iterative)
- Agent observes: GAT output + network topology metrics
- Agent adjusts: edge threshold, evidence re-weighting
- Reward: F1 vs held-out gold-standard + biological plausibility
- Iterates until convergence

### Step 5: Context-Specific Annotation (as v1.0)
- Hypergeometric enrichment for treatment/tissue per edge
- Uses refined network from Step 4
- → 84 context-specific subnetworks

### Step 6: Master Regulator Enrichment (as v1.0)
- Hypergeometric test for TF enrichment in functional gene sets
- Applied after learning is complete
- → master regulators per context x domain

## Gold-Standard Data Sources

### Already in Release 1.0
| Dataset | Edges | TFs |
|---------|-------|-----|
| ATRM | ~1,405 | 323 |
| AtRegNet | ~1.6M | 289+ |
| DAP-seq | ~1.8M | 349 |

### New sources to integrate
- **ConnecTF** — 3.7M interactions, 423 TFs (ChIP-seq + DAP-seq + TARGET)
- **ATRM v2** — updated, 388 TFs, 974 references
- **TransAt v1.0** — TF binding atlas (Zenodo, 2025)

## Key Design Principles
1. **No hard thresholds** — everything stays continuous until final output
2. **Preserve interpretability** — PWM features alongside learned CNN features
3. **Leverage existing computation** — load v1.0 .rds intermediates, don't recompute
4. **Gold-standard supervision** — train GAT on validated interactions
5. **Comparable to v1.0** — same data, same inputs, better integration

## Publication Strategy
- Bioinformatics application note or full paper
- Compare v2.0 (GAT-integrated) vs v1.0 (lead-support) on same data
- Show improvement on gold-standard benchmarks
- Context-specific networks as biological validation

## Open Questions
- Can we accumulate more expression data beyond GSE69995?
- SCENIC/pySCENIC-style regulon analysis as complementary view?
- Single-cell extension?
- AraCyc version placeholder in manuscript ("XXX")
