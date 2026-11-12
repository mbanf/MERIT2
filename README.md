# MERIT v2.0

**MastEr Regulator Inference Tool** — a pipeline for inferring context-specific gene regulatory networks (GRNs) in *Arabidopsis thaliana*.

MERIT v2.0 is a Python rewrite of the [original R implementation](https://github.com/mbanf/MERIT), extended with deep learning (GNN, CNN) and reinforcement learning approaches. It includes a Nextflow pipeline for scalable execution.

> For the validated Python port of Release 1.0 (reproducing the original results exactly), see the [`release-1.0`](../../tree/release-1.0) branch.

## Pipeline

```
Gene expression matrix (21,678 genes x 435 experiments)
    |
    v
Step 1: GRN Inference (choose method)
    - ensemble: RF (lead) + LASSO + CLR (support) + Monte Carlo thresholding
    - gnn: Graph Attention Network trained on gold-standard edges
    |
    v
Step 2: TFBS Filtering (choose method)
    - pwm: Position Weight Matrix scanning with Bonferroni correction
    - cnn: CNN-based motif learning from sequence (DeepBind-style)
    |
    v
Step 3: Context-Specific Annotation
    Hypergeometric test for treatment/tissue enrichment per edge
    -> 84 context-specific subnetworks
    |
    v
Step 4: Master Regulator Enrichment
    Hypergeometric test for TF enrichment in functional gene sets
    |
    v
Optional: RL-based Network Refinement
    REINFORCE agent learns optimal thresholds and method weights
```

## Installation

```bash
# Core (ensemble methods only)
pip install -e .

# With deep learning (GNN + CNN)
pip install -e ".[deep]"

# With RL agent
pip install -e ".[agents]"
```

## Usage

### CLI

```bash
# Full pipeline with default (ensemble) method
merit --config config/default.yaml --step all

# GNN-based inference
merit --config config/default.yaml --step all --method gnn
```

### Nextflow

```bash
nextflow run main.nf --genes data/genes.txt --fold_change data/m.foldChange.txt ...
```

## Package Structure

```
merit/
├── inference/
│   ├── random_forest.py         # GENIE3-style RF importance scoring
│   ├── stability_selection.py   # Randomized LASSO with bootstrap
│   ├── clr.py                   # Context Likelihood of Relatedness
│   ├── monte_carlo.py           # Expression shuffling + thresholding
│   ├── lead_support.py          # Lead-support combination
│   ├── ensemble.py              # Orchestrator (v1.0 pipeline)
│   └── gnn.py                   # Graph Attention Network inference [NEW]
├── tfbs/
│   ├── pwm_scanner.py           # PWM-based motif scanning
│   └── cnn_scanner.py           # CNN-based motif learning [NEW]
├── context/
│   └── annotate.py              # Treatment/tissue annotation
├── enrichment/
│   └── master_regulators.py     # Master regulator analysis
├── agents/
│   └── refiner.py               # RL network refinement agent [NEW]
├── utils/
│   └── data.py                  # Data loading
└── cli.py                       # Command-line interface
```

## Data Curation

The input differential gene expression data is derived from the GSE69995 re-analyzed expression compendium (He et al. 2015). The curation pipeline is documented in [mbanf/differential-gene-expression](https://github.com/mbanf/differential-gene-expression) and ported on the [`data-curation`](../../tree/data-curation) branch.

## Validation

The core pipeline (ensemble methods) has been validated against Release 1.0:

| Test | Result |
|------|--------|
| Data loading (21,678 genes x 435 experiments) | PASS |
| GRN inference (RF, CLR, Monte Carlo) | PASS |
| TFBS loading (1,011 TFs, 1,786 PWMs) | PASS |
| Master regulator enrichment (1,752 / 403 TFs) | PASS (exact match) |

```bash
python tests/test_validation.py
```

## References

- Banf M, Zhao K. MERIT — MastEr Regulator Inference Tool.
- Original R implementation: [mbanf/MERIT](https://github.com/mbanf/MERIT)
- Expression data: He et al. 2015 (GSE69995)
