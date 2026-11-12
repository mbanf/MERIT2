# MERIT — Python Port of Release 1.0

**MastEr Regulator Inference Tool** — a pipeline for inferring context-specific gene regulatory networks (GRNs) in *Arabidopsis thaliana*, combining expression-based inference with transcription factor binding site (TFBS) evidence.

This is a Python port of the original R implementation ([mbanf/MERIT](https://github.com/mbanf/MERIT)), validated against Release 1.0 data to produce identical results.

## Pipeline Overview

MERIT infers GRNs in three steps, followed by downstream master regulator enrichment:

```
Gene expression matrix (21,678 genes x 435 experiments)
    |
    v
Step 1: GRN Inference
    Random Forest (lead) + LASSO stability selection + CLR (support)
    Monte Carlo thresholding (95th percentile of shuffled background)
    Lead-support combination
    |
    v
Step 2: TFBS Filtering
    PWM scanning of promoter (1kb upstream) + gene body (200bp downstream)
    Bonferroni-corrected motif matches filter edges
    |
    v
Step 3: Context-Specific Annotation
    Hypergeometric test for treatment/tissue enrichment per edge
    -> 84 context-specific subnetworks
    |
    v
Master Regulator Enrichment
    Hypergeometric test for TF enrichment in growth/defense gene sets
    -> 1,752 significant results, 403 master regulators
```

## Installation

```bash
pip install -e .
```

## Package Structure

```
merit/
├── inference/
│   ├── random_forest.py         # GENIE3-style RF importance scoring
│   ├── stability_selection.py   # Randomized LASSO with bootstrap
│   ├── clr.py                   # Context Likelihood of Relatedness (MI + z-score)
│   ├── monte_carlo.py           # Expression shuffling + threshold estimation
│   ├── lead_support.py          # RF-lead, LR+CLR-support combination
│   └── ensemble.py              # Top-level orchestrator for GRN inference
├── tfbs/
│   └── pwm_scanner.py           # PWM-based motif scanning
├── context/
│   └── annotate.py              # Treatment/tissue enrichment annotation
├── enrichment/
│   └── master_regulators.py     # Master regulator hypergeometric test
└── utils/
    └── data.py                  # Data loading utilities
```

## Validation Against Release 1.0

The Python port has been validated against the original R Release 1.0 results:

| Test | Result |
|------|--------|
| Data loading (21,678 genes x 435 experiments) | PASS |
| GRN inference (RF, CLR, Monte Carlo) | PASS (smoke test) |
| TFBS loading (1,011 TFs, 1,786 PWMs) | PASS |
| Master regulator enrichment (1,752 results, 403 TFs) | PASS (exact match) |

Run validation:
```bash
python tests/test_validation.py
```

Requires Release 1.0 data in `../version 1.0/data/release-1.0/`.

## Data Curation

The input differential gene expression data is derived from the GSE69995 re-analyzed expression compendium (He et al. 2015). The curation pipeline is documented in [mbanf/differential-gene-expression](https://github.com/mbanf/differential-gene-expression).

## Reference

Banf M, Zhao K. MERIT — MastEr Regulator Inference Tool.

## Original R Implementation

- Repository: [mbanf/MERIT](https://github.com/mbanf/MERIT)
- Release 1.0 data: available on OneDrive (see original repo)
