"""Top-level ensemble GRN inference — orchestrates RF + LASSO + CLR + MC thresholding."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from merit.inference.random_forest import compute_rf_grn
from merit.inference.stability_selection import compute_lasso_grn
from merit.inference.clr import compute_clr_grn
from merit.inference.monte_carlo import shuffle_expression
from merit.inference.lead_support import combine_lead_support

logger = logging.getLogger(__name__)


def _zscore(expression: np.ndarray) -> np.ndarray:
    """Z-score normalize each gene (row) across samples."""
    mean = expression.mean(axis=1, keepdims=True)
    std = expression.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (expression - mean) / std


def infer_grn_ensemble(
    expression: np.ndarray,
    genes: list[str],
    regulators: list[str],
    n_trees: int = 1000,
    n_bootstrap: int = 100,
    n_steps_lars: int = 5,
    n_mc_shuffles: int = 1,
    th_lead: float = 0.95,
    th_support: float = 0.95,
    n_grn_support: int = 1,
    seed: int = 1234,
    n_cpus: int = 4,
    cache_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Run the full MERIT ensemble GRN inference pipeline (Step 1).

    Port of grn_inference_from_expression() from
    compute_ensemble_regression_with_montecarlo_based_stability_selection.R

    Pipeline:
        1. Z-score normalize expression
        2. Run RF, LASSO (stability selection), CLR on real data
        3. Run all three on MC-shuffled data (n_mc_shuffles times)
        4. Threshold using MC background quantiles
        5. Combine via lead-support framework

    Parameters
    ----------
    expression : np.ndarray
        Fold-change matrix, shape (n_genes, n_experiments).
    genes : list[str]
        Gene names.
    regulators : list[str]
        TF names (must be subset of genes).
    n_trees : int
        RF trees per gene model.
    n_bootstrap : int
        Bootstrap iterations for stability selection.
    n_steps_lars : int
        LARS steps for stability selection.
    n_mc_shuffles : int
        Number of Monte Carlo background shuffles.
    th_lead : float
        Quantile threshold for lead method (RF).
    th_support : float
        Quantile threshold for support methods (LR, CLR).
    n_grn_support : int
        Minimum number of support methods confirming a link.
    seed : int
        Random seed.
    n_cpus : int
        Parallel workers.
    cache_dir : Path, optional
        Directory to cache intermediate results.

    Returns
    -------
    dict with keys 'lead_support', 'rf', 'lr', 'clr' — numpy arrays
        shape (n_regulators, n_genes).
    """
    # Filter regulators to those in expression matrix
    regulators = [r for r in regulators if r in genes]
    logger.info(
        "Ensemble GRN inference: %d genes, %d regulators, %d MC shuffles",
        len(genes), len(regulators), n_mc_shuffles,
    )

    # Z-score
    X = _zscore(expression)

    # --- Real data ---
    logger.info("Running RF on real data...")
    rf_scores = compute_rf_grn(X, genes, regulators, n_trees=n_trees, seed=seed, n_cpus=n_cpus)

    logger.info("Running LASSO stability selection on real data...")
    lr_scores = compute_lasso_grn(
        X, genes, regulators, n_bootstrap=n_bootstrap, n_steps=n_steps_lars,
        seed=seed, n_cpus=n_cpus,
    )

    logger.info("Running CLR on real data...")
    clr_scores = compute_clr_grn(X, genes, regulators)

    # --- Monte Carlo background ---
    rf_bg, lr_bg, clr_bg = [], [], []
    for i in range(n_mc_shuffles):
        mc_seed = seed + 25 * (i + 1)
        logger.info("MC shuffle %d/%d (seed=%d)...", i + 1, n_mc_shuffles, mc_seed)

        X_shuffled = shuffle_expression(X, mc_seed)

        rf_bg.append(compute_rf_grn(
            X_shuffled, genes, regulators, n_trees=n_trees, seed=mc_seed, n_cpus=n_cpus,
        ))
        lr_bg.append(compute_lasso_grn(
            X_shuffled, genes, regulators, n_bootstrap=n_bootstrap, n_steps=n_steps_lars,
            seed=mc_seed, n_cpus=n_cpus,
        ))
        clr_bg.append(compute_clr_grn(X_shuffled, genes, regulators))

    # --- Combine ---
    result = combine_lead_support(
        rf_scores, rf_bg, lr_scores, lr_bg, clr_scores, clr_bg,
        th_lead=th_lead, th_support=th_support, n_support=n_grn_support,
    )

    return result
