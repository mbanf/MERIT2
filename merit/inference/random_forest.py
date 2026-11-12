"""Random Forest-based GRN inference — port of compute_randomforest_based_GRN()."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def compute_rf_grn(
    expression: np.ndarray,
    genes: list[str],
    regulators: list[str],
    n_trees: int = 1000,
    mtry: str | int = "sqrt",
    seed: int = 1234,
    n_cpus: int = 4,
) -> np.ndarray:
    """Infer GRN edges via Random Forest importance scores.

    For each target gene, fit an RF model predicting its expression from all
    regulators (excluding itself). The importance scores (impurity-based) give
    the edge weights.

    Parameters
    ----------
    expression : np.ndarray
        Z-scored expression matrix, shape (n_genes, n_samples).
    genes : list[str]
        Gene names (rows of expression).
    regulators : list[str]
        TF names (subset of genes).
    n_trees : int
        Number of trees per RF model.
    mtry : str or int
        Features per split. "sqrt" or "all" or an integer.
    seed : int
        Random seed.
    n_cpus : int
        Number of parallel threads for each RF fit.

    Returns
    -------
    np.ndarray
        Edge weight matrix, shape (n_regulators, n_genes). Higher = stronger
        predicted regulatory link.
    """
    n_samples = expression.shape[1]
    gene_idx = {g: i for i, g in enumerate(genes)}
    reg_idx = {r: i for i, r in enumerate(regulators)}

    # Resolve mtry
    if isinstance(mtry, str):
        if mtry == "sqrt":
            max_features = "sqrt"
        elif mtry == "all":
            max_features = len(regulators) - 1
        else:
            raise ValueError(f"Invalid mtry: {mtry}")
    else:
        max_features = mtry

    mat = np.zeros((len(regulators), len(genes)), dtype=np.float64)

    # Transpose: samples x genes for sklearn
    X_all = expression.T  # (n_samples, n_genes)

    logger.info(
        "RF GRN inference: %d genes, %d regulators, %d trees",
        len(genes), len(regulators), n_trees,
    )

    for i, target in enumerate(genes):
        if (i + 1) % 500 == 0 or i == 0:
            logger.info("  RF: gene %d/%d (%.1f%%)", i + 1, len(genes), (i + 1) / len(genes) * 100)

        target_idx_in_expr = gene_idx[target]
        y = X_all[:, target_idx_in_expr]

        # Regulators minus the target itself
        reg_names_i = [r for r in regulators if r != target]
        reg_col_indices = [gene_idx[r] for r in reg_names_i]
        X = X_all[:, reg_col_indices]

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_features=max_features,
            random_state=seed,
            n_jobs=n_cpus,
        )
        rf.fit(X, y)

        for j, r in enumerate(reg_names_i):
            mat[reg_idx[r], i] = rf.feature_importances_[j]

    # Normalize by number of samples (matching R code)
    return mat / n_samples
