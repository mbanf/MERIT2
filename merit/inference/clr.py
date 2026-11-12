"""Context Likelihood of Relatedness (CLR) — port of parmigene::clr()."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


def _compute_mi_matrix(expression: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Compute mutual information matrix between all gene pairs.

    Port of parmigene::knnmi.all(). Uses sklearn's kNN-based MI estimator.
    """
    n_genes = expression.shape[0]
    X = expression.T  # samples x genes

    mi = np.zeros((n_genes, n_genes), dtype=np.float64)

    for i in range(n_genes):
        if (i + 1) % 1000 == 0:
            logger.info("  MI: gene %d/%d", i + 1, n_genes)
        mi_scores = mutual_info_regression(X, X[:, i], n_neighbors=n_neighbors)
        mi[i, :] = mi_scores

    # Symmetrize
    mi = (mi + mi.T) / 2
    np.fill_diagonal(mi, 0)
    return mi


def _clr_transform(mi: np.ndarray) -> np.ndarray:
    """Apply CLR (Context Likelihood of Relatedness) transformation.

    For each pair (i, j), computes z-scores of MI(i,j) relative to the
    distribution of MI values for gene i and gene j, then combines them.

    CLR(i,j) = sqrt(max(z_i, 0)^2 + max(z_j, 0)^2)
    """
    n = mi.shape[0]

    # Row-wise z-scores
    row_mean = mi.mean(axis=1, keepdims=True)
    row_std = mi.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    z_rows = (mi - row_mean) / row_std

    # Column-wise z-scores
    col_mean = mi.mean(axis=0, keepdims=True)
    col_std = mi.std(axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    z_cols = (mi - col_mean) / col_std

    # CLR score
    z_rows = np.maximum(z_rows, 0)
    z_cols = np.maximum(z_cols, 0)
    clr = np.sqrt(z_rows**2 + z_cols**2)

    np.fill_diagonal(clr, 0)
    return clr


def compute_clr_grn(
    expression: np.ndarray,
    genes: list[str],
    regulators: list[str],
    n_neighbors: int = 5,
) -> np.ndarray:
    """Infer GRN via CLR (mutual information + z-score transformation).

    Returns
    -------
    np.ndarray
        CLR score matrix, shape (n_regulators, n_genes).
    """
    logger.info("CLR GRN inference: %d genes, %d regulators", len(genes), len(regulators))

    mi = _compute_mi_matrix(expression, n_neighbors=n_neighbors)
    clr = _clr_transform(mi)

    # Subset to regulators x genes
    reg_indices = [genes.index(r) for r in regulators]
    return clr[np.ix_(reg_indices, range(len(genes)))]
