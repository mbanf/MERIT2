"""LASSO with stability selection — port of stabilitySelection() + compute_linearRegressionWithStabilitySelection_based_GRN()."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import lars_path

logger = logging.getLogger(__name__)


def _stability_selection_single(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 100,
    n_steps: int = 5,
    alpha: float = 0.2,
) -> np.ndarray:
    """Stability selection for a single target gene.

    Port of stabilitySelection() from functions.R.
    Runs randomized LASSO (LARS) on bootstrap half-samples, tracks how
    frequently each regulator is selected across steps, returns max frequency.
    """
    n_samples, n_features = x.shape
    halfsize = n_samples // 2
    freq = np.zeros((n_steps + 1, n_features))

    for _ in range(n_bootstrap):
        # Randomly reweight features
        weights = np.random.uniform(alpha, 1.0, size=n_features)
        xs = x * weights[np.newaxis, :]

        # Random split into two halves
        perm = np.random.permutation(n_samples)
        i1, i2 = perm[:halfsize], perm[halfsize:]

        for idx in (i1, i2):
            try:
                _, _, coefs = lars_path(
                    xs[idx], y[idx], method="lasso", max_iter=n_steps
                )
                # coefs shape: (n_features, n_steps_actual)
                n_actual = min(coefs.shape[1], n_steps + 1)
                freq[:n_actual] += np.abs(np.sign(coefs[:, :n_actual].T))
            except Exception:
                continue

    freq /= 2 * n_bootstrap
    # Max frequency across LARS steps for each feature
    return freq.max(axis=0)


def _infer_single_gene(args):
    """Worker function for parallel gene processing."""
    target_idx, expression_T, gene_idx, reg_indices, reg_names, target_name, n_bootstrap, n_steps, seed = args
    np.random.seed(seed + target_idx)

    y = expression_T[:, gene_idx[target_name]]
    reg_names_i = [r for r in reg_names if r != target_name]
    reg_col_indices = [gene_idx[r] for r in reg_names_i]
    X = expression_T[:, reg_col_indices]

    scores = _stability_selection_single(X, y, n_bootstrap=n_bootstrap, n_steps=n_steps)
    return target_idx, reg_names_i, scores


def compute_lasso_grn(
    expression: np.ndarray,
    genes: list[str],
    regulators: list[str],
    n_bootstrap: int = 100,
    n_steps: int = 5,
    seed: int = 1234,
    n_cpus: int = 4,
) -> np.ndarray:
    """Infer GRN via LASSO with stability selection.

    For each target gene, runs randomized LASSO (LARS) with bootstrap
    half-sampling. The stability score (max selection frequency across
    LARS steps) gives the edge weight.

    Returns
    -------
    np.ndarray
        Edge weight matrix, shape (n_regulators, n_genes).
    """
    n_samples = expression.shape[1]
    gene_idx = {g: i for i, g in enumerate(genes)}
    reg_idx = {r: i for i, r in enumerate(regulators)}

    mat = np.zeros((len(regulators), len(genes)), dtype=np.float64)
    X_all = expression.T  # samples x genes

    logger.info(
        "LASSO stability selection: %d genes, %d regulators, %d bootstraps",
        len(genes), len(regulators), n_bootstrap,
    )

    # Build args for parallel processing
    args_list = []
    for i, target in enumerate(genes):
        reg_indices = [gene_idx[r] for r in regulators if r != target]
        args_list.append((
            i, X_all, gene_idx, reg_indices, regulators, target,
            n_bootstrap, n_steps, seed,
        ))

    # Process sequentially or in parallel
    if n_cpus <= 1:
        results = [_infer_single_gene(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(executor.map(_infer_single_gene, args_list))

    for target_idx, reg_names_i, scores in results:
        for j, r in enumerate(reg_names_i):
            mat[reg_idx[r], target_idx] = scores[j]

    return mat / n_samples
