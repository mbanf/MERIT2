"""Monte Carlo-based threshold selection for GRN methods."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def shuffle_expression(expression: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle expression values per gene (row-wise), destroying regulatory signal.

    This creates a null distribution for Monte Carlo threshold estimation.
    Port of: X.shuffled <- t(apply(X, 1, function(m) sample(m, length(m))))
    """
    rng = np.random.RandomState(seed)
    shuffled = expression.copy()
    for i in range(shuffled.shape[0]):
        rng.shuffle(shuffled[i])
    return shuffled


def compute_threshold(
    background_matrices: list[np.ndarray],
    quantile: float = 0.95,
) -> float:
    """Compute Monte Carlo threshold from background (shuffled) score matrices.

    The threshold is the average of the given quantile across all background runs.
    Port of: th.grns = sum(v.th.grns) / n.lead_method_expression_shuffling
    """
    thresholds = [np.quantile(bg, quantile) for bg in background_matrices]
    threshold = float(np.mean(thresholds))
    logger.info(
        "MC threshold (q=%.2f): %.6f (from %d background runs)",
        quantile, threshold, len(background_matrices),
    )
    return threshold


def apply_monte_carlo_threshold(
    scores: np.ndarray,
    background_matrices: list[np.ndarray],
    quantile: float = 0.95,
) -> np.ndarray:
    """Zero out scores below the Monte Carlo-derived threshold.

    Returns a copy with sub-threshold values set to zero.
    """
    threshold = compute_threshold(background_matrices, quantile)
    result = scores.copy()
    result[result < threshold] = 0.0
    return result
