"""Lead-support GRN combination — port of load_lead_support_grn.R."""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def combine_lead_support(
    rf_scores: np.ndarray,
    rf_bg: list[np.ndarray],
    lr_scores: np.ndarray,
    lr_bg: list[np.ndarray],
    clr_scores: np.ndarray,
    clr_bg: list[np.ndarray],
    th_lead: float = 0.95,
    th_support: float = 0.95,
    n_support: int = 1,
) -> dict[str, np.ndarray]:
    """Combine lead (RF) and support (LR, CLR) GRN methods.

    1. Threshold RF using Monte Carlo background at th_lead quantile.
    2. Threshold LR and CLR using MC background at th_support quantile, binarize.
    3. Support matrix = LR_binary + CLR_binary >= n_support.
    4. Final network = RF_thresholded * support_mask.

    Returns dict with 'lead_support', 'rf', 'lr', 'clr' matrices.
    """
    from merit.inference.monte_carlo import apply_monte_carlo_threshold, compute_threshold

    # Lead method: RF
    rf_th = apply_monte_carlo_threshold(rf_scores, rf_bg, th_lead)
    logger.info("RF after threshold: %d non-zero edges", np.count_nonzero(rf_th))

    # Support method: LR
    lr_threshold = compute_threshold(lr_bg, th_support)
    lr_binary = (lr_scores >= lr_threshold).astype(np.float64)
    logger.info("LR binary edges: %d", int(lr_binary.sum()))

    # Support method: CLR
    clr_threshold = compute_threshold(clr_bg, th_support)
    clr_binary = (clr_scores >= clr_threshold).astype(np.float64)
    logger.info("CLR binary edges: %d", int(clr_binary.sum()))

    # Combine support
    support = lr_binary + clr_binary
    support_mask = (support >= n_support).astype(np.float64)
    logger.info("Support mask edges (>=%d methods): %d", n_support, int(support_mask.sum()))

    # Final: lead * support
    lead_support = rf_th * support_mask
    logger.info("Final lead-support edges: %d", np.count_nonzero(lead_support))

    return {
        "lead_support": lead_support,
        "rf": rf_th,
        "lr": lr_scores,
        "clr": clr_scores,
    }
