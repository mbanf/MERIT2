"""Load R .rds files containing intermediate MERIT v1.0 results.

The Release 1.0 tmp/ directory contains precomputed score matrices:
    m.grn.RF.rds        — Random Forest importance scores (continuous)
    m.grn.LR.rds        — LASSO stability selection scores (continuous)
    m.grn.CLR.rds       — CLR mutual information z-scores (continuous)
    m.grn.RF_bg_N.rds   — Monte Carlo background RF scores (N = run index)
    m.grn.LR_bg_N.rds   — Monte Carlo background LASSO scores
    m.grn.CLR_bg_N.rds  — Monte Carlo background CLR scores
    m.grn.motifs.rds    — TFBS motif match matrix

These can be loaded and used as edge features for the GAT integration
step without recomputing the expensive GRN inference.

Requires: pip install pyreadr
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class V1Intermediates:
    """Precomputed v1.0 intermediate score matrices."""

    rf_scores: np.ndarray           # (n_tfs, n_genes) continuous RF importance
    lr_scores: np.ndarray           # (n_tfs, n_genes) LASSO stability scores
    clr_scores: np.ndarray          # (n_tfs, n_genes) CLR z-scores
    motif_matrix: np.ndarray        # (n_tfs, n_genes) binary TFBS matches
    rf_bg: list[np.ndarray]         # Monte Carlo background RF matrices
    lr_bg: list[np.ndarray]         # Monte Carlo background LASSO matrices
    clr_bg: list[np.ndarray]        # Monte Carlo background CLR matrices
    tf_names: list[str] | None = None
    gene_names: list[str] | None = None


def load_rds_matrix(filepath: str | Path) -> np.ndarray:
    """Load a single .rds file containing a numeric matrix."""
    try:
        import pyreadr
    except ImportError:
        raise ImportError(
            "Loading .rds files requires pyreadr. Install with: pip install pyreadr"
        )

    result = pyreadr.read_r(str(filepath))
    # pyreadr returns a dict of DataFrames; RDS files have one object
    df = list(result.values())[0]
    return df.values


def load_v1_intermediates(
    tmp_dir: str | Path,
    n_bg_runs: int | None = None,
) -> V1Intermediates:
    """Load all precomputed v1.0 intermediate matrices from the tmp/ directory.

    Parameters
    ----------
    tmp_dir : path
        Path to the Release 1.0 tmp/ directory containing .rds files.
    n_bg_runs : int, optional
        Number of Monte Carlo background runs to load.
        If None, auto-detects by scanning for m.grn.RF_bg_*.rds files.

    Returns
    -------
    V1Intermediates with all score matrices loaded.
    """
    tmp_dir = Path(tmp_dir)

    logger.info("Loading v1.0 intermediates from %s", tmp_dir)

    rf = load_rds_matrix(tmp_dir / "m.grn.RF.rds")
    lr = load_rds_matrix(tmp_dir / "m.grn.LR.rds")
    clr = load_rds_matrix(tmp_dir / "m.grn.CLR.rds")

    logger.info("  RF: %s, LR: %s, CLR: %s", rf.shape, lr.shape, clr.shape)

    # Motif matrix
    motif_path = tmp_dir / "m.grn.motifs.rds"
    if motif_path.exists():
        motifs = load_rds_matrix(motif_path)
        logger.info("  Motifs: %s", motifs.shape)
    else:
        logger.warning("  Motif matrix not found at %s", motif_path)
        motifs = np.ones_like(rf)

    # Monte Carlo backgrounds
    if n_bg_runs is None:
        bg_files = sorted(tmp_dir.glob("m.grn.RF_bg_*.rds"))
        n_bg_runs = len(bg_files)
    logger.info("  Loading %d Monte Carlo background runs...", n_bg_runs)

    rf_bg, lr_bg, clr_bg = [], [], []
    for i in range(1, n_bg_runs + 1):
        rf_bg_path = tmp_dir / f"m.grn.RF_bg_{i}.rds"
        lr_bg_path = tmp_dir / f"m.grn.LR_bg_{i}.rds"
        clr_bg_path = tmp_dir / f"m.grn.CLR_bg_{i}.rds"

        if rf_bg_path.exists():
            rf_bg.append(load_rds_matrix(rf_bg_path))
        if lr_bg_path.exists():
            lr_bg.append(load_rds_matrix(lr_bg_path))
        if clr_bg_path.exists():
            clr_bg.append(load_rds_matrix(clr_bg_path))

    logger.info("  Background runs: RF=%d, LR=%d, CLR=%d", len(rf_bg), len(lr_bg), len(clr_bg))

    return V1Intermediates(
        rf_scores=rf,
        lr_scores=lr,
        clr_scores=clr,
        motif_matrix=motifs,
        rf_bg=rf_bg,
        lr_bg=lr_bg,
        clr_bg=clr_bg,
    )


def compute_mc_confidence(
    real_scores: np.ndarray,
    bg_scores: list[np.ndarray],
    quantile: float = 0.95,
) -> np.ndarray:
    """Compute Monte Carlo confidence for each edge.

    Instead of a hard threshold (v1.0 approach), returns a continuous
    confidence score: the fraction of background runs where the real
    score exceeds the background score at that position.

    Parameters
    ----------
    real_scores : np.ndarray
        Real score matrix (n_tfs, n_genes).
    bg_scores : list[np.ndarray]
        List of background score matrices from shuffled data.
    quantile : float
        Reference quantile for normalization.

    Returns
    -------
    np.ndarray of same shape — confidence values in [0, 1].
    """
    if not bg_scores:
        logger.warning("No background scores; returning binary (score > 0)")
        return (real_scores > 0).astype(np.float64)

    n_runs = len(bg_scores)
    exceeded = np.zeros_like(real_scores, dtype=np.float64)

    for bg in bg_scores:
        # For each background run, compute the threshold at the given quantile
        bg_nonzero = bg[bg > 0]
        if len(bg_nonzero) > 0:
            threshold = np.quantile(bg_nonzero, quantile)
        else:
            threshold = 0.0
        exceeded += (real_scores > threshold).astype(np.float64)

    confidence = exceeded / n_runs
    return confidence


def build_edge_features(intermediates: V1Intermediates) -> np.ndarray:
    """Build per-edge feature matrix from v1.0 intermediates.

    Returns array of shape (n_tfs, n_genes, n_features) with features:
        0: RF importance score (normalized)
        1: LASSO stability score (normalized)
        2: CLR z-score (normalized)
        3: RF Monte Carlo confidence
        4: LR Monte Carlo confidence
        5: CLR Monte Carlo confidence
        6: Motif support (binary)
    """
    def _normalize(x):
        mx = x.max()
        return x / mx if mx > 0 else x

    rf_norm = _normalize(intermediates.rf_scores)
    lr_norm = _normalize(intermediates.lr_scores)
    clr_norm = _normalize(intermediates.clr_scores)

    rf_conf = compute_mc_confidence(intermediates.rf_scores, intermediates.rf_bg)
    lr_conf = compute_mc_confidence(intermediates.lr_scores, intermediates.lr_bg)
    clr_conf = compute_mc_confidence(intermediates.clr_scores, intermediates.clr_bg)

    motif = (intermediates.motif_matrix > 0).astype(np.float64)

    features = np.stack([
        rf_norm, lr_norm, clr_norm,
        rf_conf, lr_conf, clr_conf,
        motif,
    ], axis=-1)

    logger.info("Edge features: shape %s", features.shape)
    return features
