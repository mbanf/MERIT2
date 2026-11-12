"""PWM-based TFBS scanning — port of transcription_factor_binding.R (Step 2).

Scans promoter sequences (1kb upstream) and gene body (200bp downstream of TSS)
for TF binding motif matches using Position Weight Matrices.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)


def load_tf_motif_mapping(filepath: str | Path) -> dict[str, list[str]]:
    """Load TF -> motif ID mapping. Returns {tf_id: [motif_id, ...]}."""
    import pandas as pd

    df = pd.read_csv(filepath, sep="\t")
    mapping = {}
    for _, row in df.iterrows():
        tf_id = row["TF_ID"]
        motif_str = str(row["Motif_ID"])
        if motif_str and motif_str != "nan":
            mapping[tf_id] = [m.strip() for m in motif_str.split(",") if m.strip()]
    return mapping


def load_pwms(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load Position Weight Matrices from the custom MERIT format.

    The file is FASTA-like: each entry has a >motif_name header followed by
    four lines for A, C, G, T with tab-delimited values in brackets, e.g.:
        A\t[0\t1\t0\t]
        C\t[1\t0\t1\t]
        ...

    Returns {motif_id: np.ndarray of shape (4, motif_length)} with rows A,C,G,T
    normalized to probabilities.
    """
    pwms = {}
    with open(filepath) as f:
        current_name = None
        current_lines: list[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name and current_lines:
                    pwms[current_name] = _parse_pwm_lines(current_lines)
                current_name = line[1:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_name and current_lines:
            pwms[current_name] = _parse_pwm_lines(current_lines)

    return pwms


def _parse_pwm_lines(lines: list[str]) -> np.ndarray:
    """Parse four PWM lines (A, C, G, T) into a (4, L) probability matrix."""
    nuc_order = {"A": 0, "C": 1, "G": 2, "T": 3}
    rows = {}
    for line in lines:
        parts = line.split("\t")
        nuc = parts[0].strip().upper()
        if nuc not in nuc_order:
            continue
        vals = []
        for tok in parts[1:]:
            tok = tok.strip().replace("[", "").replace("]", "")
            if tok:
                vals.append(float(tok))
        rows[nuc] = vals

    if len(rows) != 4:
        raise ValueError(f"Expected 4 nucleotide rows, got {len(rows)}: {list(rows.keys())}")

    L = len(rows["A"])
    mat = np.zeros((4, L))
    for nuc, idx in nuc_order.items():
        mat[idx] = rows[nuc]

    # Normalize columns to probabilities
    col_sums = mat.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    mat = mat / col_sums

    return mat


def _pwm_to_log_odds(
    pwm: np.ndarray,
    bg: dict[str, float] | None = None,
    pseudocount: float = 0.001,
) -> np.ndarray:
    """Convert probability matrix to log-odds score matrix."""
    if bg is None:
        bg = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    bg_arr = np.array([bg["A"], bg["C"], bg["G"], bg["T"]])[:, np.newaxis]

    # Add pseudocount
    pwm_pc = pwm + pseudocount
    pwm_pc = pwm_pc / pwm_pc.sum(axis=0, keepdims=True)

    return np.log2(pwm_pc / bg_arr)


def _score_sequence(
    seq_str: str,
    log_odds: np.ndarray,
) -> float:
    """Score a sequence against a log-odds matrix, return max score across all positions."""
    nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    L = log_odds.shape[1]

    if len(seq_str) < L:
        return float("-inf")

    max_score = float("-inf")

    # Forward strand
    for i in range(len(seq_str) - L + 1):
        subseq = seq_str[i : i + L]
        score = 0.0
        valid = True
        for j, nuc in enumerate(subseq):
            if nuc in nuc_map:
                score += log_odds[nuc_map[nuc], j]
            else:
                valid = False
                break
        if valid:
            max_score = max(max_score, score)

    # Reverse complement
    rc_log_odds = log_odds[::-1, ::-1]
    for i in range(len(seq_str) - L + 1):
        subseq = seq_str[i : i + L]
        score = 0.0
        valid = True
        for j, nuc in enumerate(subseq):
            if nuc in nuc_map:
                score += rc_log_odds[nuc_map[nuc], j]
            else:
                valid = False
                break
        if valid:
            max_score = max(max_score, score)

    return max_score


def _compute_score_threshold(
    log_odds: np.ndarray,
    p_cutoff: float = 0.05,
    n_samples: int = 10000,
    bg: dict[str, float] | None = None,
) -> float:
    """Estimate score threshold for a given p-value via random sequence sampling."""
    if bg is None:
        bg = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

    rng = np.random.RandomState(42)
    nucs = ["A", "C", "G", "T"]
    probs = [bg[n] for n in nucs]
    L = log_odds.shape[1]

    scores = []
    for _ in range(n_samples):
        seq = "".join(rng.choice(nucs, size=L + 50, p=probs))
        s = _score_sequence(seq, log_odds)
        if s > float("-inf"):
            scores.append(s)

    if not scores:
        return 0.0
    scores.sort()
    idx = int(len(scores) * (1 - p_cutoff))
    return scores[min(idx, len(scores) - 1)]


def load_sequences(filepath: str | Path, max_length: int | None = None) -> dict[str, str]:
    """Load FASTA sequences, optionally truncating to max_length."""
    sequences = {}
    for record in SeqIO.parse(filepath, "fasta"):
        gene_id = record.id.split(".")[0]  # Remove gene model suffix
        seq = str(record.seq).upper()
        if max_length and len(seq) > max_length:
            seq = seq[:max_length]
        sequences[gene_id] = seq
    return sequences


def scan_tfbs(
    tf_motif_file: str | Path,
    pwm_file: str | Path,
    promoter_file: str | Path,
    gene_seq_file: str | Path,
    th_pre_tss: int = 1000,
    th_post_tss: int = 200,
    bg: dict[str, float] | None = None,
    p_cutoff: float = 0.05,
    correction: str = "bonferroni",
) -> tuple[list[str], list[str], np.ndarray]:
    """Scan for TF binding sites in promoter and gene body sequences.

    Port of transcription_factor_binding() from transcription_factor_binding.R.

    Returns (tf_names, gene_names, binary_matrix) where matrix[i,j] = 1 if
    TF i has a motif match in gene j's regulatory region.
    """
    if bg is None:
        bg = {"A": 0.3253439, "C": 0.1746561, "G": 0.1746561, "T": 0.3253439}

    # Load data
    tf_motifs = load_tf_motif_mapping(tf_motif_file)
    pwms = load_pwms(pwm_file)
    promoter_seqs = load_sequences(promoter_file, max_length=th_pre_tss)
    gene_seqs = load_sequences(gene_seq_file, max_length=th_post_tss)

    tfs = sorted(tf_motifs.keys())
    genes = sorted(set(promoter_seqs.keys()) | set(gene_seqs.keys()))

    logger.info("TFBS scanning: %d TFs with motifs, %d genes", len(tfs), len(genes))

    # Bonferroni correction
    if correction == "bonferroni":
        effective_p = p_cutoff / len(genes)
    else:
        effective_p = p_cutoff

    # Pre-compute log-odds and thresholds for each motif
    motif_log_odds = {}
    motif_thresholds = {}
    for motif_id, pwm in pwms.items():
        lo = _pwm_to_log_odds(pwm, bg)
        motif_log_odds[motif_id] = lo
        motif_thresholds[motif_id] = _compute_score_threshold(lo, effective_p, bg=bg)

    gene_idx = {g: i for i, g in enumerate(genes)}
    mat = np.zeros((len(tfs), len(genes)), dtype=np.int8)

    for i, tf in enumerate(tfs):
        if (i + 1) % 100 == 0:
            logger.info("  TFBS: TF %d/%d", i + 1, len(tfs))

        motif_ids = tf_motifs.get(tf, [])
        if not motif_ids:
            continue

        for j, gene in enumerate(genes):
            matched = False

            # Check promoter (pre-TSS)
            if gene in promoter_seqs:
                for mid in motif_ids:
                    if mid not in motif_log_odds:
                        continue
                    score = _score_sequence(promoter_seqs[gene], motif_log_odds[mid])
                    if score >= motif_thresholds[mid]:
                        matched = True
                        break

            # Check gene body (post-TSS)
            if not matched and gene in gene_seqs:
                for mid in motif_ids:
                    if mid not in motif_log_odds:
                        continue
                    score = _score_sequence(gene_seqs[gene], motif_log_odds[mid])
                    if score >= motif_thresholds[mid]:
                        matched = True
                        break

            if matched:
                mat[i, j] = 1

    logger.info("TFBS matrix: %d non-zero entries (%.2f%%)", mat.sum(), mat.sum() / mat.size * 100)
    return tfs, genes, mat


# ---------------------------------------------------------------------------
# Continuous PWM scoring — richer features per TF-gene pair
# ---------------------------------------------------------------------------

def _score_all_hits(
    seq_str: str,
    log_odds: np.ndarray,
    min_score: float = 0.0,
) -> list[tuple[float, int, bool]]:
    """Return all motif hits above min_score as (score, position, is_forward)."""
    nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = log_odds.shape[1]
    hits = []

    if len(seq_str) < L:
        return hits

    rc_log_odds = log_odds[::-1, ::-1]

    for strand, lo in [(True, log_odds), (False, rc_log_odds)]:
        for i in range(len(seq_str) - L + 1):
            score = 0.0
            valid = True
            for j, nuc in enumerate(seq_str[i : i + L]):
                if nuc in nuc_map:
                    score += lo[nuc_map[nuc], j]
                else:
                    valid = False
                    break
            if valid and score >= min_score:
                hits.append((score, i, strand))

    return hits


def scan_tfbs_continuous(
    tf_motif_file: str | Path,
    pwm_file: str | Path,
    promoter_file: str | Path,
    gene_seq_file: str | Path,
    th_pre_tss: int = 1000,
    th_post_tss: int = 200,
    bg: dict[str, float] | None = None,
    n_features: int = 4,
) -> tuple[list[str], list[str], np.ndarray]:
    """Compute continuous TFBS features per TF-gene pair.

    Instead of a binary match, returns a multi-dimensional feature matrix
    capturing the strength and distribution of motif matches.

    Features per TF-gene pair (n_features=4):
        0: best_score — max log-odds score across all motif hits (both strands)
        1: n_hits — number of hits above the 50th percentile threshold
        2: best_position — position of best hit relative to TSS (0=at TSS, 1=distal)
        3: score_percentile — rank of this gene's best score among all genes for this TF

    Parameters
    ----------
    tf_motif_file, pwm_file, promoter_file, gene_seq_file : path
        Same as scan_tfbs().
    th_pre_tss, th_post_tss : int
        Sequence lengths.
    bg : dict, optional
        Background nucleotide frequencies.
    n_features : int
        Number of features per TF-gene pair.

    Returns
    -------
    tf_names : list[str]
    gene_names : list[str]
    features : np.ndarray of shape (n_tfs, n_genes, n_features)
    """
    if bg is None:
        bg = {"A": 0.3253439, "C": 0.1746561, "G": 0.1746561, "T": 0.3253439}

    tf_motifs = load_tf_motif_mapping(tf_motif_file)
    pwms = load_pwms(pwm_file)
    promoter_seqs = load_sequences(promoter_file, max_length=th_pre_tss)
    gene_seqs = load_sequences(gene_seq_file, max_length=th_post_tss)

    tfs = sorted(tf_motifs.keys())
    genes = sorted(set(promoter_seqs.keys()) | set(gene_seqs.keys()))
    n_tfs = len(tfs)
    n_genes = len(genes)

    logger.info("Continuous TFBS scoring: %d TFs, %d genes", n_tfs, n_genes)

    # Pre-compute log-odds for each motif
    motif_log_odds = {}
    for motif_id, pwm in pwms.items():
        motif_log_odds[motif_id] = _pwm_to_log_odds(pwm, bg)

    # Compute raw best scores per TF-gene
    best_scores = np.full((n_tfs, n_genes), -np.inf)
    hit_counts = np.zeros((n_tfs, n_genes), dtype=np.int32)
    best_positions = np.zeros((n_tfs, n_genes))
    seq_length = th_pre_tss + th_post_tss

    for i, tf in enumerate(tfs):
        if (i + 1) % 100 == 0:
            logger.info("  TFBS features: TF %d/%d", i + 1, n_tfs)

        motif_ids = tf_motifs.get(tf, [])
        if not motif_ids:
            continue

        for j, gene in enumerate(genes):
            all_hits = []

            # Scan promoter
            if gene in promoter_seqs:
                for mid in motif_ids:
                    if mid not in motif_log_odds:
                        continue
                    hits = _score_all_hits(promoter_seqs[gene], motif_log_odds[mid])
                    all_hits.extend(hits)

            # Scan gene body
            if gene in gene_seqs:
                for mid in motif_ids:
                    if mid not in motif_log_odds:
                        continue
                    hits = _score_all_hits(gene_seqs[gene], motif_log_odds[mid])
                    # Offset positions by promoter length
                    all_hits.extend((s, p + th_pre_tss, fw) for s, p, fw in hits)

            if all_hits:
                scores = [h[0] for h in all_hits]
                best_idx = int(np.argmax(scores))
                best_scores[i, j] = scores[best_idx]
                hit_counts[i, j] = len(all_hits)
                best_positions[i, j] = all_hits[best_idx][1] / seq_length

    # Compute per-TF percentile ranks
    score_percentiles = np.zeros((n_tfs, n_genes))
    for i in range(n_tfs):
        valid = best_scores[i] > -np.inf
        if valid.sum() > 0:
            from scipy.stats import rankdata
            ranks = np.zeros(n_genes)
            ranks[valid] = rankdata(best_scores[i, valid]) / valid.sum()
            score_percentiles[i] = ranks

    # Replace -inf with 0 for best_scores
    best_scores[best_scores == -np.inf] = 0.0

    # Stack features
    features = np.stack([
        best_scores,
        np.log1p(hit_counts.astype(np.float64)),  # log-transform counts
        best_positions,
        score_percentiles,
    ], axis=-1)  # (n_tfs, n_genes, 4)

    logger.info(
        "Continuous TFBS features: shape %s, genes with hits: %d/%d",
        features.shape,
        (best_scores > 0).any(axis=0).sum(), n_genes,
    )
    return tfs, genes, features
