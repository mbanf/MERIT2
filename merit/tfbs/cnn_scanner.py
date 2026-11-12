"""CNN-based motif scanning with PWM-initialized convolutional filters.

Architecture:
    1. Conv1d filters initialized from known PWMs (CIS-BP) + learnable random filters
    2. Train on DAP-seq / ChIP-seq binding data to refine motifs and learn grammar
    3. Output continuous binding probability per TF-gene pair

The key insight: start from known biology (PWMs as conv filter initializations)
and let the network learn what static PWMs miss — motif context, spacing rules,
co-factor binding preferences.

Inspired by DeepBind (Alipanahi et al. 2015) and BPNet (Avsec et al. 2021).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(seq: str, max_len: int = 1000) -> np.ndarray:
    """One-hot encode a DNA sequence. Returns (4, max_len) array."""
    encoded = np.zeros((4, max_len), dtype=np.float32)
    for i, nuc in enumerate(seq[:max_len]):
        if nuc in _NUC_TO_IDX:
            encoded[_NUC_TO_IDX[nuc], i] = 1.0
    return encoded


def _pwm_to_conv_filter(pwm: np.ndarray) -> np.ndarray:
    """Convert PWM (4, L) probability matrix to log-odds Conv1d filter weights."""
    bg = 0.25
    pseudocount = 0.001
    pwm_pc = pwm + pseudocount
    pwm_pc = pwm_pc / pwm_pc.sum(axis=0, keepdims=True)
    return np.log2(pwm_pc / bg).astype(np.float32)


def build_cnn_scanner(
    pwms: dict[str, np.ndarray],
    tf_names: list[str],
    n_random_filters: int = 256,
    random_kernel_sizes: list[int] | None = None,
    device: str = "cpu",
):
    """Build a MotifCNN with PWM-initialized + random convolutional filters.

    Parameters
    ----------
    pwms : dict[str, np.ndarray]
        Motif ID -> PWM matrix (4, L).
    tf_names : list[str]
        TFs to predict binding for.
    n_random_filters : int
        Additional randomly initialized filters for learning novel motifs.
    random_kernel_sizes : list[int]
        Kernel sizes for random filters.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    model : nn.Module (MotifCNN)
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "CNN TFBS scanning requires PyTorch. Install with: pip install 'merit[deep]'"
        )

    if random_kernel_sizes is None:
        random_kernel_sizes = [7, 11, 15, 21]

    # Group PWMs by length
    pwm_by_length: dict[int, list[tuple[str, np.ndarray]]] = {}
    for mid, pwm in pwms.items():
        L = pwm.shape[1]
        if L not in pwm_by_length:
            pwm_by_length[L] = []
        pwm_by_length[L].append((mid, pwm))

    pwm_lengths = sorted(pwm_by_length.keys())
    n_tfs = len(tf_names)

    logger.info(
        "Building MotifCNN: %d PWM groups (%d total filters), %d random filters, %d TFs",
        len(pwm_lengths), sum(len(v) for v in pwm_by_length.values()),
        n_random_filters, n_tfs,
    )

    class MotifCNN(nn.Module):
        """CNN with PWM-initialized + random convolutional filters.

        PWM filters start from known motif biology. Random filters can
        learn novel sequence patterns (cofactor motifs, spacing rules).
        Both are refined during training on binding data.
        """

        def __init__(self):
            super().__init__()

            self.pwm_convs = nn.ModuleDict()
            self._pwm_filter_names: dict[int, list[str]] = {}
            total_pwm_filters = 0

            for L in pwm_lengths:
                motifs = pwm_by_length[L]
                n_f = len(motifs)
                conv = nn.Conv1d(4, n_f, L, padding=L // 2, bias=False)

                with torch.no_grad():
                    for fi, (mid, pwm) in enumerate(motifs):
                        conv.weight[fi] = torch.tensor(_pwm_to_conv_filter(pwm))

                self.pwm_convs[f"k{L}"] = conv
                self._pwm_filter_names[L] = [mid for mid, _ in motifs]
                total_pwm_filters += n_f

            n_per_kernel = max(1, n_random_filters // len(random_kernel_sizes))
            self.random_convs = nn.ModuleList([
                nn.Conv1d(4, n_per_kernel, k, padding=k // 2)
                for k in random_kernel_sizes
            ])
            total_random = n_per_kernel * len(random_kernel_sizes)

            self._total_filters = total_pwm_filters + total_random

            self.head = nn.Sequential(
                nn.Linear(self._total_filters, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_tfs),
            )

        def forward(self, x):
            """x: (batch, 4, seq_len) -> (batch, n_tfs) binding probabilities."""
            pooled = []
            for conv in self.pwm_convs.values():
                h = torch.relu(conv(x))
                pooled.append(h.max(dim=2)[0])
            for conv in self.random_convs:
                h = torch.relu(conv(x))
                pooled.append(h.max(dim=2)[0])
            h = torch.cat(pooled, dim=1)
            return torch.sigmoid(self.head(h))

        def get_learned_motifs(self) -> dict[str, np.ndarray]:
            """Extract learned filter weights for interpretation."""
            motifs = {}
            for key, conv in self.pwm_convs.items():
                L = int(key[1:])
                w = conv.weight.detach().cpu().numpy()
                for fi, mid in enumerate(self._pwm_filter_names.get(L, [])):
                    motifs[f"pwm_{mid}"] = w[fi]
            for ci, conv in enumerate(self.random_convs):
                w = conv.weight.detach().cpu().numpy()
                for fi in range(w.shape[0]):
                    motifs[f"learned_{ci}_{fi}"] = w[fi]
            return motifs

    model = MotifCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("MotifCNN: %d filters, %d parameters", model._total_filters, n_params)

    return model


def train_cnn(
    model,
    sequences: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> list[float]:
    """Train the CNN on binding labels.

    Parameters
    ----------
    model : MotifCNN
    sequences : np.ndarray of shape (n_genes, 4, seq_len)
        One-hot encoded promoter sequences.
    labels : np.ndarray of shape (n_genes, n_tfs)
        Binary binding labels (1 = bound, 0 = not bound).
    mask : np.ndarray of shape (n_genes, n_tfs), optional
        1 = known label, 0 = unknown (loss ignored).
        If None, all labels are treated as known.
    epochs : int
    batch_size : int
    learning_rate : float
    device : str

    Returns
    -------
    losses : list[float] — per-epoch average loss
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(sequences, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32)

    if mask is not None:
        M = torch.tensor(mask, dtype=torch.float32)
        dataset = TensorDataset(X, Y, M)
    else:
        M = None
        dataset = TensorDataset(X, Y)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device)

            pred = model(xb)

            if M is not None:
                mb = batch[2].to(device)
                loss = F.binary_cross_entropy(pred, yb, weight=mb, reduction="sum")
                loss = loss / (mb.sum() + 1e-8)
            else:
                loss = F.binary_cross_entropy(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, epochs, avg_loss)

    return losses


def predict_binding(
    model,
    sequences: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Predict binding probabilities. Returns (n_genes, n_tfs) array."""
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            xb = torch.tensor(
                sequences[start:start + batch_size], dtype=torch.float32
            ).to(device)
            preds.append(model(xb).cpu().numpy())

    return np.vstack(preds)


def scan_tfbs_cnn(
    pwm_file: str | Path,
    promoter_file: str | Path,
    tf_names: list[str],
    gold_edges: list[tuple[str, str]] | None = None,
    n_random_filters: int = 256,
    max_seq_len: int = 1000,
    epochs: int = 50,
    device: str = "cpu",
) -> tuple[list[str], list[str], np.ndarray]:
    """End-to-end CNN TFBS scanning: build, optionally train, predict.

    Parameters
    ----------
    pwm_file : path
        PWM file (MERIT format) for filter initialization.
    promoter_file : path
        Promoter FASTA sequences.
    tf_names : list[str]
        TFs to predict binding for.
    gold_edges : list of (tf_id, gene_id) tuples, optional
        Known binding interactions for supervised training (DAP-seq, ChIP-seq).
        If None, returns untrained PWM-initialized predictions.
    n_random_filters : int
        Additional learnable filters beyond PWMs.
    max_seq_len : int
        Input sequence length.
    epochs : int
        Training epochs (if gold_edges provided).
    device : str
        "cpu" or "cuda".

    Returns
    -------
    tf_names : list[str]
    gene_names : list[str]
    binding_probs : np.ndarray of shape (n_tfs, n_genes)
        Continuous binding probabilities (0-1).
    """
    from .pwm_scanner import load_pwms, load_sequences

    pwms = load_pwms(pwm_file)
    promoter_seqs = load_sequences(promoter_file, max_length=max_seq_len)
    gene_names = sorted(promoter_seqs.keys())

    logger.info("CNN TFBS: %d PWMs, %d genes, %d TFs", len(pwms), len(gene_names), len(tf_names))

    # Build model
    model = build_cnn_scanner(pwms, tf_names, n_random_filters=n_random_filters, device=device)

    # Encode sequences
    X = np.stack([one_hot_encode(promoter_seqs.get(g, ""), max_seq_len) for g in gene_names])

    # Train if gold-standard edges provided
    if gold_edges:
        gene_idx = {g: i for i, g in enumerate(gene_names)}
        tf_idx = {t: i for i, t in enumerate(tf_names)}

        Y = np.zeros((len(gene_names), len(tf_names)), dtype=np.float32)
        mask = np.zeros_like(Y)

        for tf, gene in gold_edges:
            if tf in tf_idx and gene in gene_idx:
                Y[gene_idx[gene], tf_idx[tf]] = 1.0
                mask[gene_idx[gene], tf_idx[tf]] = 1.0

        # Sample negatives (3:1 ratio)
        rng = np.random.RandomState(42)
        for ti, tf in enumerate(tf_names):
            pos_genes = {g for t, g in gold_edges if t == tf and g in gene_idx}
            neg_pool = [g for g in gene_names if g not in pos_genes]
            n_neg = min(len(pos_genes) * 3, len(neg_pool))
            if n_neg > 0:
                for g in rng.choice(neg_pool, size=n_neg, replace=False):
                    mask[gene_idx[g], ti] = 1.0

        logger.info("Training on %d positive, %d total labeled edges", int(Y.sum()), int(mask.sum()))
        train_cnn(model, X, Y, mask, epochs=epochs, device=device)

    # Predict
    predictions = predict_binding(model, X, device=device)
    binding_probs = predictions.T  # (n_tfs, n_genes)

    logger.info("CNN binding: mean=%.4f, >0.5: %d edges", binding_probs.mean(), (binding_probs > 0.5).sum())
    return tf_names, gene_names, binding_probs
