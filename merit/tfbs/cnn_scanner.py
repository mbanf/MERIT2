"""CNN-based motif scanning — learns motif grammar from sequence.

Instead of matching fixed PWMs, this module trains a CNN to predict TF binding
from raw DNA sequence. Inspired by DeepBind / BPNet.

Architecture: One-hot encoded sequence -> Conv layers (multiple kernel sizes)
              -> Global max-pool -> Dense -> binding probability per TF
"""

from __future__ import annotations

import logging

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


def scan_tfbs_cnn(
    promoter_sequences: dict[str, str],
    tf_names: list[str],
    positive_examples: dict[str, list[str]] | None = None,
    n_filters: int = 64,
    kernel_sizes: list[int] | None = None,
    learning_rate: float = 0.001,
    epochs: int = 50,
    max_seq_len: int = 1000,
    device: str = "cpu",
) -> np.ndarray:
    """Predict TF binding sites using a CNN.

    Parameters
    ----------
    promoter_sequences : dict[str, str]
        Gene ID -> promoter sequence.
    tf_names : list[str]
        TF names to predict binding for.
    positive_examples : dict[str, list[str]], optional
        TF -> list of known target gene IDs (for supervised training).
    n_filters : int
        Number of convolutional filters per kernel size.
    kernel_sizes : list[int]
        Convolutional kernel sizes (motif lengths to learn).
    epochs : int
        Training epochs.

    Returns
    -------
    np.ndarray
        Binary matrix (n_tfs x n_genes) of predicted binding.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            "CNN TFBS scanning requires PyTorch. Install with: pip install 'merit[deep]'"
        )

    if kernel_sizes is None:
        kernel_sizes = [7, 11, 15]

    genes = sorted(promoter_sequences.keys())
    n_genes = len(genes)
    n_tfs = len(tf_names)

    logger.info("CNN TFBS: %d genes, %d TFs, kernels=%s", n_genes, n_tfs, kernel_sizes)

    # Encode all sequences
    X = np.stack([one_hot_encode(promoter_sequences[g], max_seq_len) for g in genes])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    class MotifCNN(nn.Module):
        def __init__(self, n_tfs, n_filters, kernel_sizes, seq_len):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Conv1d(4, n_filters, k, padding=k // 2)
                for k in kernel_sizes
            ])
            self.fc = nn.Sequential(
                nn.Linear(n_filters * len(kernel_sizes), 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_tfs),
            )

        def forward(self, x):
            # x: (batch, 4, seq_len)
            conv_outs = []
            for conv in self.convs:
                h = F.relu(conv(x))
                h = h.max(dim=2)[0]  # Global max-pool
                conv_outs.append(h)
            h = torch.cat(conv_outs, dim=1)
            return torch.sigmoid(self.fc(h))

    model = MotifCNN(n_tfs, n_filters, kernel_sizes, max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if positive_examples:
        # Build label matrix
        gene_idx = {g: i for i, g in enumerate(genes)}
        tf_idx = {t: i for i, t in enumerate(tf_names)}
        Y = np.zeros((n_genes, n_tfs), dtype=np.float32)
        for tf, targets in positive_examples.items():
            if tf in tf_idx:
                for tg in targets:
                    if tg in gene_idx:
                        Y[gene_idx[tg], tf_idx[tf]] = 1.0

        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

        logger.info("Training CNN on %d positive examples...", int(Y.sum()))
        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for xb, yb in loader:
                pred = model(xb)
                loss = F.binary_cross_entropy(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, epochs, total_loss / len(loader))

    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()  # (n_genes, n_tfs)

    # Transpose to (n_tfs, n_genes) and binarize at 0.5
    mat = (predictions.T > 0.5).astype(np.int8)
    logger.info("CNN predictions: %d binding events", mat.sum())
    return mat
