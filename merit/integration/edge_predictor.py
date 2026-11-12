"""Edge predictor: integrates multi-source evidence into edge probabilities.

This is the core v2.0 component that replaces the hard lead-support
thresholding from v1.0. It takes continuous edge features from:
- RF, LASSO, CLR inference scores + Monte Carlo confidence
- PWM continuous features or CNN binding probabilities
- (optionally) Gold-standard priors

And learns to predict edge probability using either:
1. An MLP edge classifier (fast, MLX-compatible)
2. A Graph Attention Network (captures neighborhood structure)

The predictor is trained on gold-standard edges (ATRM, AtRegNet, DAP-seq).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EdgePrediction:
    """Result of edge prediction."""

    probabilities: np.ndarray  # (n_tfs, n_genes) edge probabilities
    tf_names: list[str]
    gene_names: list[str]

    def to_binary(self, threshold: float = 0.5) -> np.ndarray:
        return (self.probabilities >= threshold).astype(np.int8)

    def to_edge_list(self, threshold: float = 0.5) -> list[tuple[str, str, float]]:
        edges = []
        for i, tf in enumerate(self.tf_names):
            for j, gene in enumerate(self.gene_names):
                p = self.probabilities[i, j]
                if p >= threshold:
                    edges.append((tf, gene, float(p)))
        return edges


def train_edge_predictor_mlx(
    edge_features: np.ndarray,
    gold_standard_matrix: np.ndarray,
    mask: np.ndarray | None = None,
    hidden_dims: list[int] | None = None,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 4096,
) -> tuple:
    """Train an MLP edge predictor using MLX (Apple Silicon GPU).

    Parameters
    ----------
    edge_features : np.ndarray of shape (n_tfs, n_genes, n_features)
        Per-edge feature vectors.
    gold_standard_matrix : np.ndarray of shape (n_tfs, n_genes)
        Binary gold-standard labels (1 = known edge, 0 = no edge).
    mask : np.ndarray of shape (n_tfs, n_genes), optional
        1 = labeled (use in loss), 0 = unknown (skip).
        If None, all entries are used.
    hidden_dims : list[int]
        MLP hidden layer sizes.
    learning_rate : float
    epochs : int
    batch_size : int

    Returns
    -------
    (model, loss_fn, losses) — trained MLX model and training losses.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    if hidden_dims is None:
        hidden_dims = [128, 64, 32]

    n_tfs, n_genes, n_features = edge_features.shape

    # Flatten to (n_edges, n_features)
    X = edge_features.reshape(-1, n_features)
    Y = gold_standard_matrix.reshape(-1)

    if mask is not None:
        M = mask.reshape(-1)
    else:
        # Build mask: use all positives + sample negatives (10:1 ratio)
        rng = np.random.RandomState(42)
        M = np.zeros_like(Y)
        pos_idx = np.where(Y > 0)[0]
        M[pos_idx] = 1.0
        neg_idx = np.where(Y == 0)[0]
        n_neg = min(len(pos_idx) * 10, len(neg_idx))
        neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
        M[neg_sample] = 1.0

    # Filter to labeled examples only
    labeled_idx = np.where(M > 0)[0]
    X_labeled = X[labeled_idx]
    Y_labeled = Y[labeled_idx]

    n_pos = int(Y_labeled.sum())
    n_neg = len(Y_labeled) - n_pos
    logger.info(
        "Training edge predictor: %d features, %d labeled (%d pos, %d neg)",
        n_features, len(Y_labeled), n_pos, n_neg,
    )

    # Convert to MLX arrays
    X_mx = mx.array(X_labeled.astype(np.float32))
    Y_mx = mx.array(Y_labeled.astype(np.float32))

    # Positive class weight for imbalanced data
    pos_weight = n_neg / max(n_pos, 1)

    # Build model
    layers = []
    in_dim = n_features
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))

    model = nn.Sequential(*layers)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model, x, y):
        logits = model(x).squeeze(-1)
        # Weighted BCE
        pos_mask = y > 0.5
        neg_mask = ~pos_mask
        bce = -y * mx.log(mx.sigmoid(logits) + 1e-7) * pos_weight \
              - (1 - y) * mx.log(1 - mx.sigmoid(logits) + 1e-7)
        return mx.mean(bce)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    losses = []
    n_samples = len(Y_labeled)

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_mx[mx.array(idx)]
            yb = Y_mx[mx.array(idx)]

            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            # Compute training accuracy
            model.eval()
            logits = model(X_mx).squeeze(-1)
            preds = mx.sigmoid(logits) > 0.5
            acc = mx.mean(preds == (Y_mx > 0.5)).item()
            model.train()
            logger.info("  Epoch %d/%d, loss=%.4f, acc=%.3f", epoch + 1, epochs, avg_loss, acc)

    logger.info("Training complete. Final loss: %.4f", losses[-1])
    return model, losses


def predict_edges_mlx(
    model,
    edge_features: np.ndarray,
) -> np.ndarray:
    """Predict edge probabilities using trained MLX model.

    Parameters
    ----------
    model : MLX nn.Sequential
    edge_features : np.ndarray of shape (n_tfs, n_genes, n_features)

    Returns
    -------
    np.ndarray of shape (n_tfs, n_genes) — edge probabilities.
    """
    import mlx.core as mx

    n_tfs, n_genes, n_features = edge_features.shape
    X = edge_features.reshape(-1, n_features).astype(np.float32)

    model.eval()
    # Predict in chunks to avoid memory issues
    chunk_size = 100_000
    probs = []
    for start in range(0, len(X), chunk_size):
        xb = mx.array(X[start:start + chunk_size])
        logits = model(xb).squeeze(-1)
        p = mx.sigmoid(logits)
        probs.append(np.array(p))

    all_probs = np.concatenate(probs)
    return all_probs.reshape(n_tfs, n_genes)


def train_edge_predictor_torch(
    edge_features: np.ndarray,
    gold_standard_matrix: np.ndarray,
    mask: np.ndarray | None = None,
    hidden_dims: list[int] | None = None,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 4096,
    device: str = "cpu",
) -> tuple:
    """Train an MLP edge predictor using PyTorch (fallback for non-Mac).

    Same interface as train_edge_predictor_mlx but uses PyTorch.
    """
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if hidden_dims is None:
        hidden_dims = [128, 64, 32]

    n_tfs, n_genes, n_features = edge_features.shape
    X = edge_features.reshape(-1, n_features).astype(np.float32)
    Y = gold_standard_matrix.reshape(-1).astype(np.float32)

    if mask is not None:
        M = mask.reshape(-1)
    else:
        rng = np.random.RandomState(42)
        M = np.zeros_like(Y)
        pos_idx = np.where(Y > 0)[0]
        M[pos_idx] = 1.0
        neg_idx = np.where(Y == 0)[0]
        n_neg = min(len(pos_idx) * 10, len(neg_idx))
        M[neg_idx[rng.choice(len(neg_idx), size=n_neg, replace=False)]] = 1.0

    labeled_idx = np.where(M > 0)[0]
    X_l = torch.tensor(X[labeled_idx]).to(device)
    Y_l = torch.tensor(Y[labeled_idx]).to(device)

    n_pos = int(Y_l.sum().item())
    n_neg = len(Y_l) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

    layers = []
    in_dim = n_features
    for h in hidden_dims:
        layers.extend([tnn.Linear(in_dim, h), tnn.ReLU(), tnn.Dropout(0.2)])
        in_dim = h
    layers.append(tnn.Linear(in_dim, 1))
    model = tnn.Sequential(*layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(X_l, Y_l)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            logits = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

        if (epoch + 1) % 20 == 0:
            logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, epochs, losses[-1])

    return model, losses
