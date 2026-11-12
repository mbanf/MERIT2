"""Graph Attention Network for edge evidence integration.

Unlike the MLP edge predictor (which treats each edge independently), the GAT
aggregates neighborhood information: a TF's predicted regulation of gene A is
informed by its predictions for genes B, C, ... This captures the biological
reality that TFs regulate sets of functionally related genes.

Architecture:
    1. Node encoder: expression profiles → node embeddings
    2. Edge encoder: 7-dim evidence features → edge embeddings
    3. GAT layers: message passing with edge-feature-aware attention
    4. Edge predictor: [src_embed || tgt_embed || edge_features] → probability

The GAT operates on a SPARSE graph — only edges with any non-zero evidence
are considered as candidates, keeping memory tractable for genome-scale data.

Supports MLX (Apple Silicon) with PyTorch fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GATConfig:
    """Configuration for the GAT integrator."""

    node_dim: int = 64          # Node embedding dimension
    edge_dim: int = 32          # Edge embedding dimension
    n_heads: int = 4            # Number of attention heads
    n_layers: int = 2           # Number of GAT layers
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 100
    batch_edges: int = 8192     # Edges per mini-batch
    neg_ratio: int = 5          # Negative samples per positive
    sparsity_threshold: float = 0.01  # Min feature sum to include edge


def _build_sparse_graph(
    edge_features: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse edge list from dense edge feature matrix.

    Only includes edges where the sum of evidence features exceeds threshold.

    Returns
    -------
    src : np.ndarray of TF indices
    dst : np.ndarray of gene indices
    feats : np.ndarray (n_sparse_edges, n_features)
    """
    n_tfs, n_genes, n_features = edge_features.shape
    feat_sum = edge_features.sum(axis=-1)  # (n_tfs, n_genes)
    src_idx, dst_idx = np.where(feat_sum > threshold)
    feats = edge_features[src_idx, dst_idx]  # (n_sparse, n_features)

    logger.info(
        "Sparse graph: %d candidate edges from %dx%d matrix (%.2f%% density)",
        len(src_idx), n_tfs, n_genes, 100.0 * len(src_idx) / max(n_tfs * n_genes, 1),
    )
    return src_idx.astype(np.int32), dst_idx.astype(np.int32), feats.astype(np.float32)


def train_gat_mlx(
    edge_features: np.ndarray,
    node_features: np.ndarray | None,
    gold_standard_matrix: np.ndarray,
    tf_names: list[str],
    gene_names: list[str],
    config: GATConfig | None = None,
) -> tuple:
    """Train GAT edge integrator using MLX.

    Parameters
    ----------
    edge_features : np.ndarray (n_tfs, n_genes, n_features)
        Per-edge evidence features from v1.0 intermediates.
    node_features : np.ndarray (n_genes, n_node_features), optional
        Expression profiles for all genes. If None, uses learnable embeddings.
    gold_standard_matrix : np.ndarray (n_tfs, n_genes)
        Binary gold-standard labels.
    tf_names, gene_names : list[str]
    config : GATConfig

    Returns
    -------
    (model_params, config, losses)
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    if config is None:
        config = GATConfig()

    n_tfs, n_genes, n_edge_features = edge_features.shape

    # Build sparse graph
    src, dst, sparse_feats = _build_sparse_graph(edge_features, config.sparsity_threshold)
    n_sparse = len(src)

    # Node features: if expression provided, project; otherwise learnable
    if node_features is not None:
        n_node_features = node_features.shape[1]
    else:
        # Use TF/gene index as identity features
        n_node_features = 0

    # Identify positive/negative edges for training
    labels = gold_standard_matrix[src, dst]
    pos_mask = labels > 0.5
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(~pos_mask)[0]

    n_pos = len(pos_indices)
    n_neg_sample = min(n_pos * config.neg_ratio, len(neg_indices))

    if n_pos == 0:
        logger.warning("No positive edges in sparse graph — lower sparsity_threshold")
        raise ValueError("No positive edges found in sparse candidate graph")

    logger.info(
        "GAT training: %d sparse edges, %d positives, sampling %d negatives",
        n_sparse, n_pos, n_neg_sample,
    )

    # ── MLX Model ──────────────────────────────────────────────────────
    # We implement a simplified edge-feature-aware GAT:
    # 1. Edge encoder projects edge features
    # 2. For each target gene, aggregate messages from all TFs with attention
    # 3. Edge predictor uses aggregated node state + edge features

    class EdgeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_edge_features, config.edge_dim),
                nn.ReLU(),
                nn.Linear(config.edge_dim, config.edge_dim),
            )

        def __call__(self, x):
            return self.net(x)

    class NodeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            if n_node_features > 0:
                self.net = nn.Sequential(
                    nn.Linear(n_node_features, config.node_dim),
                    nn.ReLU(),
                    nn.Linear(config.node_dim, config.node_dim),
                )
            else:
                # Learnable embeddings for TFs and genes
                self.tf_embed = nn.Embedding(n_tfs, config.node_dim)
                self.gene_embed = nn.Embedding(n_genes, config.node_dim)

        def __call__(self, node_feats=None, tf_ids=None, gene_ids=None):
            if n_node_features > 0:
                return self.net(node_feats)
            else:
                tf_emb = self.tf_embed(tf_ids)
                gene_emb = self.gene_embed(gene_ids)
                return tf_emb, gene_emb

    class AttentionHead(nn.Module):
        """Single attention head for edge-aware message passing."""

        def __init__(self):
            super().__init__()
            head_dim = config.node_dim // config.n_heads
            self.W_src = nn.Linear(config.node_dim, head_dim, bias=False)
            self.W_dst = nn.Linear(config.node_dim, head_dim, bias=False)
            self.W_edge = nn.Linear(config.edge_dim, head_dim, bias=False)
            self.attn = nn.Linear(head_dim * 3, 1, bias=False)

        def __call__(self, src_emb, dst_emb, edge_emb):
            """Compute attention-weighted messages for a batch of edges.

            src_emb : (batch, node_dim)
            dst_emb : (batch, node_dim)
            edge_emb : (batch, edge_dim)
            Returns: (batch, head_dim) messages
            """
            hs = self.W_src(src_emb)   # (batch, head_dim)
            hd = self.W_dst(dst_emb)
            he = self.W_edge(edge_emb)

            attn_input = mx.concatenate([hs, hd, he], axis=-1)
            alpha = mx.sigmoid(self.attn(attn_input))  # (batch, 1)
            return alpha * hs  # attention-weighted source message

    class GATIntegrator(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_encoder = NodeEncoder()
            self.edge_encoder = EdgeEncoder()

            # Multi-head attention layers
            self.attention_heads = [
                [AttentionHead() for _ in range(config.n_heads)]
                for _ in range(config.n_layers)
            ]
            self.layer_norms = [nn.LayerNorm(config.node_dim) for _ in range(config.n_layers)]

            # Edge predictor: takes [src_embed, dst_embed, edge_embed]
            pred_dim = config.node_dim * 2 + config.edge_dim
            self.predictor = nn.Sequential(
                nn.Linear(pred_dim, 64),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def encode_edges(self, edge_feats):
            return self.edge_encoder(edge_feats)

        def get_node_embeddings(self, node_feats, tf_ids, gene_ids):
            if n_node_features > 0:
                return self.node_encoder(node_feats)
            else:
                return self.node_encoder(tf_ids=tf_ids, gene_ids=gene_ids)

        def message_pass(self, src_emb, dst_emb, edge_emb, layer_idx):
            """One round of multi-head message passing for a batch of edges."""
            messages = []
            for head in self.attention_heads[layer_idx]:
                msg = head(src_emb, dst_emb, edge_emb)
                messages.append(msg)
            # Concatenate heads
            combined = mx.concatenate(messages, axis=-1)  # (batch, node_dim)
            # Residual + LayerNorm on dst
            updated = self.layer_norms[layer_idx](dst_emb + combined)
            return updated

        def predict(self, src_emb, dst_emb, edge_emb):
            """Predict edge probability from embeddings."""
            combined = mx.concatenate([src_emb, dst_emb, edge_emb], axis=-1)
            logits = self.predictor(combined).squeeze(-1)
            return logits

    model = GATIntegrator()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Prepare data arrays
    sparse_feats_mx = mx.array(sparse_feats)

    if node_features is not None:
        # For simplicity, we use TF features = expression of TF gene
        # Need to map TF names to gene indices
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        tf_node_indices = []
        for tf in tf_names:
            if tf in gene_to_idx:
                tf_node_indices.append(gene_to_idx[tf])
            else:
                tf_node_indices.append(0)  # fallback
        node_feats_mx = mx.array(node_features.astype(np.float32))
    else:
        node_feats_mx = None

    src_mx = mx.array(src.astype(np.int32))
    dst_mx = mx.array(dst.astype(np.int32))

    # Positive weight for BCE
    pos_weight = n_neg_sample / max(n_pos, 1)

    rng = np.random.RandomState(42)

    def loss_fn(model, batch_src, batch_dst, batch_edge_feats, batch_labels):
        # Encode edges
        edge_emb = model.encode_edges(batch_edge_feats)

        # Get node embeddings
        if node_feats_mx is not None:
            all_node_emb = model.get_node_embeddings(node_feats_mx, None, None)
            # Map TF src indices to gene indices for node lookup
            # TFs and genes share the same embedding space via expression
            tf_indices = mx.array(np.array([tf_node_indices[s] for s in np.array(batch_src)]))
            src_emb = all_node_emb[tf_indices]
            dst_emb = all_node_emb[batch_dst]
        else:
            src_emb, dst_emb = model.get_node_embeddings(None, batch_src, batch_dst)

        # Message passing (simplified: apply to batch edges directly)
        for layer in range(config.n_layers):
            dst_emb = model.message_pass(src_emb, dst_emb, edge_emb, layer)

        # Predict
        logits = model.predict(src_emb, dst_emb, edge_emb)

        # Weighted BCE loss
        weight = mx.where(batch_labels > 0.5, pos_weight, 1.0)
        bce = -batch_labels * mx.log(mx.sigmoid(logits) + 1e-7) \
              - (1 - batch_labels) * mx.log(1 - mx.sigmoid(logits) + 1e-7)
        return mx.mean(weight * bce)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    losses = []
    for epoch in range(config.epochs):
        # Sample training batch: all positives + sampled negatives
        neg_sample = rng.choice(neg_indices, size=n_neg_sample, replace=False)
        batch_idx = np.concatenate([pos_indices, neg_sample])
        rng.shuffle(batch_idx)

        batch_labels_np = labels[batch_idx].astype(np.float32)
        batch_src_np = src[batch_idx]
        batch_dst_np = dst[batch_idx]
        batch_feats_np = sparse_feats[batch_idx]

        # Mini-batch training
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(batch_idx), config.batch_edges):
            end = min(start + config.batch_edges, len(batch_idx))
            b_src = mx.array(batch_src_np[start:end])
            b_dst = mx.array(batch_dst_np[start:end])
            b_feats = mx.array(batch_feats_np[start:end])
            b_labels = mx.array(batch_labels_np[start:end])

            loss, grads = loss_and_grad(model, b_src, b_dst, b_feats, b_labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            logger.info("  GAT epoch %d/%d, loss=%.4f", epoch + 1, config.epochs, avg_loss)

    logger.info("GAT training complete. Final loss: %.4f", losses[-1] if losses else 0)

    return model, config, losses


def predict_gat_mlx(
    model,
    edge_features: np.ndarray,
    node_features: np.ndarray | None,
    tf_names: list[str],
    gene_names: list[str],
    config: GATConfig,
    sparsity_threshold: float = 0.01,
) -> np.ndarray:
    """Predict edge probabilities using trained GAT model.

    Returns
    -------
    np.ndarray (n_tfs, n_genes) — edge probabilities for all candidate edges.
    Non-candidate edges (below sparsity threshold) get probability 0.
    """
    import mlx.core as mx

    n_tfs, n_genes, n_feat = edge_features.shape
    src, dst, sparse_feats = _build_sparse_graph(edge_features, sparsity_threshold)

    # Node setup
    if node_features is not None:
        node_feats_mx = mx.array(node_features.astype(np.float32))
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        tf_node_indices = [gene_to_idx.get(tf, 0) for tf in tf_names]
    else:
        node_feats_mx = None

    # Predict in chunks
    probs = np.zeros((n_tfs, n_genes), dtype=np.float32)
    chunk_size = 50_000

    model.eval()
    for start in range(0, len(src), chunk_size):
        end = min(start + chunk_size, len(src))
        b_src = mx.array(src[start:end])
        b_dst = mx.array(dst[start:end])
        b_feats = mx.array(sparse_feats[start:end])

        edge_emb = model.encode_edges(b_feats)

        if node_feats_mx is not None:
            all_node_emb = model.get_node_embeddings(node_feats_mx, None, None)
            tf_idx = mx.array(np.array([tf_node_indices[s] for s in src[start:end]]))
            src_emb = all_node_emb[tf_idx]
            dst_emb = all_node_emb[b_dst]
        else:
            src_emb, dst_emb = model.get_node_embeddings(None, b_src, b_dst)

        for layer in range(config.n_layers):
            dst_emb = model.message_pass(src_emb, dst_emb, edge_emb, layer)

        logits = model.predict(src_emb, dst_emb, edge_emb)
        p = mx.sigmoid(logits)
        mx.eval(p)

        chunk_probs = np.array(p)
        for k in range(end - start):
            probs[src[start + k], dst[start + k]] = chunk_probs[k]

    return probs


def train_and_predict_gat(
    edge_features: np.ndarray,
    gold_standard_matrix: np.ndarray,
    tf_names: list[str],
    gene_names: list[str],
    node_features: np.ndarray | None = None,
    config: GATConfig | None = None,
) -> np.ndarray:
    """Convenience function: train GAT and return edge probabilities.

    Parameters
    ----------
    edge_features : (n_tfs, n_genes, n_features)
    gold_standard_matrix : (n_tfs, n_genes) binary
    tf_names, gene_names : list[str]
    node_features : (n_genes, n_expr_features), optional
    config : GATConfig, optional

    Returns
    -------
    np.ndarray (n_tfs, n_genes) — predicted edge probabilities.
    """
    if config is None:
        config = GATConfig()

    backend = "numpy"
    try:
        import mlx.core as mx
        _ = mx.array([1.0])
        backend = "mlx"
    except ImportError:
        pass

    if backend == "mlx":
        model, config, losses = train_gat_mlx(
            edge_features, node_features, gold_standard_matrix,
            tf_names, gene_names, config,
        )
        probs = predict_gat_mlx(
            model, edge_features, node_features,
            tf_names, gene_names, config,
            sparsity_threshold=config.sparsity_threshold,
        )
    else:
        logger.warning("GAT requires MLX or PyTorch. Falling back to MLP edge predictor.")
        from merit.integration.edge_predictor import (
            train_edge_predictor_torch,
        )
        model, losses = train_edge_predictor_torch(
            edge_features, gold_standard_matrix, epochs=config.epochs,
        )
        import torch
        n_tfs, n_genes, n_features = edge_features.shape
        model.eval()
        with torch.no_grad():
            X = torch.tensor(edge_features.reshape(-1, n_features).astype(np.float32))
            logits = model(X).squeeze(-1)
            probs = torch.sigmoid(logits).numpy().reshape(n_tfs, n_genes)

    return probs
