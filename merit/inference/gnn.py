"""GNN-based GRN inference — learns to predict regulatory edges from expression.

Uses a Graph Neural Network that takes gene expression profiles as node features
and learns edge probabilities supervised by gold-standard interactions (ATRM,
AtRegNet, DAP-seq).

Architecture: Expression -> Node embedding (MLP) -> Message passing (GAT/GCN) -> Edge predictor
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def infer_grn_gnn(
    expression: np.ndarray,
    genes: list[str],
    regulators: list[str],
    gold_standard_edges: list[tuple[str, str]] | None = None,
    hidden_dim: int = 128,
    n_layers: int = 3,
    learning_rate: float = 0.001,
    epochs: int = 100,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Infer GRN using a Graph Neural Network.

    The GNN is trained on gold-standard interactions (if provided) and predicts
    edge probabilities for all TF-gene pairs.

    Parameters
    ----------
    expression : np.ndarray
        Z-scored expression matrix (n_genes x n_experiments).
    genes : list[str]
        Gene names.
    regulators : list[str]
        TF names.
    gold_standard_edges : list of (tf, tg) tuples, optional
        Known regulatory interactions for supervised training.
    hidden_dim : int
        GNN hidden layer dimension.
    n_layers : int
        Number of message-passing layers.
    learning_rate : float
        Adam learning rate.
    epochs : int
        Training epochs.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    dict with 'lead_support' key containing predicted edge matrix.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "GNN inference requires PyTorch Geometric. "
            "Install with: pip install 'merit[deep]'"
        )

    logger.info("GNN GRN inference: %d genes, %d regulators", len(genes), len(regulators))

    gene_idx = {g: i for i, g in enumerate(genes)}
    reg_idx = {r: i for i, r in enumerate(regulators)}
    n_genes = len(genes)
    n_features = expression.shape[1]

    # Node features: expression profiles
    x = torch.tensor(expression, dtype=torch.float32).to(device)

    # Build initial graph from correlation (top-k neighbors)
    corr = np.corrcoef(expression)
    np.fill_diagonal(corr, 0)
    k = min(50, n_genes - 1)
    edge_src, edge_dst = [], []
    for i in range(n_genes):
        top_k = np.argsort(np.abs(corr[i]))[-k:]
        for j in top_k:
            edge_src.append(i)
            edge_dst.append(j)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)

    # Positive labels from gold standard
    pos_edges = []
    if gold_standard_edges:
        for tf, tg in gold_standard_edges:
            if tf in gene_idx and tg in gene_idx:
                pos_edges.append((gene_idx[tf], gene_idx[tg]))

    # Model
    class GRNPredictor(nn.Module):
        def __init__(self, in_dim, hid_dim, n_layers):
            super().__init__()
            self.encoder = nn.Linear(in_dim, hid_dim)
            self.convs = nn.ModuleList([
                GATConv(hid_dim, hid_dim, heads=4, concat=False)
                for _ in range(n_layers)
            ])
            self.edge_mlp = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 1),
            )

        def forward(self, x, edge_index):
            h = F.relu(self.encoder(x))
            for conv in self.convs:
                h = F.relu(conv(h, edge_index))
            return h

        def predict_edge(self, h, src, dst):
            edge_feat = torch.cat([h[src], h[dst]], dim=-1)
            return torch.sigmoid(self.edge_mlp(edge_feat)).squeeze(-1)

    model = GRNPredictor(n_features, hidden_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if pos_edges:
        logger.info("Training GNN on %d gold-standard edges...", len(pos_edges))
        pos_src = torch.tensor([e[0] for e in pos_edges], dtype=torch.long).to(device)
        pos_dst = torch.tensor([e[1] for e in pos_edges], dtype=torch.long).to(device)

        # Negative sampling
        for epoch in range(epochs):
            model.train()
            h = model(x, edge_index)

            # Positive scores
            pos_scores = model.predict_edge(h, pos_src, pos_dst)

            # Random negative edges
            neg_src = torch.randint(0, n_genes, (len(pos_edges),)).to(device)
            neg_dst = torch.randint(0, n_genes, (len(pos_edges),)).to(device)
            neg_scores = model.predict_edge(h, neg_src, neg_dst)

            loss = F.binary_cross_entropy(
                torch.cat([pos_scores, neg_scores]),
                torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, epochs, loss.item())

    # Predict all TF-gene edges
    model.eval()
    with torch.no_grad():
        h = model(x, edge_index)
        mat = np.zeros((len(regulators), n_genes), dtype=np.float64)

        for i, tf in enumerate(regulators):
            tf_node = gene_idx[tf]
            src = torch.full((n_genes,), tf_node, dtype=torch.long).to(device)
            dst = torch.arange(n_genes, dtype=torch.long).to(device)
            scores = model.predict_edge(h, src, dst).cpu().numpy()
            mat[i] = scores

    logger.info("GNN prediction: %d edges above 0.5", (mat > 0.5).sum())
    return {"lead_support": mat, "rf": mat, "lr": mat, "clr": mat}
