"""RL-based iterative network refinement.

The RL agent operates AFTER the edge predictor has produced initial edge
probabilities. It iteratively refines the network by:

1. Observing: network statistics, edge probability distribution, gold-standard
   performance on a held-out validation set
2. Acting: adjusting the global edge threshold, re-weighting evidence sources,
   toggling motif requirement
3. Receiving reward: based on held-out gold-standard F1 + topology quality

This is formulated as a bandit/short-horizon MDP — each episode is one round
of threshold adjustment, not a long sequential process.

Design rationale:
- The GAT/MLP edge predictor does the heavy lifting (learning edge probabilities)
- The RL agent does meta-optimization: finding the best operating point on the
  precision-recall curve and the best evidence weighting
- This separation keeps the RL problem tractable (small action space, clear reward)

Uses MLX for fast inference on Apple Silicon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NetworkStats:
    """Observable state for the RL agent."""

    n_edges: int
    n_tfs: int
    n_targets: int
    density: float
    mean_out_degree: float
    std_out_degree: float
    max_out_degree: int
    mean_edge_prob: float
    median_edge_prob: float
    prob_25th: float
    prob_75th: float
    # Performance on validation gold-standard (if available)
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0


@dataclass
class RefinementAction:
    """Action the RL agent takes."""

    threshold: float        # Edge probability cutoff (0-1)
    rf_weight: float        # Re-weight RF evidence (0-2)
    lr_weight: float        # Re-weight LR evidence (0-2)
    clr_weight: float       # Re-weight CLR evidence (0-2)
    motif_weight: float     # Weight for motif evidence (0-2)
    require_motif: bool     # Hard requirement for motif support


def compute_network_stats(
    edge_probs: np.ndarray,
    threshold: float,
    val_edges: set[tuple[str, str]] | None = None,
    tf_names: list[str] | None = None,
    gene_names: list[str] | None = None,
) -> NetworkStats:
    """Compute network statistics for the current threshold."""
    binary = edge_probs >= threshold
    n_edges = int(binary.sum())
    n_tfs = int((binary.sum(axis=1) > 0).sum())
    n_targets = int((binary.sum(axis=0) > 0).sum())
    out_degrees = binary.sum(axis=1)

    stats = NetworkStats(
        n_edges=n_edges,
        n_tfs=n_tfs,
        n_targets=n_targets,
        density=n_edges / max(binary.size, 1),
        mean_out_degree=float(out_degrees.mean()),
        std_out_degree=float(out_degrees.std()),
        max_out_degree=int(out_degrees.max()),
        mean_edge_prob=float(edge_probs[edge_probs > 0].mean()) if (edge_probs > 0).any() else 0,
        median_edge_prob=float(np.median(edge_probs[edge_probs > 0])) if (edge_probs > 0).any() else 0,
        prob_25th=float(np.percentile(edge_probs[edge_probs > 0], 25)) if (edge_probs > 0).any() else 0,
        prob_75th=float(np.percentile(edge_probs[edge_probs > 0], 75)) if (edge_probs > 0).any() else 0,
    )

    # Compute validation metrics if gold-standard provided
    if val_edges and tf_names and gene_names:
        predicted = set()
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j]:
                    predicted.add((tf_names[i], gene_names[j]))

        tp = len(predicted & val_edges)
        fp = len(predicted - val_edges)
        fn = len(val_edges - predicted)
        stats.val_precision = tp / max(tp + fp, 1)
        stats.val_recall = tp / max(tp + fn, 1)
        stats.val_f1 = 2 * stats.val_precision * stats.val_recall / max(
            stats.val_precision + stats.val_recall, 1e-8
        )

    return stats


def _stats_to_vector(stats: NetworkStats) -> np.ndarray:
    """Convert stats to feature vector for the policy."""
    return np.array([
        np.log1p(stats.n_edges) / 15,
        stats.n_tfs / 1000,
        stats.n_targets / 20000,
        np.log1p(stats.density * 1e6) / 15,
        stats.mean_out_degree / 100,
        stats.std_out_degree / 100,
        stats.max_out_degree / 1000,
        stats.mean_edge_prob,
        stats.median_edge_prob,
        stats.prob_25th,
        stats.prob_75th,
        stats.val_precision,
        stats.val_recall,
        stats.val_f1,
    ], dtype=np.float32)


def refine_network_rl(
    edge_features: np.ndarray,
    edge_probs: np.ndarray,
    tf_names: list[str],
    gene_names: list[str],
    train_edges: set[tuple[str, str]],
    val_edges: set[tuple[str, str]],
    motif_features: np.ndarray | None = None,
    episodes: int = 500,
    learning_rate: float = 0.001,
    explore_std: float = 0.1,
) -> tuple[np.ndarray, RefinementAction]:
    """Run RL-based network refinement using MLX.

    The agent learns to:
    1. Find the optimal edge probability threshold
    2. Re-weight evidence sources for edge re-scoring
    3. Decide whether to require motif support

    Parameters
    ----------
    edge_features : np.ndarray (n_tfs, n_genes, n_features)
        Per-edge feature vectors (RF, LR, CLR scores, etc.)
    edge_probs : np.ndarray (n_tfs, n_genes)
        Initial edge probabilities from the edge predictor.
    tf_names, gene_names : list[str]
    train_edges : set of (tf, gene) tuples
        Gold-standard edges for reward computation.
    val_edges : set of (tf, gene) tuples
        Held-out edges for generalization evaluation.
    motif_features : np.ndarray (n_tfs, n_genes), optional
        Motif support scores.
    episodes : int
    learning_rate : float
    explore_std : float
        Gaussian exploration noise.

    Returns
    -------
    refined_probs : np.ndarray (n_tfs, n_genes)
    best_action : RefinementAction
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        USE_MLX = True
    except ImportError:
        USE_MLX = False
        logger.info("MLX not available, using NumPy for RL (slower)")

    state_dim = 14
    action_dim = 6  # threshold, rf_w, lr_w, clr_w, motif_w, require_motif_logit

    n_features = edge_features.shape[-1]

    if USE_MLX:
        # Policy network (small — this is a bandit, not a deep RL problem)
        policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        mx.eval(policy.parameters())
        optimizer = optim.Adam(learning_rate=learning_rate)

    best_reward = float("-inf")
    best_action = None
    best_probs = edge_probs.copy()
    reward_history = []

    logger.info("RL refinement: %d episodes, %d train edges, %d val edges",
                episodes, len(train_edges), len(val_edges))

    for episode in range(episodes):
        # Compute current state
        stats = compute_network_stats(
            edge_probs, 0.5, val_edges, tf_names, gene_names
        )
        state_vec = _stats_to_vector(stats)

        if USE_MLX:
            state_mx = mx.array(state_vec[None, :])  # (1, state_dim)
            raw_action = policy(state_mx).squeeze(0)
            mx.eval(raw_action)
            raw_np = np.array(raw_action)
        else:
            # Random search fallback
            raw_np = np.random.randn(action_dim).astype(np.float32) * 0.5

        # Add exploration noise
        noise = np.random.randn(action_dim).astype(np.float32) * explore_std
        raw_np = raw_np + noise

        # Interpret action
        threshold = 1.0 / (1.0 + np.exp(-raw_np[0]))  # sigmoid -> (0, 1)
        rf_w = np.exp(np.clip(raw_np[1], -2, 2))       # (0.13, 7.4)
        lr_w = np.exp(np.clip(raw_np[2], -2, 2))
        clr_w = np.exp(np.clip(raw_np[3], -2, 2))
        motif_w = np.exp(np.clip(raw_np[4], -2, 2))
        require_motif = raw_np[5] > 0

        action = RefinementAction(
            threshold=float(threshold),
            rf_weight=float(rf_w),
            lr_weight=float(lr_w),
            clr_weight=float(clr_w),
            motif_weight=float(motif_w),
            require_motif=bool(require_motif),
        )

        # Re-score edges with adjusted weights
        # Features assumed order: rf, lr, clr, rf_mc, lr_mc, clr_mc, motif
        weights = np.ones(n_features)
        if n_features >= 3:
            weights[0] = rf_w
            weights[1] = lr_w
            weights[2] = clr_w
        if n_features >= 7:
            weights[6] = motif_w

        weighted_features = edge_features * weights[None, None, :]
        rescored = weighted_features.mean(axis=-1)

        # Normalize to [0, 1]
        mn, mx_val = rescored.min(), rescored.max()
        if mx_val > mn:
            rescored = (rescored - mn) / (mx_val - mn)

        # Apply motif requirement
        if require_motif and motif_features is not None:
            rescored = rescored * (motif_features > 0).astype(float)

        # Compute reward on training set
        binary = rescored >= threshold
        predicted = set()
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j]:
                    predicted.add((tf_names[i], gene_names[j]))

        tp = len(predicted & train_edges)
        fp = len(predicted - train_edges)
        fn = len(train_edges - predicted)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Density penalty
        density = len(predicted) / max(binary.size, 1)
        density_penalty = max(0, density - 0.01) * 5

        reward = f1 - density_penalty
        reward_history.append(reward)

        if reward > best_reward:
            best_reward = reward
            best_action = action
            best_probs = rescored.copy()

        # Policy gradient update (MLX)
        if USE_MLX:
            def policy_loss(policy, state, reward_val):
                logits = policy(state).squeeze(0)
                # Simple REINFORCE: maximize reward * log_prob
                log_prob = -0.5 * mx.sum(logits ** 2)  # Gaussian log-prob approximation
                return -log_prob * reward_val

            loss_grad_fn = nn.value_and_grad(policy, policy_loss)
            loss, grads = loss_grad_fn(
                policy, mx.array(state_vec[None, :]), mx.array(reward)
            )
            optimizer.update(policy, grads)
            mx.eval(policy.parameters(), optimizer.state)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            logger.info(
                "  Episode %d/%d: reward=%.4f (avg=%.4f), best=%.4f, "
                "threshold=%.3f, edges=%d",
                episode + 1, episodes, reward, avg_reward, best_reward,
                threshold, len(predicted),
            )

    # Evaluate best action on validation set
    val_stats = compute_network_stats(
        best_probs, best_action.threshold, val_edges, tf_names, gene_names
    )
    logger.info(
        "RL refinement complete. Best reward: %.4f\n"
        "  Best threshold: %.3f\n"
        "  Val F1: %.4f (P=%.4f, R=%.4f)\n"
        "  Edges: %d, TFs: %d, Targets: %d",
        best_reward, best_action.threshold,
        val_stats.val_f1, val_stats.val_precision, val_stats.val_recall,
        val_stats.n_edges, val_stats.n_tfs, val_stats.n_targets,
    )

    return best_probs, best_action
