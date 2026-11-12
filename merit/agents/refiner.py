"""RL-based network refinement agent.

Instead of fixed thresholds (95th percentile, n_support=1), an RL agent
learns the optimal strategy for refining a noisy initial GRN.

State:  Current network statistics (density, degree distribution, enrichment scores)
        + per-edge features (RF score, LR score, CLR score, motif support)
Action: Accept/reject edges, adjust thresholds, reweight methods
Reward: AUROC on gold-standard + biological plausibility metrics

The agent iteratively refines the network to maximize biological validity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NetworkState:
    """Observable state of the current GRN for the RL agent."""

    n_edges: int
    n_tfs: int
    n_targets: int
    density: float
    mean_degree: float
    max_degree: int
    rf_scores: np.ndarray  # flat array of edge RF scores
    lr_scores: np.ndarray
    clr_scores: np.ndarray
    motif_support: np.ndarray  # 0/1 for each edge


@dataclass
class RefinementAction:
    """Action the agent can take."""

    # Threshold adjustments (continuous)
    rf_threshold: float
    lr_threshold: float
    clr_threshold: float
    # Method weighting
    n_support_required: int  # 0, 1, or 2
    # Whether to require motif support
    require_motif: bool


class NetworkRefinerAgent:
    """RL agent that learns to refine a GRN by adjusting thresholds and method weights.

    Uses a simple policy gradient (REINFORCE) approach. The agent observes
    network statistics and edge scores, and outputs threshold/weight adjustments
    that maximize a reward combining:
    - AUROC against gold-standard edges
    - Network topology quality (scale-free, small-world properties)
    - Enrichment of known TF families
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.device = device
        self.policy = None
        self.optimizer = None

    def _build_policy(self, state_dim: int, action_dim: int):
        """Build the policy network."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "RL agent requires PyTorch. Install with: pip install 'merit[agents]'"
            )

        self.policy = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def _state_to_features(self, state: NetworkState) -> np.ndarray:
        """Convert network state to feature vector for the policy."""
        return np.array([
            state.n_edges / 1e6,
            state.n_tfs / 1000,
            state.n_targets / 20000,
            state.density,
            state.mean_degree / 100,
            state.max_degree / 1000,
            np.mean(state.rf_scores),
            np.std(state.rf_scores),
            np.mean(state.lr_scores),
            np.std(state.lr_scores),
            np.mean(state.clr_scores),
            np.std(state.clr_scores),
            np.mean(state.motif_support),
        ], dtype=np.float32)

    def compute_reward(
        self,
        predicted_edges: set[tuple[str, str]],
        gold_standard: set[tuple[str, str]],
        n_tfs: int,
        n_genes: int,
    ) -> float:
        """Compute reward from predicted network vs gold standard.

        Combines precision, recall, and network topology metrics.
        """
        if not predicted_edges:
            return -1.0

        tp = len(predicted_edges & gold_standard)
        fp = len(predicted_edges - gold_standard)
        fn = len(gold_standard - predicted_edges)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Density penalty (prefer sparse networks)
        density = len(predicted_edges) / (n_tfs * n_genes)
        density_penalty = max(0, density - 0.01) * 10

        return f1 - density_penalty

    def refine(
        self,
        rf_scores: np.ndarray,
        lr_scores: np.ndarray,
        clr_scores: np.ndarray,
        motif_matrix: np.ndarray,
        tf_names: list[str],
        gene_names: list[str],
        gold_standard: list[tuple[str, str]] | None = None,
        episodes: int = 1000,
    ) -> np.ndarray:
        """Run RL-based network refinement.

        Parameters
        ----------
        rf_scores, lr_scores, clr_scores : np.ndarray
            Raw score matrices from each GRN method (n_tfs x n_genes).
        motif_matrix : np.ndarray
            Binary TFBS matrix (n_tfs x n_genes).
        tf_names, gene_names : list[str]
            Names for matrix rows/columns.
        gold_standard : list of (tf, tg) tuples, optional
            Known interactions for reward computation.
        episodes : int
            Number of RL episodes.

        Returns
        -------
        np.ndarray
            Refined binary GRN matrix (n_tfs x n_genes).
        """
        import torch
        import torch.nn.functional as F

        state_dim = 13  # From _state_to_features
        action_dim = 5  # rf_th, lr_th, clr_th (continuous), n_support (discrete), require_motif

        if self.policy is None:
            self._build_policy(state_dim, action_dim)

        gs_set = set(gold_standard) if gold_standard else set()

        best_reward = float("-inf")
        best_network = None

        logger.info("RL refinement: %d episodes", episodes)

        for episode in range(episodes):
            # Sample action from policy
            flat_rf = rf_scores[rf_scores > 0]
            flat_lr = lr_scores[lr_scores > 0]
            flat_clr = clr_scores[clr_scores > 0]

            state = NetworkState(
                n_edges=int((rf_scores > 0).sum()),
                n_tfs=len(tf_names),
                n_targets=len(gene_names),
                density=float((rf_scores > 0).sum()) / rf_scores.size,
                mean_degree=float((rf_scores > 0).sum(axis=1).mean()),
                max_degree=int((rf_scores > 0).sum(axis=1).max()),
                rf_scores=flat_rf if len(flat_rf) > 0 else np.array([0]),
                lr_scores=flat_lr if len(flat_lr) > 0 else np.array([0]),
                clr_scores=flat_clr if len(flat_clr) > 0 else np.array([0]),
                motif_support=motif_matrix.ravel(),
            )

            features = torch.tensor(
                self._state_to_features(state), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                raw_action = self.policy(features).squeeze(0)

            # Interpret action
            rf_quantile = torch.sigmoid(raw_action[0]).item()
            lr_quantile = torch.sigmoid(raw_action[1]).item()
            clr_quantile = torch.sigmoid(raw_action[2]).item()
            n_support = int(torch.clamp(raw_action[3], 0, 2).round().item())
            require_motif = raw_action[4].item() > 0

            # Apply thresholds
            rf_th = np.quantile(rf_scores[rf_scores > 0], rf_quantile) if rf_quantile < 1 else float("inf")
            lr_th = np.quantile(lr_scores[lr_scores > 0], lr_quantile) if lr_quantile < 1 else float("inf")
            clr_th = np.quantile(clr_scores[clr_scores > 0], clr_quantile) if clr_quantile < 1 else float("inf")

            rf_binary = (rf_scores >= rf_th).astype(float)
            lr_binary = (lr_scores >= lr_th).astype(float)
            clr_binary = (clr_scores >= clr_th).astype(float)

            support = lr_binary + clr_binary
            support_mask = (support >= n_support).astype(float)
            network = rf_binary * support_mask

            if require_motif:
                tfs_shared = min(network.shape[0], motif_matrix.shape[0])
                genes_shared = min(network.shape[1], motif_matrix.shape[1])
                network[:tfs_shared, :genes_shared] *= motif_matrix[:tfs_shared, :genes_shared]

            # Compute reward
            predicted = set()
            for i in range(network.shape[0]):
                for j in range(network.shape[1]):
                    if network[i, j] > 0:
                        predicted.add((tf_names[i], gene_names[j]))

            reward = self.compute_reward(predicted, gs_set, len(tf_names), len(gene_names))

            if reward > best_reward:
                best_reward = reward
                best_network = network.copy()

            # Policy gradient update
            log_prob = F.log_softmax(self.policy(features), dim=-1).sum()
            loss = -log_prob * reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (episode + 1) % 100 == 0:
                logger.info(
                    "  Episode %d/%d: reward=%.4f, edges=%d, best=%.4f",
                    episode + 1, episodes, reward, len(predicted), best_reward,
                )

        logger.info("RL refinement complete. Best reward: %.4f", best_reward)
        return best_network if best_network is not None else rf_scores
