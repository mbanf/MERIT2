"""End-to-end v2.0 integration pipeline.

Orchestrates the full MERIT v2.0 workflow:
    1. Load v1.0 intermediate score matrices (.rds)
    2. Load gold-standard edges — DAP-seq for training, ATRM+AtRegNet for eval
    3. Build per-edge feature vectors
    4. Train MLP or GAT edge predictor (MLX or PyTorch)
    5. RL-based network refinement (train on DAP-seq subset)
    6. Evaluate on ATRM + AtRegNet (same benchmarks as v1.0 for comparability)
    7. Export refined network

Evaluation strategy:
    - DAP-seq (~1.8M in vitro binding edges) → training signal
    - ATRM (~1,165 literature-curated edges) → evaluation (highest quality)
    - AtRegNet (~228 confirmed edges) → evaluation (high quality)
    - AUROC reported per eval source for direct v1.0 comparison

Can run on actual Release 1.0 data (requires pyreadr) or in
synthetic mode for pipeline testing.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from merit.utils.gold_standard import (
    GoldStandard,
    load_all_gold_standards,
    split_by_source,
    train_test_split_edges,
)
from merit.utils.rds_loader import (
    V1Intermediates,
    build_edge_features,
    load_v1_intermediates,
)
from merit.integration.edge_predictor import (
    EdgePrediction,
    predict_edges_mlx,
    train_edge_predictor_mlx,
    train_edge_predictor_torch,
)
from merit.integration.gat_integrator import GATConfig, train_and_predict_gat
from merit.integration.rl_refiner import refine_network_rl

logger = logging.getLogger(__name__)


def _detect_backend() -> str:
    """Detect best available compute backend."""
    try:
        import mlx.core as mx
        _ = mx.array([1.0])
        return "mlx"
    except ImportError:
        pass
    try:
        import torch
        return "torch"
    except ImportError:
        pass
    return "numpy"


def _synthetic_intermediates(
    n_tfs: int = 50,
    n_genes: int = 200,
    n_bg: int = 5,
    seed: int = 42,
) -> tuple[V1Intermediates, list[str], list[str]]:
    """Generate synthetic intermediates for pipeline testing."""
    rng = np.random.RandomState(seed)
    tf_names = [f"AT1G{i:05d}" for i in range(n_tfs)]
    gene_names = [f"AT2G{i:05d}" for i in range(n_genes)]

    rf = rng.exponential(0.5, (n_tfs, n_genes))
    lr = rng.exponential(0.3, (n_tfs, n_genes))
    clr = rng.exponential(0.4, (n_tfs, n_genes))
    motifs = (rng.rand(n_tfs, n_genes) > 0.7).astype(float)

    rf_bg = [rng.exponential(0.5, (n_tfs, n_genes)) for _ in range(n_bg)]
    lr_bg = [rng.exponential(0.3, (n_tfs, n_genes)) for _ in range(n_bg)]
    clr_bg = [rng.exponential(0.4, (n_tfs, n_genes)) for _ in range(n_bg)]

    intermediates = V1Intermediates(
        rf_scores=rf, lr_scores=lr, clr_scores=clr,
        motif_matrix=motifs, rf_bg=rf_bg, lr_bg=lr_bg, clr_bg=clr_bg,
        tf_names=tf_names, gene_names=gene_names,
    )
    return intermediates, tf_names, gene_names


def _synthetic_gold_standard(
    tf_names: list[str],
    gene_names: list[str],
    n_edges: int = 300,
    seed: int = 42,
) -> GoldStandard:
    """Generate synthetic gold-standard for pipeline testing."""
    rng = np.random.RandomState(seed)
    edges = set()
    while len(edges) < n_edges:
        tf = rng.choice(tf_names)
        gene = rng.choice(gene_names)
        if tf != gene:
            edges.add((tf, gene))

    records = [
        {"tf": tf, "target": tgt, "sources": "synthetic",
         "n_sources": 1, "src_synthetic": True}
        for tf, tgt in sorted(edges)
    ]
    df = pd.DataFrame(records)
    return GoldStandard(
        edges=df,
        tf_names=sorted(set(r["tf"] for r in records)),
        target_names=sorted(set(r["target"] for r in records)),
    )


def compute_auroc(
    probs: np.ndarray,
    eval_edges: set[tuple[str, str]],
    tf_names: list[str],
    gene_names: list[str],
) -> float:
    """Compute AUROC for edge predictions vs gold-standard.

    Evaluates only TF-gene pairs where the TF appears in the eval set,
    matching v1.0's evaluation approach (restricted to TFs with known targets).
    """
    # Identify TFs that appear in eval edges
    eval_tfs = {tf for tf, _ in eval_edges}
    tf_idx = {t: i for i, t in enumerate(tf_names)}
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    # Collect predictions and labels for relevant TF-gene pairs
    y_true = []
    y_score = []
    for tf in eval_tfs:
        ti = tf_idx.get(tf)
        if ti is None:
            continue
        for j, gene in enumerate(gene_names):
            y_score.append(probs[ti, j])
            y_true.append(1.0 if (tf, gene) in eval_edges else 0.0)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        logger.warning("AUROC undefined: all labels identical")
        return 0.0

    # Compute AUROC (manual — no sklearn dependency)
    # Sort by descending score
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    # Trapezoidal integration
    tp = 0.0
    fp = 0.0
    auroc = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0

    for label in y_sorted:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Trapezoidal rule
        auroc += 0.5 * (tpr + prev_tpr) * (fpr - prev_fpr)
        prev_tpr = tpr
        prev_fpr = fpr

    return float(auroc)


def _evaluate(
    label: str,
    probs: np.ndarray,
    threshold: float,
    eval_edge_sets: dict[str, set[tuple[str, str]]],
    tf_names: list[str],
    gene_names: list[str],
) -> dict:
    """Evaluate edge predictions against multiple gold-standard sources.

    Reports precision/recall/F1 at threshold + AUROC per source.
    """
    binary = probs >= threshold
    # Build predicted set efficiently using numpy
    pred_i, pred_j = np.where(binary)
    predicted = {(tf_names[i], gene_names[j]) for i, j in zip(pred_i, pred_j)}

    results = {}
    for source, eval_edges in eval_edge_sets.items():
        tp = len(predicted & eval_edges)
        fp_count = len(predicted) - tp  # approx: FP relative to this source
        fn = len(eval_edges - predicted)
        precision = tp / max(tp + fp_count, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        auroc = compute_auroc(probs, eval_edges, tf_names, gene_names)

        logger.info(
            "%s [%s] (threshold=%.3f): AUROC=%.4f P=%.4f R=%.4f F1=%.4f "
            "(TP=%d FN=%d, %d predicted, %d gold)",
            label, source, threshold, auroc, precision, recall, f1,
            tp, fn, len(predicted), len(eval_edges),
        )
        results[source] = {
            "auroc": auroc, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fn": fn,
        }

    return results


def run_pipeline(
    rds_dir: str | Path | None = None,
    gold_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    backend: str | None = None,
    model: str = "mlp",
    synthetic: bool = False,
    epochs_predictor: int = 100,
    epochs_rl: int = 300,
    seed: int = 42,
) -> EdgePrediction:
    """Run the full v2.0 integration pipeline.

    Parameters
    ----------
    rds_dir : path, optional
        Directory with v1.0 intermediate .rds files.
    gold_dir : path, optional
        Directory with gold-standard files (ATRM, AtRegNet, DAP-seq).
    output_dir : path, optional
        Where to write results. If None, results are returned but not saved.
    backend : str, optional
        "mlx", "torch", or "numpy". Auto-detected if None.
    model : str
        "mlp" (independent edge prediction) or "gat" (graph attention network).
    synthetic : bool
        Use synthetic data for testing (no file dependencies).
    epochs_predictor : int
        Training epochs for the edge predictor.
    epochs_rl : int
        Episodes for RL refinement.
    seed : int

    Returns
    -------
    EdgePrediction with refined edge probabilities.
    """
    t0 = time.time()

    if backend is None:
        backend = _detect_backend()
    logger.info("Pipeline backend: %s", backend)

    # ── Step 1: Load data ──────────────────────────────────────────────
    if synthetic:
        logger.info("Using synthetic data for pipeline testing")
        intermediates, tf_names, gene_names = _synthetic_intermediates(seed=seed)
        gold = _synthetic_gold_standard(tf_names, gene_names, seed=seed)
        # Synthetic: random split since there's only one source
        train_list, test_list = train_test_split_edges(gold, test_fraction=0.2, seed=seed)
        train_edges = set(train_list)
        eval_edge_sets = {"synthetic": set(test_list)}
    else:
        if rds_dir is None or gold_dir is None:
            raise ValueError("rds_dir and gold_dir required when not in synthetic mode")
        intermediates = load_v1_intermediates(rds_dir)
        gold = load_all_gold_standards(gold_dir)

        tf_names = intermediates.tf_names
        gene_names = intermediates.gene_names
        if tf_names is None or gene_names is None:
            raise ValueError(
                "V1 intermediates must include tf_names and gene_names. "
                "Set them after loading RDS matrices."
            )

        # Source-based split: train on DAP-seq, evaluate on ATRM + AtRegNet
        train_edges, eval_edge_sets = split_by_source(
            gold,
            train_sources=["dapseq"],
            eval_sources=["atrm", "atregnet"],
        )

    logger.info("Data: %d TFs x %d genes", len(tf_names), len(gene_names))

    # ── Step 2: Build edge features ────────────────────────────────────
    edge_features = build_edge_features(intermediates)
    n_tfs, n_genes, n_features = edge_features.shape
    logger.info("Edge features: %s (%d features per edge)", edge_features.shape, n_features)

    # ── Step 3: Build training matrix from DAP-seq edges ───────────────
    tf_idx_map = {t: i for i, t in enumerate(tf_names)}
    gene_idx_map = {g: i for i, g in enumerate(gene_names)}

    train_matrix = np.zeros((n_tfs, n_genes), dtype=np.float64)
    n_mapped = 0
    for tf, tgt in train_edges:
        ti = tf_idx_map.get(tf)
        gi = gene_idx_map.get(tgt)
        if ti is not None and gi is not None:
            train_matrix[ti, gi] = 1.0
            n_mapped += 1

    # Count eval edges that map to our TF x gene space
    all_eval_edges = set()
    for edges in eval_edge_sets.values():
        all_eval_edges |= edges
    n_eval_mapped = sum(
        1 for tf, tgt in all_eval_edges
        if tf in tf_idx_map and tgt in gene_idx_map
    )

    logger.info(
        "Training: %d DAP-seq edges mapped to matrix (%d total train)",
        n_mapped, len(train_edges),
    )
    logger.info(
        "Evaluation: %d ATRM+AtRegNet edges mapped (%d total eval)",
        n_eval_mapped, len(all_eval_edges),
    )

    # Split DAP-seq training into train/val for RL
    train_list = list(train_edges)
    rng = np.random.RandomState(seed)
    rng.shuffle(train_list)
    split = int(len(train_list) * 0.9)
    rl_train = set(train_list[:split])
    rl_val = set(train_list[split:])

    # ── Step 4: Train edge predictor ───────────────────────────────────
    logger.info("Training edge predictor (model=%s, backend=%s, %d epochs)...",
                model, backend, epochs_predictor)

    if model == "gat":
        gat_config = GATConfig(epochs=epochs_predictor)
        edge_probs = train_and_predict_gat(
            edge_features, train_matrix,
            tf_names, gene_names,
            config=gat_config,
        )
    elif backend == "mlx":
        predictor, losses = train_edge_predictor_mlx(
            edge_features, train_matrix,
            epochs=epochs_predictor,
        )
        edge_probs = predict_edges_mlx(predictor, edge_features)
    elif backend == "torch":
        predictor, losses = train_edge_predictor_torch(
            edge_features, train_matrix,
            epochs=epochs_predictor,
        )
        import torch
        predictor.eval()
        with torch.no_grad():
            X = torch.tensor(edge_features.reshape(-1, n_features).astype(np.float32))
            logits = predictor(X).squeeze(-1)
            edge_probs = torch.sigmoid(logits).numpy().reshape(n_tfs, n_genes)
    else:
        logger.warning("No ML backend — using weighted feature average as baseline")
        edge_probs = edge_features.mean(axis=-1)
        mn, mx_val = edge_probs.min(), edge_probs.max()
        if mx_val > mn:
            edge_probs = (edge_probs - mn) / (mx_val - mn)

    logger.info("Edge predictor done. Prob range: [%.4f, %.4f], mean: %.4f",
                edge_probs.min(), edge_probs.max(), edge_probs.mean())

    # Evaluate predictor before RL (on ATRM + AtRegNet)
    _evaluate("Pre-RL", edge_probs, 0.5, eval_edge_sets, tf_names, gene_names)

    # ── Step 5: RL refinement ──────────────────────────────────────────
    logger.info("Running RL refinement (%d episodes)...", epochs_rl)

    refined_probs, best_action = refine_network_rl(
        edge_features=edge_features,
        edge_probs=edge_probs,
        tf_names=tf_names,
        gene_names=gene_names,
        train_edges=rl_train,
        val_edges=rl_val,
        motif_features=intermediates.motif_matrix,
        episodes=epochs_rl,
    )

    logger.info("Best action: threshold=%.3f, rf_w=%.2f, lr_w=%.2f, clr_w=%.2f, "
                "motif_w=%.2f, require_motif=%s",
                best_action.threshold, best_action.rf_weight, best_action.lr_weight,
                best_action.clr_weight, best_action.motif_weight, best_action.require_motif)

    # Evaluate after RL (on ATRM + AtRegNet — never seen during training)
    eval_results = _evaluate("Post-RL", refined_probs, best_action.threshold,
                             eval_edge_sets, tf_names, gene_names)

    # ── Step 6: Build result ───────────────────────────────────────────
    result = EdgePrediction(
        probabilities=refined_probs,
        tf_names=tf_names,
        gene_names=gene_names,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_results(result, best_action, eval_results, output_dir)

    elapsed = time.time() - t0
    n_edges = int((refined_probs >= best_action.threshold).sum())
    logger.info("Pipeline complete in %.1fs. %d edges at threshold %.3f",
                elapsed, n_edges, best_action.threshold)

    return result


def _save_results(
    result: EdgePrediction,
    best_action,
    eval_results: dict,
    output_dir: Path,
) -> None:
    """Save pipeline results to disk."""
    np.save(output_dir / "edge_probabilities.npy", result.probabilities)

    edges = result.to_edge_list(threshold=best_action.threshold)
    df = pd.DataFrame(edges, columns=["tf", "target", "probability"])
    df.to_csv(output_dir / "refined_network.csv", index=False)
    logger.info("Saved %d edges to %s", len(edges), output_dir / "refined_network.csv")

    # Save parameters
    params = {
        "threshold": best_action.threshold,
        "rf_weight": best_action.rf_weight,
        "lr_weight": best_action.lr_weight,
        "clr_weight": best_action.clr_weight,
        "motif_weight": best_action.motif_weight,
        "require_motif": best_action.require_motif,
    }
    pd.Series(params).to_csv(output_dir / "best_params.csv")

    # Save evaluation metrics
    eval_rows = []
    for source, metrics in eval_results.items():
        eval_rows.append({"source": source, **metrics})
    pd.DataFrame(eval_rows).to_csv(output_dir / "evaluation.csv", index=False)
    logger.info("Saved evaluation to %s", output_dir / "evaluation.csv")


def main():
    parser = argparse.ArgumentParser(description="MERIT v2.0 integration pipeline")
    parser.add_argument("--rds-dir", type=str, help="Path to v1.0 tmp/ with .rds files")
    parser.add_argument("--gold-dir", type=str, help="Path to gold-standard data directory")
    parser.add_argument("--output-dir", type=str, default="results/v2_pipeline")
    parser.add_argument("--backend", choices=["mlx", "torch", "numpy"], default=None)
    parser.add_argument("--model", choices=["mlp", "gat"], default="mlp",
                        help="Edge predictor: mlp (independent) or gat (graph attention)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--epochs-predictor", type=int, default=100)
    parser.add_argument("--epochs-rl", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        rds_dir=args.rds_dir,
        gold_dir=args.gold_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        model=args.model,
        synthetic=args.synthetic,
        epochs_predictor=args.epochs_predictor,
        epochs_rl=args.epochs_rl,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
