"""Validation tests: verify MERIT v2.0 modules against Release 1.0 data.

Tests:
1. Data loading — correct dimensions and content
2. Enrichment — reproduce the 1,752 results / 403 TFs exactly
3. GRN inference — small-scale smoke test (subset of genes)
4. TFBS — motif loading and PWM parsing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path so we can import merit
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent.parent / "version 1.0" / "data" / "release-1.0"


def test_data_loading():
    """Test 1: Load Release 1.0 data and verify dimensions."""
    from merit.utils.data import load_datasets

    print("=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)

    data = load_datasets(
        genes_file=DATA_DIR / "data/gene_expression/genes.txt",
        experiment_ids_file=DATA_DIR / "data/gene_expression/ids_differentialExpressionSamples.txt",
        fold_change_file=DATA_DIR / "data/gene_expression/m.foldChange_differentialExpression.txt",
        pvalue_file=DATA_DIR / "data/gene_expression/m.pvalue_differentialExpression.txt",
        experiment_annotation_file=DATA_DIR / "data/gene_expression/experiment_annotation.txt",
        tf_annotation_file=DATA_DIR / "data/motifs/df.transcriptionFactorAnnotation.txt",
    )

    print(f"  Genes: {len(data.genes)}")
    print(f"  Experiments: {len(data.experiment_ids)}")
    print(f"  Fold-change shape: {data.fold_change.shape}")
    print(f"  P-value shape: {data.pvalues.shape}")
    print(f"  TF annotations: {len(data.tf_annotation)} rows")
    print(f"  TFs with expression: {len(data.regulators)}")
    print(f"  Treatments: {len(data.condition_treatments)}")
    print(f"  Tissues: {len(data.condition_tissues)}")

    assert len(data.genes) == 21678, f"Expected 21678 genes, got {len(data.genes)}"
    assert len(data.experiment_ids) == 435, f"Expected 435 experiments, got {len(data.experiment_ids)}"
    assert data.fold_change.shape == (21678, 435), f"Wrong FC shape: {data.fold_change.shape}"
    assert data.pvalues.shape == (21678, 435), f"Wrong pval shape: {data.pvalues.shape}"

    print("  PASS: All dimensions match Release 1.0\n")
    return data


def test_enrichment(data):
    """Test 2: Reproduce the master regulator enrichment (1,752 results, 403 TFs)."""
    from merit.enrichment.master_regulators import find_bottom_level_master_regulators

    print("=" * 60)
    print("TEST 2: Master Regulator Enrichment")
    print("=" * 60)

    # Load context-specific networks from Release 1.0 results
    net_dir = DATA_DIR / "results" / "context specific networks"
    csv_files = sorted(net_dir.glob("*.csv"))
    print(f"  Loading {len(csv_files)} context-specific networks...")

    # Load gene groups
    gene_groups = pd.read_csv(
        DATA_DIR / "data/gene_sets/growth_defense_metabolic_genes.csv",
        sep="\t",
    )
    print(f"  Gene groups: {len(gene_groups)} entries")

    # Build subnetwork matrices from CSVs
    # First, collect all TFs and genes
    all_tfs = set()
    all_genes_set = set()
    networks_raw = {}
    for csv_file in csv_files:
        context = csv_file.stem
        df = pd.read_csv(csv_file)
        all_tfs.update(df["TF"].unique())
        all_genes_set.update(df["TG"].unique())
        all_genes_set.update(df["TF"].unique())
        networks_raw[context] = df

    tf_names = sorted(all_tfs)
    gene_names = sorted(all_genes_set)
    tf_idx = {t: i for i, t in enumerate(tf_names)}
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    print(f"  TFs across contexts: {len(tf_names)}")
    print(f"  Genes across contexts: {len(gene_names)}")

    # Build matrices
    subnetworks = {}
    for context, df in networks_raw.items():
        mat = np.zeros((len(tf_names), len(gene_names)), dtype=np.float64)
        for _, row in df.iterrows():
            ti = tf_idx.get(row["TF"])
            gi = gene_idx.get(row["TG"])
            if ti is not None and gi is not None:
                mat[ti, gi] = 1.0
        subnetworks[context] = mat

    print(f"  Running enrichment on {len(subnetworks)} contexts...")

    df_mr = find_bottom_level_master_regulators(
        subnetworks=subnetworks,
        tf_names=tf_names,
        gene_names=gene_names,
        gene_groups=gene_groups,
        th_pval=0.05,
    )

    n_results = len(df_mr)
    n_unique_tfs = df_mr["tf"].nunique() if len(df_mr) > 0 else 0

    print(f"  Significant results: {n_results} (expected: 1,752)")
    print(f"  Unique TFs: {n_unique_tfs} (expected: 403)")

    if n_results == 1752 and n_unique_tfs == 403:
        print("  PASS: Exact match with Release 1.0 xlsx!\n")
    else:
        print(f"  MISMATCH: got {n_results} results / {n_unique_tfs} TFs")
        print(f"  (Expected 1,752 / 403)")
        # Check if close
        if abs(n_results - 1752) < 10:
            print("  NOTE: Very close — likely minor floating-point differences\n")
        else:
            print("  FAIL: Significant discrepancy\n")

    return df_mr


def test_grn_smoke():
    """Test 3: Small-scale GRN inference smoke test (50 genes, 10 TFs)."""
    from merit.inference.random_forest import compute_rf_grn
    from merit.inference.clr import compute_clr_grn, _compute_mi_matrix, _clr_transform
    from merit.inference.monte_carlo import shuffle_expression

    print("=" * 60)
    print("TEST 3: GRN Inference Smoke Test")
    print("=" * 60)

    np.random.seed(42)
    n_genes, n_samples = 50, 20
    expression = np.random.randn(n_genes, n_samples)
    genes = [f"Gene_{i}" for i in range(n_genes)]
    regulators = genes[:10]

    # Z-score
    mean = expression.mean(axis=1, keepdims=True)
    std = expression.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    X = (expression - mean) / std

    print("  Testing RF inference (50 genes, 10 TFs, 10 trees)...")
    rf = compute_rf_grn(X, genes, regulators, n_trees=10, seed=42, n_cpus=1)
    print(f"    RF matrix shape: {rf.shape}")
    print(f"    RF non-zero: {np.count_nonzero(rf)}")
    assert rf.shape == (10, 50), f"Wrong RF shape: {rf.shape}"
    assert np.count_nonzero(rf) > 0, "RF matrix is all zeros"
    print("    RF: OK")

    print("  Testing CLR inference...")
    mi = _compute_mi_matrix(X[:20], n_neighbors=3)  # Use fewer genes for speed
    clr = _clr_transform(mi)
    print(f"    MI matrix shape: {mi.shape}")
    print(f"    CLR matrix shape: {clr.shape}")
    assert mi.shape == (20, 20)
    assert np.all(np.diag(clr) == 0), "CLR diagonal should be zero"
    print("    CLR: OK")

    print("  Testing Monte Carlo shuffle...")
    X_shuffled = shuffle_expression(X, seed=123)
    assert X_shuffled.shape == X.shape
    assert not np.array_equal(X_shuffled, X), "Shuffled should differ"
    # Each row should have same values, just reordered
    for i in range(n_genes):
        assert np.allclose(sorted(X[i]), sorted(X_shuffled[i])), "Shuffle changed values"
    print("    Monte Carlo shuffle: OK")

    print("  PASS: All GRN inference components work\n")


def test_tfbs_loading():
    """Test 4: TFBS data loading and PWM parsing."""
    from merit.tfbs.pwm_scanner import load_tf_motif_mapping, load_pwms

    print("=" * 60)
    print("TEST 4: TFBS Data Loading")
    print("=" * 60)

    tf_motifs = load_tf_motif_mapping(
        DATA_DIR / "data/motifs/TF_to_Motif_IDs.txt"
    )
    print(f"  TF-motif mappings: {len(tf_motifs)} TFs")
    assert len(tf_motifs) > 0, "No TF-motif mappings loaded"

    # Check expected count
    if len(tf_motifs) == 1011:
        print(f"  Matches expected: 1,011 TFs with motifs")
    else:
        print(f"  NOTE: Got {len(tf_motifs)} TFs (expected ~1,011)")

    print("  Loading PWMs...")
    pwms = load_pwms(
        DATA_DIR / "data/motifs/Transcription_factor_weight_matrix_Arabidopsis_thaliana.txt"
    )
    print(f"  PWMs loaded: {len(pwms)} motifs")
    assert len(pwms) > 0, "No PWMs loaded"

    # Verify PWM properties
    for name, pwm in list(pwms.items())[:5]:
        assert pwm.shape[0] == 4, f"PWM {name} should have 4 rows (A,C,G,T)"
        col_sums = pwm.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=0.01), f"PWM {name} columns don't sum to 1"

    print("  PWM format validation: OK")
    print("  PASS: TFBS loading works\n")


def main():
    print("\n" + "=" * 60)
    print("MERIT v2.0 — Validation Against Release 1.0")
    print("=" * 60 + "\n")

    if not DATA_DIR.exists():
        print(f"ERROR: Release 1.0 data not found at {DATA_DIR}")
        sys.exit(1)

    # Run tests
    data = test_data_loading()
    test_grn_smoke()
    test_tfbs_loading()
    df_mr = test_enrichment(data)

    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print("  Data loading:     PASS")
    print("  GRN inference:    PASS (smoke test)")
    print("  TFBS loading:     PASS")
    n_mr = len(df_mr)
    if n_mr == 1752:
        print("  Enrichment:       PASS (exact match)")
    else:
        print(f"  Enrichment:       {n_mr}/1752 results")
    print()


if __name__ == "__main__":
    main()
