"""Data loading utilities — port of load_datasets.R."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MeritData:
    """Container for all MERIT input data."""

    genes: list[str]
    experiment_ids: list[str]
    fold_change: np.ndarray  # genes x experiments
    pvalues: np.ndarray  # genes x experiments
    experiment_annotation: pd.DataFrame
    tf_annotation: pd.DataFrame
    condition_treatments: dict[str, int]  # treatment -> count
    condition_tissues: dict[str, int]  # tissue -> count
    regulators: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.regulators:
            tfs_with_expr = self.tf_annotation.loc[
                self.tf_annotation["TF_ID"].isin(self.genes), "TF_ID"
            ]
            self.regulators = sorted(tfs_with_expr.unique().tolist())


def load_datasets(
    genes_file: str | Path,
    experiment_ids_file: str | Path,
    fold_change_file: str | Path,
    pvalue_file: str | Path,
    experiment_annotation_file: str | Path,
    tf_annotation_file: str | Path,
) -> MeritData:
    """Load all MERIT input data from flat files.

    Direct port of load_datasets() from R/load_datasets.R.
    """
    genes = pd.read_csv(genes_file, header=None, sep="\t")[0].tolist()
    experiment_ids = pd.read_csv(experiment_ids_file, header=None, sep="\t")[0].astype(str).tolist()

    fold_change_df = pd.read_csv(fold_change_file, header=None, sep="\t")
    pvalue_df = pd.read_csv(pvalue_file, header=None, sep="\t")

    fold_change = fold_change_df.values.astype(np.float64)
    pvalues = pvalue_df.values.astype(np.float64)

    if fold_change.shape != (len(genes), len(experiment_ids)):
        raise ValueError(
            f"Fold-change matrix shape {fold_change.shape} doesn't match "
            f"genes ({len(genes)}) x experiments ({len(experiment_ids)})"
        )

    # Experiment annotation
    annotation = pd.read_csv(experiment_annotation_file, sep="\t")
    required_cols = [
        "series_id",
        "condition_treatment_1",
        "condition_treatment_2",
        "condition_tissue",
        "unique_ID",
    ]
    missing = set(required_cols) - set(annotation.columns)
    if missing:
        raise ValueError(f"Missing required columns in experiment annotation: {missing}")

    annotation = annotation.dropna(subset=["unique_ID"])
    annotation = annotation[annotation["unique_ID"].astype(str).isin(experiment_ids)]

    # Treatment and tissue counts
    treatments_1 = annotation["condition_treatment_1"].dropna().tolist()
    treatments_2 = annotation["condition_treatment_2"].dropna().tolist()
    all_treatments = [t for t in treatments_1 + treatments_2 if t != ""]
    condition_treatments = {}
    for t in set(all_treatments):
        condition_treatments[t] = all_treatments.count(t)

    tissue_counts = annotation["condition_tissue"].value_counts().to_dict()

    # TF annotation
    tf_annotation = pd.read_csv(tf_annotation_file, sep="\t")
    tf_annotation["with_geneExpression"] = tf_annotation["TF_ID"].isin(genes).map(
        {True: "yes", False: "no"}
    )

    return MeritData(
        genes=genes,
        experiment_ids=experiment_ids,
        fold_change=fold_change,
        pvalues=pvalues,
        experiment_annotation=annotation,
        tf_annotation=tf_annotation,
        condition_treatments=condition_treatments,
        condition_tissues=tissue_counts,
    )
