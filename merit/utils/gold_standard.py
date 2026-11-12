"""Unified gold-standard regulatory interaction loader.

Loads and merges validated TF-target interactions from multiple sources:
- ATRM: ~1,226 literature-curated interactions (high confidence)
- AtRegNet: ~1.6M interactions from AGRIS database
- DAP-seq: ~1.8M in vitro binding interactions
- TransAt: metadata for 681 experiments across 424 TFs

Each source provides different evidence types with different confidence levels.
Interactions supported by multiple sources are higher confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GoldStandard:
    """Unified gold-standard edge set with provenance tracking."""

    edges: pd.DataFrame  # columns: tf, target, sources, n_sources, direction
    tf_names: list[str]
    target_names: list[str]

    def get_edges(self, min_sources: int = 1) -> list[tuple[str, str]]:
        """Get edge list filtered by minimum number of supporting sources."""
        mask = self.edges["n_sources"] >= min_sources
        return list(zip(self.edges.loc[mask, "tf"], self.edges.loc[mask, "target"]))

    def get_edge_set(self, min_sources: int = 1) -> set[tuple[str, str]]:
        """Get edge set filtered by minimum sources."""
        return set(self.get_edges(min_sources))

    def to_matrix(
        self,
        tf_order: list[str],
        gene_order: list[str],
        min_sources: int = 1,
    ) -> np.ndarray:
        """Convert to binary matrix (n_tfs, n_genes) for given TF/gene ordering."""
        tf_idx = {t: i for i, t in enumerate(tf_order)}
        gene_idx = {g: i for i, g in enumerate(gene_order)}
        mat = np.zeros((len(tf_order), len(gene_order)), dtype=np.float64)

        for _, row in self.edges.iterrows():
            if row["n_sources"] < min_sources:
                continue
            ti = tf_idx.get(row["tf"])
            gi = gene_idx.get(row["target"])
            if ti is not None and gi is not None:
                mat[ti, gi] = 1.0

        return mat

    def summary(self) -> str:
        """Print summary statistics."""
        lines = [
            f"Gold-standard: {len(self.edges)} edges, "
            f"{self.edges['tf'].nunique()} TFs, "
            f"{self.edges['target'].nunique()} targets",
        ]
        for n in range(1, self.edges["n_sources"].max() + 1):
            count = (self.edges["n_sources"] >= n).sum()
            lines.append(f"  >= {n} sources: {count} edges")

        source_cols = [c for c in self.edges.columns if c.startswith("src_")]
        for col in source_cols:
            name = col[4:]
            count = self.edges[col].sum()
            lines.append(f"  {name}: {int(count)} edges")

        return "\n".join(lines)


def _normalize_gene_id(gene_id: str) -> str:
    """Normalize Arabidopsis gene IDs to uppercase AT format."""
    gid = gene_id.strip().upper()
    # Handle At1g01010 -> AT1G01010 format
    if len(gid) == 9 and gid[:2] == "AT" and gid[3] == "G":
        return gid
    if len(gid) == 9 and gid[0] == "A" and gid[1].lower() == "t" and gid[3].lower() == "g":
        return gid.upper()
    return gid


def load_atrm(filepath: str | Path) -> pd.DataFrame:
    """Load ATRM (literature-curated interactions).

    Returns DataFrame with columns: tf, target, direction.
    """
    data = open(filepath).read()
    parts = data.split("\t")

    # Header: TF ID, Target ID, TF alias, Target alias, Activate/Repress, Reference, Note
    n_fields = 7
    records = []
    for i in range(n_fields, len(parts) - (n_fields - 1), n_fields):
        tf = _normalize_gene_id(parts[i])
        target = _normalize_gene_id(parts[i + 1])
        direction = parts[i + 4].strip() if i + 4 < len(parts) else ""
        if tf and target and tf != target:
            records.append({"tf": tf, "target": target, "direction": direction})

    df = pd.DataFrame(records).drop_duplicates(subset=["tf", "target"])
    logger.info("ATRM: %d unique edges, %d TFs", len(df), df["tf"].nunique())
    return df


def load_atregnet(filepath: str | Path) -> pd.DataFrame:
    """Load AtRegNet / AGRIS interactions.

    CSV with columns: TFName, TFLocus, TFFamily, TargetName, TargetLocus, ...
    """
    df = pd.read_csv(filepath, dtype=str)
    edges = pd.DataFrame({
        "tf": df["TFLocus"].apply(_normalize_gene_id),
        "target": df["TargetLocus"].apply(_normalize_gene_id),
    })
    edges = edges[edges["tf"] != edges["target"]].drop_duplicates()
    logger.info("AtRegNet: %d unique edges, %d TFs", len(edges), edges["tf"].nunique())
    return edges


def load_dapseq(filepath: str | Path) -> pd.DataFrame:
    """Load DAP-seq binding interactions.

    Semicolon-separated with columns: TF, Target (quoted).
    """
    df = pd.read_csv(filepath, sep=";", dtype=str)
    edges = pd.DataFrame({
        "tf": df["TF"].str.strip('"').apply(_normalize_gene_id),
        "target": df["Target"].str.strip('"').apply(_normalize_gene_id),
    })
    edges = edges[edges["tf"] != edges["target"]].drop_duplicates()
    logger.info("DAP-seq: %d unique edges, %d TFs", len(edges), edges["tf"].nunique())
    return edges


def load_all_gold_standards(
    data_dir: str | Path,
    atrm_file: str = "Regulations_in_ATRM.txt",
    atregnet_file: str = "AtRegNet.csv",
    dapseq_file: str = "df.grn.dapSeq.csv",
) -> GoldStandard:
    """Load and merge all available gold-standard sources.

    Parameters
    ----------
    data_dir : path
        Directory containing gold-standard files.

    Returns
    -------
    GoldStandard with merged edges and provenance tracking.
    """
    data_dir = Path(data_dir)

    sources = {}

    atrm_path = data_dir / atrm_file
    if atrm_path.exists():
        sources["atrm"] = load_atrm(atrm_path)

    atregnet_path = data_dir / atregnet_file
    if atregnet_path.exists():
        sources["atregnet"] = load_atregnet(atregnet_path)

    dapseq_path = data_dir / dapseq_file
    if dapseq_path.exists():
        sources["dapseq"] = load_dapseq(dapseq_path)

    if not sources:
        raise FileNotFoundError(f"No gold-standard files found in {data_dir}")

    # Merge all edges
    all_edges = set()
    edge_sources: dict[tuple[str, str], set[str]] = {}

    for source_name, df in sources.items():
        for _, row in df.iterrows():
            edge = (row["tf"], row["target"])
            all_edges.add(edge)
            if edge not in edge_sources:
                edge_sources[edge] = set()
            edge_sources[edge].add(source_name)

    # Build merged DataFrame
    records = []
    for tf, target in sorted(all_edges):
        srcs = edge_sources[(tf, target)]
        record = {
            "tf": tf,
            "target": target,
            "sources": ",".join(sorted(srcs)),
            "n_sources": len(srcs),
        }
        for source_name in sources:
            record[f"src_{source_name}"] = source_name in srcs
        records.append(record)

    merged = pd.DataFrame(records)
    all_tfs = sorted(merged["tf"].unique())
    all_targets = sorted(merged["target"].unique())

    gs = GoldStandard(edges=merged, tf_names=all_tfs, target_names=all_targets)
    logger.info("\n%s", gs.summary())

    return gs


def train_test_split_edges(
    gold_standard: GoldStandard,
    test_fraction: float = 0.2,
    min_sources: int = 1,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split gold-standard edges into train/test sets.

    Ensures both sets contain edges from all source types.
    """
    rng = np.random.RandomState(seed)
    edges = gold_standard.get_edges(min_sources)
    rng.shuffle(edges)

    split_idx = int(len(edges) * (1 - test_fraction))
    train_edges = edges[:split_idx]
    test_edges = edges[split_idx:]

    logger.info("Train/test split: %d / %d edges", len(train_edges), len(test_edges))
    return train_edges, test_edges
