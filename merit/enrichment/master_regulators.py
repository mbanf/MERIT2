"""Master regulator identification — port of extract_context_specific_*_master_regulators() from main.R (Step 4)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

logger = logging.getLogger(__name__)


def find_bottom_level_master_regulators(
    subnetworks: dict[str, np.ndarray],
    tf_names: list[str],
    gene_names: list[str],
    gene_groups: pd.DataFrame,
    th_pval: float = 0.05,
) -> pd.DataFrame:
    """Identify TFs enriched for regulating specific metabolic domains.

    Port of extract_context_specific_bottom_level_master_regulators().

    For each context-specific subnetwork, for each TF, tests whether its
    targets are enriched for genes in each metabolic domain (hypergeometric).

    Parameters
    ----------
    subnetworks : dict[str, np.ndarray]
        Context name -> binary matrix (n_tfs x n_genes).
    tf_names : list[str]
        TF names (rows of matrices).
    gene_names : list[str]
        Gene names (columns of matrices).
    gene_groups : pd.DataFrame
        Must have columns 'ID', 'Domain', 'Group'.
    th_pval : float
        P-value threshold.

    Returns
    -------
    pd.DataFrame with columns: tf, domain, fc, pval, context
    """
    results = []

    for context_name, mat in subnetworks.items():
        logger.debug("Bottom-level MR: %s", context_name)

        # All genes in this subnetwork
        edge_mask = mat > 0
        active_tfs = np.where(edge_mask.any(axis=1))[0]
        active_genes_mask = edge_mask.any(axis=0)

        # Population: all unique genes in this network (TFs + targets)
        net_genes = set()
        for tf_i in active_tfs:
            net_genes.add(tf_names[tf_i])
            for g_j in np.where(edge_mask[tf_i])[0]:
                net_genes.add(gene_names[g_j])
        pop_size = len(net_genes)

        if pop_size == 0:
            continue

        # Gene group membership restricted to this network
        gene_groups_in_net = gene_groups[gene_groups["ID"].isin(net_genes)]
        domains = gene_groups_in_net["Domain"].unique()

        for tf_i in active_tfs:
            tf_name = tf_names[tf_i]
            target_indices = np.where(edge_mask[tf_i])[0]
            target_names = {gene_names[j] for j in target_indices}
            sample_size = len(target_names)

            # Which domains are represented in targets?
            targets_with_domain = gene_groups[gene_groups["ID"].isin(target_names)]

            for domain in domains:
                domain_targets = targets_with_domain[targets_with_domain["Domain"] == domain]
                hit_in_sample = len(domain_targets)

                if hit_in_sample == 0:
                    continue

                # Domain genes in this network
                domain_in_pop = gene_groups_in_net[gene_groups_in_net["Domain"] == domain]
                hit_in_pop = len(domain_in_pop)
                fail_in_pop = pop_size - hit_in_pop

                if hit_in_pop == 0:
                    continue

                fc = (hit_in_sample / sample_size) / (hit_in_pop / pop_size)
                pval = hypergeom.sf(
                    hit_in_sample - 1,
                    pop_size,
                    hit_in_pop,
                    sample_size,
                )

                results.append({
                    "tf": tf_name,
                    "domain": domain,
                    "fc": fc,
                    "pval": pval,
                    "context": context_name,
                })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("pval")
        df = df[df["pval"] <= th_pval]
    logger.info("Bottom-level MRs: %d significant results", len(df))
    return df


def find_top_level_master_regulators(
    subnetworks: dict[str, np.ndarray],
    tf_names: list[str],
    gene_names: list[str],
    th_min_number: int = 2,
    th_pval: float = 0.05,
) -> pd.DataFrame:
    """Identify hub TFs enriched for regulating other TFs.

    Port of extract_context_specific_top_level_master_regulators().

    For each context, tests whether a TF's targets are enriched for other TFs
    (hypergeometric test).

    Returns
    -------
    pd.DataFrame with columns: tf, n_tgs, fc, pval, context
    """
    tf_set = set(tf_names)
    results = []

    for context_name, mat in subnetworks.items():
        edge_mask = mat > 0
        active_tfs = np.where(edge_mask.any(axis=1))[0]

        # All target genes in this network
        all_targets = set()
        for tf_i in active_tfs:
            for g_j in np.where(edge_mask[tf_i])[0]:
                all_targets.add(gene_names[g_j])

        pop_size = len(all_targets)
        if pop_size == 0:
            continue

        # TFs among targets
        tfs_in_targets = all_targets & tf_set
        hit_in_pop = len(tfs_in_targets)

        if hit_in_pop < th_min_number:
            continue

        fail_in_pop = pop_size - hit_in_pop

        for tf_i in active_tfs:
            tf_name = tf_names[tf_i]
            target_indices = np.where(edge_mask[tf_i])[0]
            target_names = {gene_names[j] for j in target_indices}
            sample_size = len(target_names)

            # How many of this TF's targets are themselves TFs?
            hit_in_sample = len(target_names & tf_set)
            if hit_in_sample < th_min_number:
                continue

            fc = (hit_in_sample / sample_size) / (hit_in_pop / pop_size)
            pval = hypergeom.sf(hit_in_sample - 1, pop_size, hit_in_pop, sample_size)

            results.append({
                "tf": tf_name,
                "n_tgs": hit_in_sample,
                "fc": fc,
                "pval": pval,
                "context": context_name,
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("pval")
        df = df[df["pval"] <= th_pval]
    logger.info("Top-level MRs: %d significant results", len(df))
    return df
