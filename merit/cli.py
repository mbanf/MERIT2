"""MERIT v2.0 command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        prog="merit",
        description="MERIT v2.0 — MastEr Regulator Inference Tool",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument(
        "--step",
        choices=["all", "grn", "tfbs", "context", "enrichment"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument("--method", choices=["ensemble", "gnn"], default="ensemble",
                        help="GRN inference method")
    parser.add_argument("--cpus", type=int, default=4, help="Number of CPUs")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    if args.step in ("all", "grn"):
        _run_grn(config, args)
    if args.step in ("all", "tfbs"):
        _run_tfbs(config, args)
    if args.step in ("all", "context"):
        _run_context(config, args)
    if args.step in ("all", "enrichment"):
        _run_enrichment(config, args)


def _run_grn(config: dict, args):
    """Step 1: GRN inference."""
    from merit.utils.data import load_datasets

    logging.info("Step 1: GRN inference (%s)", args.method)
    data = load_datasets(**config["data"])

    if args.method == "ensemble":
        from merit.inference import infer_grn_ensemble

        params = config.get("grn", {})
        result = infer_grn_ensemble(
            expression=data.fold_change,
            genes=data.genes,
            regulators=data.regulators,
            n_cpus=args.cpus,
            **params,
        )
    elif args.method == "gnn":
        from merit.inference.gnn import infer_grn_gnn

        result = infer_grn_gnn(
            expression=data.fold_change,
            genes=data.genes,
            regulators=data.regulators,
            **config.get("gnn", {}),
        )

    import numpy as np

    np.save(args.output / "grn_lead_support.npy", result["lead_support"])
    logging.info("GRN saved to %s", args.output / "grn_lead_support.npy")


def _run_tfbs(config: dict, args):
    """Step 2: TFBS scanning."""
    from merit.tfbs import scan_tfbs

    logging.info("Step 2: TFBS scanning")
    tfbs_config = config.get("tfbs", {})
    tfs, genes, mat = scan_tfbs(**tfbs_config)

    import numpy as np

    np.save(args.output / "tfbs_matrix.npy", mat)
    logging.info("TFBS matrix saved")


def _run_context(config: dict, args):
    """Step 3: Context annotation."""
    logging.info("Step 3: Context-specific annotation")
    # Load GRN and TFBS from previous steps, combine, annotate
    # Implementation depends on caching strategy


def _run_enrichment(config: dict, args):
    """Step 4: Master regulator enrichment."""
    logging.info("Step 4: Master regulator enrichment")


if __name__ == "__main__":
    main()
