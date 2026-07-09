"""
Generic positive/negative edge-list PPI dataset loader.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from ppi_dataset_utils import (
    input_protein_count,
    read_fasta,
    read_table,
    sample_negative_pairs,
    validate_negative_ratio,
    write_prepared_dataset,
)


LOADER_NAME = "generic_edges"


def register_subcommand(subparsers: Any) -> None:
    """
    Register the generic_edges dataset-prep subcommand.
    """
    parser = subparsers.add_parser(
        LOADER_NAME,
        help="Prepare a generic positive/negative PPI edge-list dataset.",
    )
    add_common_args(parser)
    parser.add_argument("--positive-pairs", required=True)
    parser.add_argument("--negative-pairs", default=None)
    parser.add_argument("--protein-a-col", default="protein_a")
    parser.add_argument("--protein-b-col", default="protein_b")
    parser.add_argument("--sample-negatives", action="store_true")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.set_defaults(func=run)


def add_common_args(parser: Any) -> None:
    """
    Add shared dataset-prep arguments to one subcommand parser.
    """
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-dir", default="processed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")


def select_pair_columns(
        table: pd.DataFrame, protein_a_col: str, protein_b_col: str,
        label: int,
    ) -> pd.DataFrame:
    """
    Return canonical protein-pair columns from one edge-list table.
    """
    missing_columns = [
        column
        for column in (protein_a_col, protein_b_col)
        if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input table is missing required pair columns: "
            f"{missing_columns}")

    pairs = table[[protein_a_col, protein_b_col]].copy()
    pairs.columns = ["protein_a", "protein_b"]
    pairs["label"] = int(label)

    return pairs


def read_edge_pairs(
        input_path: str | Path, protein_a_col: str, protein_b_col: str,
        label: int,
    ) -> pd.DataFrame:
    """
    Read an edge list and assign one binary label.
    """
    table = read_table(input_path)
    pairs = select_pair_columns(
        table=table,
        protein_a_col=protein_a_col,
        protein_b_col=protein_b_col,
        label=label,
    )

    return pairs


def make_loader_specific_options(args: Any) -> dict[str, Any]:
    """
    Return generic_edges-specific options for dataset metadata.
    """
    options = {
        "positive_pairs": str(args.positive_pairs),
        "negative_pairs": (
            None if args.negative_pairs is None else str(args.negative_pairs)
        ),
        "protein_a_col": args.protein_a_col,
        "protein_b_col": args.protein_b_col,
        "sample_negatives": bool(args.sample_negatives),
        "negative_ratio": float(args.negative_ratio),
    }

    return options


def run(args: Any) -> None:
    """
    Prepare a generic edge-list PPI dataset.
    """
    validate_negative_ratio(args.negative_ratio)
    if args.negative_pairs is not None and args.sample_negatives:
        raise ValueError(
            "Pass either --negative-pairs or --sample-negatives, not both.")

    sequences = read_fasta(args.fasta)
    positive_pairs = read_edge_pairs(
        input_path=args.positive_pairs,
        protein_a_col=args.protein_a_col,
        protein_b_col=args.protein_b_col,
        label=1,
    )
    n_positive_input = int(len(positive_pairs))
    n_negative_input = 0
    target_n_negatives = 0
    n_sampled_negatives = 0
    input_paths = {
        "fasta": args.fasta,
        "positive_pairs": args.positive_pairs,
        "negative_pairs": args.negative_pairs,
    }

    if args.negative_pairs is not None:
        negative_pairs = read_edge_pairs(
            input_path=args.negative_pairs,
            protein_a_col=args.protein_a_col,
            protein_b_col=args.protein_b_col,
            label=0,
        )
        n_negative_input = int(len(negative_pairs))
        sampled_negatives = False
    elif args.sample_negatives:
        negative_pairs, sampling_metadata = sample_negative_pairs(
            positive_pairs=positive_pairs,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
        target_n_negatives = int(sampling_metadata["target_n_negatives"])
        n_sampled_negatives = int(sampling_metadata["n_sampled_negatives"])
        sampled_negatives = True
    else:
        raise ValueError(
            "generic_edges requires --negative-pairs or --sample-negatives.")

    raw_pairs = pd.concat(
        [positive_pairs, negative_pairs],
        ignore_index=True,
    )
    loader_metadata = {
        "n_positive_input": n_positive_input,
        "n_negative_input": n_negative_input,
        "n_unique_proteins_input": input_protein_count(raw_pairs),
        "n_pairs_after_loader_filters": int(len(raw_pairs)),
        "sampled_negatives": sampled_negatives,
        "negative_ratio": float(args.negative_ratio),
        "target_n_negatives": target_n_negatives,
        "n_sampled_negatives": n_sampled_negatives,
    }
    write_prepared_dataset(
        args=args,
        loader_name=LOADER_NAME,
        raw_pairs=raw_pairs,
        sequences=sequences,
        input_paths=input_paths,
        loader_metadata=loader_metadata,
        loader_specific_options=make_loader_specific_options(args),
    )
