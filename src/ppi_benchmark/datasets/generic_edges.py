"""
Generic positive/negative edge-list PPI dataset loader.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    add_common_loader_args,
    add_negative_construction_args,
    input_protein_count,
    read_fasta_with_taxa,
    read_table,
    require_taxonomy_for_negative_sampling,
    resolve_generated_negative_spec,
    sample_negative_pairs,
    write_prepared_dataset,
)
from ..splitting.negative_sampling import (
    SOURCE_PROVIDED_POLICY,
    source_provided_negative_construction,
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
    add_common_loader_args(parser)
    parser.add_argument("--positive-pairs", required=True)
    parser.add_argument("--protein-a-col", default="protein_a")
    parser.add_argument("--protein-b-col", default="protein_b")
    add_negative_construction_args(parser, allow_source_pairs=True)
    parser.set_defaults(func=run)


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
        "fasta_id_format": args.fasta_id_format,
        "sample_negatives": bool(args.sample_negatives),
        "negative_sampling_policy": (
            args.negative_sampling_policy
            if args.sample_negatives else SOURCE_PROVIDED_POLICY
        ),
        "negative_ratio": (
            float(args.negative_ratio) if args.sample_negatives else None
        ),
        "negative_sampling_seed": (
            int(args.negative_sampling_seed) if args.sample_negatives else None
        ),
    }

    return options


def run(args: Any) -> None:
    """
    Prepare a generic edge-list PPI dataset.
    """
    if args.negative_pairs is not None and args.sample_negatives:
        raise ValueError(
            "Pass either --negative-pairs or --sample-negatives, not both.")
    if args.negative_pairs is None and not args.sample_negatives:
        raise ValueError(
            "generic_edges requires --negative-pairs or --sample-negatives."
        )
    if args.negative_pairs is not None:
        unused_options = []
        if args.negative_sampling_policy is not None:
            unused_options.append("--negative-sampling-policy")
        if args.negative_ratio is not None:
            unused_options.append("--negative-ratio")
        if args.negative_sampling_seed is not None:
            unused_options.append("--negative-sampling-seed")
        if unused_options:
            raise ValueError(
                f"{', '.join(unused_options)} can only be used with "
                "--sample-negatives."
            )
    sampling_spec = None
    if args.sample_negatives:
        sampling_spec = resolve_generated_negative_spec(args)

    fasta_data = read_fasta_with_taxa(
        fasta_path=args.fasta,
        id_format=args.fasta_id_format,
        protein_metadata_path=args.protein_metadata,
        taxon_id=args.taxon_id,
    )
    sequences = fasta_data.sequences
    if sampling_spec is not None:
        require_taxonomy_for_negative_sampling(
            sampling_spec,
            fasta_data.taxon_ids,
        )
    positive_pairs = read_edge_pairs(
        input_path=args.positive_pairs,
        protein_a_col=args.protein_a_col,
        protein_b_col=args.protein_b_col,
        label=1,
    )
    n_positive_input = int(len(positive_pairs))
    n_negative_input = 0
    sampling_metadata = {
        "target_n_negatives": 0,
        "n_sampled_negatives": 0,
    }
    input_paths = {
        "fasta": args.fasta,
        "positive_pairs": args.positive_pairs,
        "negative_pairs": args.negative_pairs,
        "protein_metadata": args.protein_metadata,
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
        sampling_metadata.update({
            "negative_construction": source_provided_negative_construction(),
            "negative_sampling_policy": SOURCE_PROVIDED_POLICY,
            "negative_sampling_seed": None,
            "species_aware_sampling": False,
        })
    else:
        assert sampling_spec is not None
        negative_pairs, sampling_metadata = sample_negative_pairs(
            positive_pairs=positive_pairs,
            spec=sampling_spec,
            allowed_protein_ids=sequences,
            protein_taxa=fasta_data.taxon_ids,
        )
        sampled_negatives = True

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
        "negative_ratio": (
            float(args.negative_ratio) if sampled_negatives else None
        ),
        **sampling_metadata,
    }
    write_prepared_dataset(
        args=args,
        loader_name=LOADER_NAME,
        raw_pairs=raw_pairs,
        sequences=sequences,
        input_paths=input_paths,
        loader_metadata=loader_metadata,
        loader_specific_options=make_loader_specific_options(args),
        protein_taxa=fasta_data.taxon_ids,
    )
