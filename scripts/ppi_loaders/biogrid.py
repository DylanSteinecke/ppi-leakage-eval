"""
BioGRID PPI dataset loader.
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


LOADER_NAME = "biogrid"
PROTEIN_A_CANDIDATES = (
    "SWISS-PROT Accessions Interactor A",
    "Systematic Name Interactor A",
    "Official Symbol Interactor A",
    "BioGRID ID Interactor A",
    "Entrez Gene Interactor A",
)
PROTEIN_B_CANDIDATES = (
    "SWISS-PROT Accessions Interactor B",
    "Systematic Name Interactor B",
    "Official Symbol Interactor B",
    "BioGRID ID Interactor B",
    "Entrez Gene Interactor B",
)
AMBIGUOUS_ID_SEPARATORS = ("|", ";", ",")


def register_subcommand(subparsers: Any) -> None:
    """
    Register the BioGRID dataset-prep subcommand.
    """
    parser = subparsers.add_parser(
        LOADER_NAME,
        help=(
            "Prepare a BioGRID-like PPI table. Selected interactor columns "
            "must already match FASTA IDs."
        ),
    )
    add_common_args(parser)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--protein-a-col", default=None)
    parser.add_argument("--protein-b-col", default=None)
    parser.add_argument("--organism-a-col", default=None)
    parser.add_argument("--organism-b-col", default=None)
    parser.add_argument("--organism-id", default=None)
    parser.add_argument("--experimental-system-type-col", default=None)
    parser.add_argument("--allowed-system-types", nargs="*", default=None)
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


def infer_biogrid_column(
        table: pd.DataFrame, explicit_column: str | None,
        candidates: tuple[str, ...], role_name: str, arg_name: str,
    ) -> str:
    """
    Return an explicit or unambiguously inferred BioGRID column name.
    """
    if explicit_column is not None:
        if explicit_column not in table.columns:
            raise ValueError(
                f"{arg_name}='{explicit_column}' is not present in the "
                "BioGRID table.")
        return explicit_column

    matches = [column for column in candidates if column in table.columns]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Could not infer {role_name} because multiple known BioGRID "
            f"columns are present: {matches}. Pass {arg_name} explicitly.")

    raise ValueError(
        f"Could not infer {role_name}. Pass {arg_name} explicitly. Known "
        f"candidate columns are: {list(candidates)}.")


def require_columns(table: pd.DataFrame, columns: list[str]) -> None:
    """
    Raise if any requested columns are absent.
    """
    missing_columns = [
        column for column in columns if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"BioGRID table is missing required columns: {missing_columns}")


def reject_ambiguous_interactor_values(
        table: pd.DataFrame, columns: list[str],
    ) -> None:
    """
    Fail on simple multi-ID interactor values instead of guessing a mapping.
    """
    for column in columns:
        values = table[column].dropna().astype(str)
        ambiguous_mask = values.apply(
            lambda value: any(separator in value
                              for separator in AMBIGUOUS_ID_SEPARATORS)
        )
        if ambiguous_mask.any():
            examples = values.loc[ambiguous_mask].head(10).tolist()
            raise ValueError(
                f"BioGRID column '{column}' contains ambiguous multi-ID "
                f"values. Pass columns that already match FASTA IDs or "
                f"preprocess ID mapping outside this loader. Examples: "
                f"{examples}")


def apply_biogrid_filters(table: pd.DataFrame, args: Any) -> pd.DataFrame:
    """
    Apply optional BioGRID organism and experimental-system filters.
    """
    filtered = table.copy()
    if args.organism_id is not None:
        if args.organism_a_col is None or args.organism_b_col is None:
            raise ValueError(
                "--organism-id requires --organism-a-col and "
                "--organism-b-col.")
        require_columns(filtered, [args.organism_a_col, args.organism_b_col])
        organism_id = str(args.organism_id)
        organism_mask = (
            filtered[args.organism_a_col].astype(str).eq(organism_id)
            & filtered[args.organism_b_col].astype(str).eq(organism_id)
        )
        filtered = filtered.loc[organism_mask].copy()

    allowed_system_types = args.allowed_system_types
    if allowed_system_types:
        if args.experimental_system_type_col is None:
            raise ValueError(
                "--allowed-system-types requires "
                "--experimental-system-type-col.")
        require_columns(filtered, [args.experimental_system_type_col])
        allowed_values = {str(value) for value in allowed_system_types}
        system_type_mask = filtered[
            args.experimental_system_type_col
        ].astype(str).isin(allowed_values)
        filtered = filtered.loc[system_type_mask].copy()

    return filtered.reset_index(drop=True)


def positive_pairs_from_table(
        table: pd.DataFrame, protein_a_col: str, protein_b_col: str,
    ) -> pd.DataFrame:
    """
    Return positive pair rows from filtered BioGRID interactions.
    """
    require_columns(table, [protein_a_col, protein_b_col])
    reject_ambiguous_interactor_values(
        table=table,
        columns=[protein_a_col, protein_b_col],
    )
    pairs = table[[protein_a_col, protein_b_col]].copy()
    pairs.columns = ["protein_a", "protein_b"]
    pairs["label"] = 1

    return pairs


def make_loader_specific_options(
        args: Any, protein_a_col: str, protein_b_col: str,
        n_positive_after_loader_filters: int,
    ) -> dict[str, Any]:
    """
    Return BioGRID-specific options for dataset metadata.
    """
    options = {
        "interactions": str(args.interactions),
        "protein_a_col": protein_a_col,
        "protein_b_col": protein_b_col,
        "organism_a_col": args.organism_a_col,
        "organism_b_col": args.organism_b_col,
        "organism_id": args.organism_id,
        "experimental_system_type_col": args.experimental_system_type_col,
        "allowed_system_types": args.allowed_system_types,
        "sample_negatives": bool(args.sample_negatives),
        "negative_ratio": float(args.negative_ratio),
        "n_positive_after_loader_filters": int(
            n_positive_after_loader_filters),
        "id_mapping_implemented": False,
    }

    return options


def run(args: Any) -> None:
    """
    Prepare a BioGRID PPI dataset.
    """
    validate_negative_ratio(args.negative_ratio)
    if not args.sample_negatives:
        raise ValueError(
            "BioGRID provides positives only in this loader version; pass "
            "--sample-negatives.")

    sequences = read_fasta(args.fasta)
    table = read_table(args.interactions)
    n_positive_input = int(len(table))
    protein_a_col = infer_biogrid_column(
        table=table,
        explicit_column=args.protein_a_col,
        candidates=PROTEIN_A_CANDIDATES,
        role_name="BioGRID protein_a column",
        arg_name="--protein-a-col",
    )
    protein_b_col = infer_biogrid_column(
        table=table,
        explicit_column=args.protein_b_col,
        candidates=PROTEIN_B_CANDIDATES,
        role_name="BioGRID protein_b column",
        arg_name="--protein-b-col",
    )
    filtered_table = apply_biogrid_filters(table, args)
    positive_pairs = positive_pairs_from_table(
        table=filtered_table,
        protein_a_col=protein_a_col,
        protein_b_col=protein_b_col,
    )
    negative_pairs, sampling_metadata = sample_negative_pairs(
        positive_pairs=positive_pairs,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    raw_pairs = pd.concat(
        [positive_pairs, negative_pairs],
        ignore_index=True,
    )
    loader_metadata = {
        "n_positive_input": n_positive_input,
        "n_negative_input": 0,
        "n_unique_proteins_input": input_protein_count(raw_pairs),
        "n_pairs_after_loader_filters": int(len(raw_pairs)),
        "sampled_negatives": True,
        "negative_ratio": float(args.negative_ratio),
        "target_n_negatives": int(sampling_metadata["target_n_negatives"]),
        "n_sampled_negatives": int(sampling_metadata["n_sampled_negatives"]),
    }
    write_prepared_dataset(
        args=args,
        loader_name=LOADER_NAME,
        raw_pairs=raw_pairs,
        sequences=sequences,
        input_paths={
            "fasta": args.fasta,
            "interactions": Path(args.interactions),
        },
        loader_metadata=loader_metadata,
        loader_specific_options=make_loader_specific_options(
            args=args,
            protein_a_col=protein_a_col,
            protein_b_col=protein_b_col,
            n_positive_after_loader_filters=len(positive_pairs),
        ),
    )
