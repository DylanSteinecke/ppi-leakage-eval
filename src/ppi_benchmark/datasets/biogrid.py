"""
BioGRID PPI dataset loader.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    add_common_loader_args,
    apply_id_mapping_to_pairs,
    file_sha256,
    input_protein_count,
    iter_table_chunks,
    read_fasta_with_taxa,
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


def register_subcommand(subparsers: Any) -> None:
    """
    Register the BioGRID dataset-prep subcommand.
    """
    parser = subparsers.add_parser(
        LOADER_NAME,
        help=(
            "Prepare a BioGRID-like PPI table. Selected interactor columns "
            "must already match FASTA IDs or be mapped with --id-map."
        ),
    )
    add_common_loader_args(parser)
    parser.add_argument("--interactions", required=True)
    parser.add_argument(
        "--archive-member",
        default=None,
        help="Table member to read when --interactions is a multi-file ZIP.",
    )
    parser.add_argument(
        "--table-chunksize",
        type=int,
        default=100_000,
        help="Rows read at once from the BioGRID interaction table.",
    )
    parser.add_argument("--protein-a-col", default=None)
    parser.add_argument("--protein-b-col", default=None)
    parser.add_argument("--id-map", default=None)
    parser.add_argument("--map-from-col", default=None)
    parser.add_argument("--map-to-col", default=None)
    parser.add_argument("--organism-a-col", default=None)
    parser.add_argument("--organism-b-col", default=None)
    parser.add_argument("--organism-id", default=None)
    parser.add_argument("--experimental-system-type-col", default=None)
    parser.add_argument("--allowed-system-types", nargs="+", default=None)
    parser.add_argument(
        "--ambiguous-id-policy",
        choices=("error", "drop"),
        default="error",
        help="How rows containing multi-ID interactor values are handled.",
    )
    parser.add_argument("--sample-negatives", action="store_true")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.set_defaults(func=run)


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


def validate_id_mapping_args(args: Any) -> None:
    """
    Validate optional explicit ID-mapping arguments.
    """
    if args.id_map is None:
        if args.map_from_col is not None or args.map_to_col is not None:
            raise ValueError(
                "--map-from-col and --map-to-col can only be used with "
                "--id-map.")
        return

    if args.map_from_col is None or args.map_to_col is None:
        raise ValueError("--id-map requires --map-from-col and --map-to-col.")


def filter_ambiguous_interactor_values(
        table: pd.DataFrame, columns: list[str],
        policy: str,
    ) -> tuple[pd.DataFrame, int]:
    """
    Reject or drop simple multi-ID interactor values without guessing a mapping.
    """
    if policy not in {"error", "drop"}:
        raise ValueError(
            "ambiguous interactor policy must be 'error' or 'drop'.")

    require_columns(table, columns)
    separator_pattern = "[|;,]"
    ambiguous_by_column = table[columns].apply(
        lambda values: values.astype("string").str.contains(
            separator_pattern,
            regex=True,
            na=False,
        )
    )
    ambiguous_mask = ambiguous_by_column.any(axis=1)
    n_ambiguous = int(ambiguous_mask.sum())
    if n_ambiguous == 0:
        return table, 0

    if policy == "error":
        examples = (
            table.loc[ambiguous_mask, columns]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"BioGRID interactor columns contain {n_ambiguous} rows with "
            "ambiguous multi-ID values. Pass --ambiguous-id-policy drop or "
            f"preprocess the mapping explicitly. Examples: {examples}")

    filtered = table.loc[~ambiguous_mask].copy()

    return filtered, n_ambiguous


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
    pairs = table[[protein_a_col, protein_b_col]].copy()
    pairs.columns = ["protein_a", "protein_b"]
    pairs["label"] = 1

    return pairs


def read_biogrid_positive_pairs(
        args: Any,
    ) -> tuple[pd.DataFrame, str, str, int, int]:
    """
    Stream selected BioGRID columns and return filtered positive pairs.
    """
    if args.table_chunksize < 1:
        raise ValueError("--table-chunksize must be at least 1.")

    table_header = read_table(
        args.interactions,
        archive_member=args.archive_member,
        nrows=0,
    )
    protein_a_col = infer_biogrid_column(
        table=table_header,
        explicit_column=args.protein_a_col,
        candidates=PROTEIN_A_CANDIDATES,
        role_name="BioGRID protein_a column",
        arg_name="--protein-a-col",
    )
    protein_b_col = infer_biogrid_column(
        table=table_header,
        explicit_column=args.protein_b_col,
        candidates=PROTEIN_B_CANDIDATES,
        role_name="BioGRID protein_b column",
        arg_name="--protein-b-col",
    )
    selected_columns = [protein_a_col, protein_b_col]
    if args.organism_id is not None:
        if args.organism_a_col is None or args.organism_b_col is None:
            raise ValueError(
                "--organism-id requires --organism-a-col and "
                "--organism-b-col.")
        selected_columns.extend([args.organism_a_col, args.organism_b_col])
    if args.allowed_system_types:
        if args.experimental_system_type_col is None:
            raise ValueError(
                "--allowed-system-types requires "
                "--experimental-system-type-col.")
        selected_columns.append(args.experimental_system_type_col)
    selected_columns = list(dict.fromkeys(selected_columns))
    require_columns(table_header, selected_columns)

    positive_pair_frames = []
    n_positive_input = 0
    n_pairs_dropped_ambiguous = 0
    for table_chunk in iter_table_chunks(
            args.interactions,
            chunksize=args.table_chunksize,
            archive_member=args.archive_member,
            usecols=selected_columns):
        n_positive_input += len(table_chunk)
        filtered_chunk = apply_biogrid_filters(table_chunk, args)
        filtered_chunk, n_ambiguous = filter_ambiguous_interactor_values(
            table=filtered_chunk,
            columns=[protein_a_col, protein_b_col],
            policy=args.ambiguous_id_policy,
        )
        n_pairs_dropped_ambiguous += n_ambiguous
        if not filtered_chunk.empty:
            positive_pair_frames.append(positive_pairs_from_table(
                table=filtered_chunk,
                protein_a_col=protein_a_col,
                protein_b_col=protein_b_col,
            ))

    if positive_pair_frames:
        positive_pairs = pd.concat(positive_pair_frames, ignore_index=True)
    else:
        positive_pairs = pd.DataFrame(
            columns=["protein_a", "protein_b", "label"])

    return (
        positive_pairs,
        protein_a_col,
        protein_b_col,
        int(n_positive_input),
        int(n_pairs_dropped_ambiguous),
    )


def make_loader_specific_options(
        args: Any, protein_a_col: str, protein_b_col: str,
        n_positive_after_loader_filters: int,
    ) -> dict[str, Any]:
    """
    Return BioGRID-specific options for dataset metadata.
    """
    options = {
        "interactions": str(args.interactions),
        "archive_member": args.archive_member,
        "table_chunksize": int(args.table_chunksize),
        "protein_a_col": protein_a_col,
        "protein_b_col": protein_b_col,
        "fasta_id_format": args.fasta_id_format,
        "organism_a_col": args.organism_a_col,
        "organism_b_col": args.organism_b_col,
        "organism_id": args.organism_id,
        "experimental_system_type_col": args.experimental_system_type_col,
        "allowed_system_types": args.allowed_system_types,
        "sample_negatives": bool(args.sample_negatives),
        "negative_ratio": float(args.negative_ratio),
        "n_positive_after_loader_filters": int(
            n_positive_after_loader_filters),
        "ambiguous_id_policy": args.ambiguous_id_policy,
        "id_mapping_used": args.id_map is not None,
        "id_map": None if args.id_map is None else str(args.id_map),
        "map_from_col": args.map_from_col,
        "map_to_col": args.map_to_col,
    }

    return options


def run(args: Any) -> None:
    """
    Prepare a BioGRID PPI dataset.
    """
    validate_id_mapping_args(args)
    validate_negative_ratio(args.negative_ratio)
    if not args.sample_negatives:
        raise ValueError(
            "BioGRID provides positives only in this loader version; pass "
            "--sample-negatives.")

    fasta_data = read_fasta_with_taxa(
        fasta_path=args.fasta,
        id_format=args.fasta_id_format,
        protein_metadata_path=args.protein_metadata,
        taxon_id=args.taxon_id or args.organism_id,
    )
    sequences = fasta_data.sequences
    (
        positive_pairs,
        protein_a_col,
        protein_b_col,
        n_positive_input,
        n_pairs_dropped_ambiguous_interactor,
    ) = read_biogrid_positive_pairs(
        args,
    )
    n_positive_after_loader_filters = int(len(positive_pairs))
    id_mapping_metadata = {
        "id_mapping_used": args.id_map is not None,
        "id_map_path": None if args.id_map is None else str(args.id_map),
        "id_map_sha256": (
            None if args.id_map is None else file_sha256(args.id_map)
        ),
        "map_from_col": args.map_from_col,
        "map_to_col": args.map_to_col,
    }
    if args.id_map is not None:
        id_map_table = read_table(args.id_map)
        mapped_positive_pairs = apply_id_mapping_to_pairs(
            pairs=positive_pairs,
            mapping_table=id_map_table,
            map_from_col=args.map_from_col,
            map_to_col=args.map_to_col,
        )
        positive_pairs = mapped_positive_pairs.pairs
        id_mapping_metadata.update(mapped_positive_pairs.metadata)

    negative_pairs, sampling_metadata = sample_negative_pairs(
        positive_pairs=positive_pairs,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        allowed_protein_ids=sequences,
        protein_taxa=fasta_data.taxon_ids,
    )
    raw_pairs = pd.concat(
        [positive_pairs, negative_pairs],
        ignore_index=True,
    )
    loader_metadata = {
        "n_positive_input": n_positive_input,
        "n_negative_input": 0,
        "n_unique_proteins_input": input_protein_count(raw_pairs),
        "n_positive_after_id_mapping": int(len(positive_pairs)),
        "n_pairs_dropped_ambiguous_interactor": (
            n_pairs_dropped_ambiguous_interactor),
        "n_pairs_after_loader_filters": int(len(raw_pairs)),
        "sampled_negatives": True,
        "negative_ratio": float(args.negative_ratio),
        **sampling_metadata,
        **id_mapping_metadata,
    }
    write_prepared_dataset(
        args=args,
        loader_name=LOADER_NAME,
        raw_pairs=raw_pairs,
        sequences=sequences,
        input_paths={
            "fasta": args.fasta,
            "interactions": Path(args.interactions),
            "id_map": args.id_map,
            "protein_metadata": args.protein_metadata,
        },
        loader_metadata=loader_metadata,
        loader_specific_options=make_loader_specific_options(
            args=args,
            protein_a_col=protein_a_col,
            protein_b_col=protein_b_col,
            n_positive_after_loader_filters=n_positive_after_loader_filters,
        ),
        protein_taxa=fasta_data.taxon_ids,
    )
