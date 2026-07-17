"""Loading and validation for split grouping artifacts."""

from pathlib import Path

import pandas as pd


SEQUENCE_CLUSTER_PROTEIN_COLUMN = "protein_id"
SEQUENCE_CLUSTER_COLUMN = "cluster_id"


def load_sequence_cluster_mapping(
    mapping_path: str | Path,
    required_proteins: set[str] | None = None,
) -> tuple[dict[str, str], pd.DataFrame, dict[str, int]]:
    """Load a canonical protein-to-sequence-cluster mapping CSV."""
    mapping_path = Path(mapping_path)
    mapping_df = pd.read_csv(
        mapping_path,
        dtype={
            SEQUENCE_CLUSTER_PROTEIN_COLUMN: "string",
            SEQUENCE_CLUSTER_COLUMN: "string",
        },
    )
    required_columns = {
        SEQUENCE_CLUSTER_PROTEIN_COLUMN,
        SEQUENCE_CLUSTER_COLUMN,
    }
    missing_columns = required_columns - set(mapping_df.columns)
    if missing_columns:
        raise ValueError(
            f"Sequence-cluster CSV is missing columns: {missing_columns}"
        )
    mapping_df = mapping_df[
        [SEQUENCE_CLUSTER_PROTEIN_COLUMN, SEQUENCE_CLUSTER_COLUMN]
    ].copy()
    for column in required_columns:
        mapping_df[column] = mapping_df[column].astype("string").str.strip()
        missing_mask = mapping_df[column].isna() | mapping_df[column].eq("")
        if missing_mask.any():
            raise ValueError(
                f"Sequence-cluster CSV has {int(missing_mask.sum())} "
                f"missing {column} value(s)."
            )
    duplicate_mask = mapping_df[
        SEQUENCE_CLUSTER_PROTEIN_COLUMN
    ].duplicated(keep=False)
    if duplicate_mask.any():
        examples = (
            mapping_df.loc[duplicate_mask, SEQUENCE_CLUSTER_PROTEIN_COLUMN]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Sequence-cluster CSV must contain one row per protein. "
            f"Duplicate examples: {examples}"
        )

    protein_to_group = dict(
        zip(
            mapping_df[SEQUENCE_CLUSTER_PROTEIN_COLUMN],
            mapping_df[SEQUENCE_CLUSTER_COLUMN],
        )
    )
    if required_proteins is not None:
        missing_proteins = set(required_proteins) - set(protein_to_group)
        if missing_proteins:
            examples = sorted(missing_proteins)[:10]
            raise ValueError(
                "Sequence-cluster CSV must map every eligible protein for "
                "C2/C3 splitting. Missing "
                f"{len(missing_proteins)} protein(s): {examples}"
            )
    mapping_df = mapping_df.sort_values(
        SEQUENCE_CLUSTER_PROTEIN_COLUMN
    ).reset_index(drop=True)
    metadata = {
        "n_mapped_proteins": len(mapping_df),
        "n_sequence_clusters": mapping_df[
            SEQUENCE_CLUSTER_COLUMN
        ].nunique(),
    }
    return protein_to_group, mapping_df, metadata
