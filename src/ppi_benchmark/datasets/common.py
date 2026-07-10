"""
Shared helpers for preparing PPI datasets.

Dataset loaders convert raw sources into pair tables with ``protein_a``,
``protein_b``, and ``label`` columns. This module owns the common
canonicalization, FASTA filtering, metadata, and output writing steps.
"""

import bz2
import gzip
import hashlib
import json
import lzma
import math
import random
import shlex
import sys
import zipfile
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import pandas as pd


PAIRS_FILENAME = "pairs.csv"
FASTA_FILENAME = "proteins.fasta"
PROTEIN_METADATA_FILENAME = "protein_metadata.csv"
METADATA_FILENAME = "dataset_metadata.json"
CANONICAL_PAIR_COLUMNS = ["pair_id", "protein_a", "protein_b", "label"]
PROTEIN_METADATA_COLUMNS = ["protein_id", "taxon_id"]
REQUIRED_PAIR_COLUMNS = {"protein_a", "protein_b", "label"}
SUPPORTED_TABLE_SUFFIXES = {
    ".csv": ",",
    ".tab": "\t",
    ".tsv": "\t",
    ".txt": "\t",
}
COMPRESSED_TEXT_SUFFIXES = {".bz2", ".gz", ".xz"}
FASTA_ID_FORMAT_CHOICES = ("first_token", "uniprot_accession")


@dataclass(frozen=True)
class PairProcessingResult:
    """
    Processed pairs plus JSON-friendly count metadata.
    """
    pairs: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FastaData:
    """
    FASTA sequences plus taxon IDs parsed from record headers.
    """
    sequences: dict[str, str]
    taxon_ids: dict[str, str]


def open_text_auto(input_path: str | Path) -> TextIO:
    """
    Open plain or gzip/bzip2/xz-compressed text for reading.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".gz":
        fin = gzip.open(input_path, "rt", encoding="utf-8")
    elif suffix == ".bz2":
        fin = bz2.open(input_path, "rt", encoding="utf-8")
    elif suffix == ".xz":
        fin = lzma.open(input_path, "rt", encoding="utf-8")
    else:
        fin = input_path.open("r", encoding="utf-8")

    return fin


def fasta_record_id(header: str, id_format: str) -> str:
    """
    Return the requested sequence ID from one FASTA header.
    """
    if id_format not in FASTA_ID_FORMAT_CHOICES:
        raise ValueError(
            f"Unknown FASTA ID format '{id_format}'. Expected one of "
            f"{list(FASTA_ID_FORMAT_CHOICES)}.")

    header_parts = header.split()
    if not header_parts:
        raise ValueError("FASTA record header is missing a sequence ID.")

    sequence_id = header_parts[0]
    if id_format == "uniprot_accession":
        token_parts = sequence_id.split("|")
        if (
                len(token_parts) < 3
                or token_parts[0] not in {"sp", "tr"}
                or not token_parts[1]):
            raise ValueError(
                "Expected a UniProt FASTA header such as "
                f"'sp|P12345|NAME', found '{sequence_id}'.")
        sequence_id = token_parts[1]

    return sequence_id


def normalize_taxon_id(value: Any, context: str = "Taxon ID") -> str:
    """
    Return one canonical positive NCBI taxonomy ID.
    """
    taxon_id = str(value).strip()
    if not taxon_id.isdecimal() or int(taxon_id) < 1:
        raise ValueError(
            f"{context} must be a positive integer, found '{taxon_id}'.")

    return str(int(taxon_id))


def fasta_record_taxon_id(header: str) -> str | None:
    """
    Return the NCBI taxonomy ID from a UniProt ``OX=`` header field.
    """
    taxon_ids = {
        normalize_taxon_id(token[3:], context="FASTA OX taxon ID")
        for token in header.split()
        if token.startswith("OX=")
    }
    if len(taxon_ids) > 1:
        raise ValueError(
            "FASTA record header contains conflicting OX taxon IDs: "
            f"{sorted(taxon_ids)}")

    return next(iter(taxon_ids), None)


def read_fasta_data(
        fasta_path: str | Path, id_format: str = "first_token",
    ) -> FastaData:
    """
    Read FASTA sequences and optional UniProt taxonomy annotations.
    """
    sequences: dict[str, str] = {}
    taxon_ids: dict[str, str] = {}
    current_id: str | None = None
    current_taxon_id: str | None = None
    chunks: list[str] = []

    def save_current_record() -> None:
        """
        Save the current FASTA record if one is active.
        """
        if current_id is None:
            return
        if not chunks:
            raise ValueError(f"FASTA record '{current_id}' has no sequence.")

        sequences[current_id] = "".join(chunks)
        if current_taxon_id is not None:
            taxon_ids[current_id] = current_taxon_id

    with open_text_auto(fasta_path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                save_current_record()
                header = line[1:]
                current_id = fasta_record_id(header, id_format)
                if current_id in sequences:
                    raise ValueError(
                        f"Duplicate FASTA sequence ID: {current_id}")
                current_taxon_id = fasta_record_taxon_id(header)
                chunks = []
            else:
                if current_id is None:
                    raise ValueError(
                        "FASTA sequence line found before any header.")
                chunks.append(line)

    save_current_record()

    return FastaData(sequences=sequences, taxon_ids=taxon_ids)


def read_fasta(
        fasta_path: str | Path, id_format: str = "first_token",
    ) -> dict[str, str]:
    """
    Read a FASTA file into a sequence dictionary.
    """
    return read_fasta_data(fasta_path, id_format=id_format).sequences


def read_protein_taxa(metadata_path: str | Path) -> dict[str, str]:
    """
    Read normalized protein-to-taxon assignments from a CSV table.
    """
    metadata = pd.read_csv(metadata_path, dtype="string")
    missing_columns = [
        column for column in PROTEIN_METADATA_COLUMNS
        if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            "Protein metadata is missing required columns: "
            f"{missing_columns}")

    protein_ids = metadata["protein_id"]
    missing_protein_mask = protein_ids.isna() | protein_ids.str.strip().eq("")
    if missing_protein_mask.any():
        raise ValueError(
            "Protein metadata contains rows with missing protein_id values.")

    protein_ids = protein_ids.str.strip()
    duplicate_mask = protein_ids.duplicated(keep=False)
    if duplicate_mask.any():
        examples = sorted(set(protein_ids.loc[duplicate_mask]))[:10]
        raise ValueError(
            "Protein metadata must contain one row per protein_id. Duplicate "
            f"examples: {examples}")

    protein_taxa = {}
    for protein_id, taxon_value in zip(protein_ids, metadata["taxon_id"]):
        if pd.isna(taxon_value) or not str(taxon_value).strip():
            continue
        protein_taxa[str(protein_id)] = normalize_taxon_id(
            taxon_value,
            context=f"Taxon ID for protein '{protein_id}'",
        )

    return protein_taxa


def merge_protein_taxa(
        base_taxa: dict[str, str], additional_taxa: dict[str, str],
        source_name: str,
    ) -> dict[str, str]:
    """
    Merge protein taxa while rejecting conflicting assignments.
    """
    conflicts = [
        protein_id
        for protein_id, taxon_id in additional_taxa.items()
        if protein_id in base_taxa and base_taxa[protein_id] != taxon_id
    ]
    if conflicts:
        examples = [
            {
                "protein_id": protein_id,
                "existing_taxon_id": base_taxa[protein_id],
                "new_taxon_id": additional_taxa[protein_id],
            }
            for protein_id in sorted(conflicts)[:10]
        ]
        raise ValueError(
            f"Conflicting protein taxon assignments from {source_name}. "
            f"Examples: {examples}")

    merged = {**base_taxa, **additional_taxa}

    return merged


def resolve_protein_taxa(
        fasta_data: FastaData,
        protein_metadata_path: str | Path | None = None,
        taxon_id: str | int | None = None,
    ) -> dict[str, str]:
    """
    Combine FASTA, sidecar, and optional dataset-wide taxon assignments.
    """
    protein_taxa = dict(fasta_data.taxon_ids)
    if taxon_id is not None:
        normalized_taxon_id = normalize_taxon_id(
            taxon_id,
            context="--taxon-id",
        )
        protein_taxa = merge_protein_taxa(
            base_taxa=protein_taxa,
            additional_taxa={
                protein_id: normalized_taxon_id
                for protein_id in fasta_data.sequences
            },
            source_name="--taxon-id",
        )
    if protein_metadata_path is not None:
        protein_taxa = merge_protein_taxa(
            base_taxa=protein_taxa,
            additional_taxa=read_protein_taxa(protein_metadata_path),
            source_name=str(protein_metadata_path),
        )

    return protein_taxa


def read_fasta_with_taxa(
        fasta_path: str | Path, id_format: str = "first_token",
        protein_metadata_path: str | Path | None = None,
        taxon_id: str | int | None = None,
    ) -> FastaData:
    """
    Read FASTA data and resolve all available protein taxon assignments.
    """
    fasta_data = read_fasta_data(fasta_path, id_format=id_format)

    return FastaData(
        sequences=fasta_data.sequences,
        taxon_ids=resolve_protein_taxa(
            fasta_data=fasta_data,
            protein_metadata_path=protein_metadata_path,
            taxon_id=taxon_id,
        ),
    )


def discover_protein_metadata_path(
        pairs_path: str | Path,
        explicit_path: str | Path | None = None,
    ) -> Path | None:
    """
    Return an explicit sidecar or discover one beside canonical pairs.csv.
    """
    if explicit_path is not None:
        return Path(explicit_path)

    pairs_path = Path(pairs_path)
    candidate = pairs_path.with_name(PROTEIN_METADATA_FILENAME)
    if pairs_path.name == PAIRS_FILENAME and candidate.exists():
        return candidate

    return None


def write_fasta(
        sequences: dict[str, str], protein_ids: Iterable[str],
        fasta_path: str | Path,
    ) -> None:
    """
    Write selected FASTA records in deterministic protein-ID order.
    """
    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    selected_ids = sorted(set(protein_ids))
    missing_ids = [protein_id for protein_id in selected_ids
                   if protein_id not in sequences]
    if missing_ids:
        examples = missing_ids[:10]
        raise ValueError(
            f"Cannot write FASTA: {len(missing_ids)} protein IDs are missing "
            f"from sequences. Examples: {examples}")

    with fasta_path.open("w", encoding="utf-8") as fout:
        for protein_id in selected_ids:
            fout.write(f">{protein_id}\n{sequences[protein_id]}\n")


def write_protein_metadata(
        protein_ids: Iterable[str], protein_taxa: dict[str, str],
        metadata_path: str | Path,
    ) -> None:
    """
    Write one compact protein-to-taxon sidecar in deterministic order.
    """
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame({
        "protein_id": sorted(set(protein_ids)),
    })
    metadata["taxon_id"] = metadata["protein_id"].map(protein_taxa)
    metadata.to_csv(metadata_path, index=False)


def protein_taxon_summary(
        protein_ids: Iterable[str], protein_taxa: dict[str, str],
    ) -> dict[str, Any]:
    """
    Return compact coverage and per-taxon protein counts.
    """
    selected_ids = set(protein_ids)
    counts: dict[str, int] = {}
    for protein_id in selected_ids:
        taxon_id = protein_taxa.get(protein_id)
        if taxon_id is not None:
            counts[taxon_id] = counts.get(taxon_id, 0) + 1

    n_with_taxon = sum(counts.values())
    summary = {
        "n_taxa": len(counts),
        "n_proteins_with_taxon": n_with_taxon,
        "n_proteins_without_taxon": len(selected_ids) - n_with_taxon,
        "protein_counts_by_taxon": dict(sorted(counts.items())),
    }

    return summary


def file_sha256(input_path: str | Path) -> str:
    """
    Return the SHA256 digest for a file.
    """
    digest = hashlib.sha256()
    with Path(input_path).open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def file_size_bytes(input_path: str | Path) -> int:
    """
    Return a file size in bytes.
    """
    return int(Path(input_path).stat().st_size)


def write_json(data: dict[str, Any], output_path: str | Path) -> None:
    """
    Write stable, human-readable JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def infer_separator(table_path: str | Path) -> str:
    """
    Infer a table separator from a supported file extension.
    """
    suffixes = [suffix.lower() for suffix in Path(table_path).suffixes]
    while suffixes and suffixes[-1] in COMPRESSED_TEXT_SUFFIXES:
        suffixes.pop()
    suffix = suffixes[-1] if suffixes else ""
    if suffix not in SUPPORTED_TABLE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_TABLE_SUFFIXES))
        raise ValueError(
            f"Cannot infer separator for '{table_path}'. Supported table "
            f"extensions are: {supported}.")

    return SUPPORTED_TABLE_SUFFIXES[suffix]


def selected_archive_member(
        archive: zipfile.ZipFile, table_path: Path,
        archive_member: str | None,
    ) -> str:
    """
    Resolve one table member from a ZIP archive.
    """
    members = [
        member for member in archive.namelist()
        if not member.endswith("/")
    ]
    if archive_member is None:
        if len(members) != 1:
            raise ValueError(
                f"ZIP archive '{table_path}' contains {len(members)} files. "
                "Pass --archive-member with the table to read. "
                f"Examples: {members[:10]}")
        archive_member = members[0]
    elif archive_member not in members:
        raise ValueError(
            f"ZIP archive member '{archive_member}' was not found in "
            f"'{table_path}'. Examples: {members[:10]}")

    return archive_member


@contextmanager
def table_source(
        table_path: str | Path, archive_member: str | None = None,
    ) -> Iterator[tuple[Any, str]]:
    """
    Yield a readable table source and its inferred separator.
    """
    table_path = Path(table_path)
    if table_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(table_path) as archive:
            member = selected_archive_member(
                archive=archive,
                table_path=table_path,
                archive_member=archive_member,
            )
            with archive.open(member) as fin:
                yield fin, infer_separator(member)
        return

    if archive_member is not None:
        raise ValueError("--archive-member can only be used with a ZIP table.")
    yield table_path, infer_separator(table_path)


def read_table(
        table_path: str | Path, archive_member: str | None = None,
        usecols: Iterable[str] | None = None, nrows: int | None = None,
    ) -> pd.DataFrame:
    """
    Read a CSV/TSV/TXT table using extension-based separator inference.
    """
    with table_source(table_path, archive_member) as (source, sep):
        table = pd.read_csv(
            source,
            sep=sep,
            dtype=str,
            usecols=usecols,
            nrows=nrows,
        )

    return table


def iter_table_chunks(
        table_path: str | Path, chunksize: int,
        archive_member: str | None = None,
        usecols: Iterable[str] | None = None,
    ) -> Iterator[pd.DataFrame]:
    """
    Yield bounded-memory chunks from a plain, compressed, or ZIP table.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1.")

    with table_source(table_path, archive_member) as (source, sep):
        yield from pd.read_csv(
            source,
            sep=sep,
            dtype=str,
            usecols=usecols,
            chunksize=chunksize,
        )


def add_common_loader_args(parser: Any) -> None:
    """
    Add shared dataset-preparation arguments to one loader parser.
    """
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument(
        "--fasta-id-format",
        choices=FASTA_ID_FORMAT_CHOICES,
        default="first_token",
        help="How sequence IDs are parsed from FASTA headers.",
    )
    parser.add_argument(
        "--protein-metadata",
        default=None,
        help=(
            "Optional CSV with protein_id and taxon_id columns. UniProt OX= "
            "header fields are used automatically when present."
        ),
    )
    parser.add_argument(
        "--taxon-id",
        default=None,
        help="Optional NCBI taxonomy ID applied to every FASTA record.",
    )
    parser.add_argument("--out-dir", default="processed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")


def validate_negative_ratio(negative_ratio: float) -> float:
    """
    Validate a finite positive negative-sampling ratio.
    """
    negative_ratio = float(negative_ratio)
    if not math.isfinite(negative_ratio) or negative_ratio <= 0.0:
        raise ValueError("negative_ratio must be finite and greater than 0.")

    return negative_ratio


def target_negative_count(n_positive: int, negative_ratio: float) -> int:
    """
    Return the requested number of sampled negatives.
    """
    negative_ratio = validate_negative_ratio(negative_ratio)
    if n_positive <= 0:
        raise ValueError("Cannot sample negatives without positive pairs.")

    target_n_negatives = round(negative_ratio * n_positive)
    target_n_negatives = max(1, target_n_negatives)

    return int(target_n_negatives)


def unordered_pair_key(protein_a: str, protein_b: str) -> tuple[str, str]:
    """
    Return the canonical unordered key for one protein pair.
    """
    return tuple(sorted((protein_a, protein_b)))


def validate_pair_columns(pairs: pd.DataFrame) -> None:
    """
    Raise if required canonical pair columns are absent.
    """
    missing_columns = sorted(REQUIRED_PAIR_COLUMNS - set(pairs.columns))
    if missing_columns:
        raise ValueError(
            "Pair table is missing required columns: "
            f"{missing_columns}")


def normalize_labels(
        labels: pd.Series, context: str = "Pair table",
    ) -> pd.Series:
    """
    Return labels as integer 0/1 values.
    """
    if labels.isna().any():
        raise ValueError(f"{context} label column contains missing values.")

    try:
        numeric_labels = pd.to_numeric(labels, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} label column must contain only 0/1 values.") from exc

    unexpected_values = sorted(
        value for value in pd.unique(numeric_labels) if value not in {0, 1})
    if unexpected_values:
        raise ValueError(
            f"{context} label column must contain only 0/1 values. "
            f"Found: {unexpected_values}")

    return numeric_labels.astype(int)


def prepare_pair_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Return normalized protein and label columns before pair canonicalization.
    """
    validate_pair_columns(pairs)
    work = pairs[["protein_a", "protein_b", "label"]].copy()
    missing_id_mask = (
        work["protein_a"].isna()
        | work["protein_b"].isna()
    )
    work["protein_a"] = work["protein_a"].astype(str).str.strip()
    work["protein_b"] = work["protein_b"].astype(str).str.strip()
    missing_id_mask = (
        missing_id_mask
        | work["protein_a"].eq("")
        | work["protein_b"].eq("")
    )
    if missing_id_mask.any():
        raise ValueError(
            f"Pair table contains {int(missing_id_mask.sum())} rows with "
            "missing protein IDs.")

    work["label"] = normalize_labels(work["label"])

    return work


def require_mapping_columns(
        mapping_table: pd.DataFrame, map_from_col: str, map_to_col: str,
    ) -> None:
    """
    Raise if a user-provided ID mapping table lacks requested columns.
    """
    missing_columns = [
        column
        for column in (map_from_col, map_to_col)
        if column not in mapping_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "ID mapping table is missing required columns: "
            f"{missing_columns}")


def normalize_mapping_table(
        mapping_table: pd.DataFrame, map_from_col: str, map_to_col: str,
    ) -> pd.DataFrame:
    """
    Return non-empty raw-to-canonical ID rows as trimmed strings.
    """
    require_mapping_columns(mapping_table, map_from_col, map_to_col)
    work = mapping_table[[map_from_col, map_to_col]].copy()
    missing_mask = (
        work[map_from_col].isna()
        | work[map_to_col].isna()
    )
    work[map_from_col] = work[map_from_col].astype(str).str.strip()
    work[map_to_col] = work[map_to_col].astype(str).str.strip()
    missing_mask = (
        missing_mask
        | work[map_from_col].eq("")
        | work[map_to_col].eq("")
    )
    work = work.loc[~missing_mask].drop_duplicates().reset_index(drop=True)

    return work


def build_id_mapping(
        mapping_table: pd.DataFrame, map_from_col: str, map_to_col: str,
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Build raw-to-canonical mappings and record one-to-many ambiguities.

    Many raw IDs may point to the same canonical ID. A single raw ID that
    points to multiple canonical IDs is treated as ambiguous and is excluded
    from the usable mapping.
    """
    work = normalize_mapping_table(
        mapping_table=mapping_table,
        map_from_col=map_from_col,
        map_to_col=map_to_col,
    )
    mapping: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for raw_id, group in work.groupby(map_from_col, sort=True):
        canonical_ids = sorted(set(group[map_to_col].tolist()))
        if len(canonical_ids) == 1:
            mapping[str(raw_id)] = canonical_ids[0]
        else:
            ambiguous[str(raw_id)] = canonical_ids

    return mapping, ambiguous


def apply_id_mapping_to_pairs(
        pairs: pd.DataFrame, mapping_table: pd.DataFrame,
        map_from_col: str, map_to_col: str,
    ) -> PairProcessingResult:
    """
    Map pair protein IDs to canonical IDs before pair deduplication.

    This helper is deliberately dataset-agnostic: loaders pass a pair table
    and an explicit user-supplied mapping table. It does not perform online
    lookup, synonym expansion, or biological guessing.
    """
    work = prepare_pair_columns(pairs)
    mapping, ambiguous_mapping = build_id_mapping(
        mapping_table=mapping_table,
        map_from_col=map_from_col,
        map_to_col=map_to_col,
    )
    observed_interactors = sorted(
        set(work["protein_a"]) | set(work["protein_b"]))
    ambiguous_ids = [
        interactor for interactor in observed_interactors
        if interactor in ambiguous_mapping
    ]
    unmapped_ids = [
        interactor for interactor in observed_interactors
        if interactor not in mapping and interactor not in ambiguous_mapping
    ]
    mapped_interactors = [
        interactor for interactor in observed_interactors
        if interactor in mapping
    ]

    ambiguous_set = set(ambiguous_ids)
    unmapped_set = set(unmapped_ids)
    ambiguous_pair_mask = (
        work["protein_a"].isin(ambiguous_set)
        | work["protein_b"].isin(ambiguous_set)
    )
    unmapped_pair_mask = (
        ~ambiguous_pair_mask
        & (
            work["protein_a"].isin(unmapped_set)
            | work["protein_b"].isin(unmapped_set)
        )
    )
    keep_mask = ~(ambiguous_pair_mask | unmapped_pair_mask)
    mapped_pairs = work.loc[keep_mask].copy().reset_index(drop=True)
    if not mapped_pairs.empty:
        mapped_pairs["protein_a"] = mapped_pairs["protein_a"].map(mapping)
        mapped_pairs["protein_b"] = mapped_pairs["protein_b"].map(mapping)

    n_pairs_before = int(len(work))
    n_pairs_dropped_ambiguous = int(ambiguous_pair_mask.sum())
    n_pairs_dropped_unmapped = int(unmapped_pair_mask.sum())
    n_pairs_after = int(len(mapped_pairs))
    metadata = {
        "n_pairs_before_id_mapping": n_pairs_before,
        "n_pairs_after_id_mapping": n_pairs_after,
        "n_pairs_dropped_id_mapping": int(n_pairs_before - n_pairs_after),
        "n_pairs_dropped_unmapped_id": n_pairs_dropped_unmapped,
        "n_pairs_dropped_ambiguous_id": n_pairs_dropped_ambiguous,
        "n_interactors_total": int(len(observed_interactors)),
        "n_interactors_mapped": int(len(mapped_interactors)),
        "n_interactors_unmapped": int(len(unmapped_ids)),
        "n_interactors_ambiguous": int(len(ambiguous_ids)),
        "unmapped_id_examples": unmapped_ids[:10],
        "ambiguous_id_examples": [
            {
                "raw_id": raw_id,
                "canonical_ids": ambiguous_mapping[raw_id],
            }
            for raw_id in ambiguous_ids[:10]
        ],
    }

    return PairProcessingResult(pairs=mapped_pairs, metadata=metadata)


def contradiction_examples(
        deduped_pairs: pd.DataFrame,
    ) -> list[dict[str, Any]]:
    """
    Return contradictory unordered pair examples after same-label deduplication.
    """
    label_counts = deduped_pairs.groupby(
        ["protein_a", "protein_b"],
        sort=True,
    )["label"].nunique()
    contradictory_keys = label_counts[label_counts > 1].index
    examples = [
        {
            "protein_a": protein_a,
            "protein_b": protein_b,
            "labels": [0, 1],
        }
        for protein_a, protein_b in contradictory_keys
    ]

    return examples


def canonicalize_pairs(pairs: pd.DataFrame) -> PairProcessingResult:
    """
    Canonicalize unordered protein pairs and count removed rows.
    """
    work = prepare_pair_columns(pairs)
    self_pair_mask = work["protein_a"] == work["protein_b"]
    n_self_pairs_removed = int(self_pair_mask.sum())
    work = work.loc[~self_pair_mask].copy()

    swap_mask = work["protein_a"] > work["protein_b"]
    work.loc[swap_mask, ["protein_a", "protein_b"]] = work.loc[
        swap_mask,
        ["protein_b", "protein_a"],
    ].to_numpy()

    n_before_deduplication = len(work)
    deduped = work.drop_duplicates(
        subset=["protein_a", "protein_b", "label"],
    ).copy()
    n_duplicate_pairs_removed = int(n_before_deduplication - len(deduped))

    contradictions = contradiction_examples(deduped)
    if contradictions:
        examples = contradictions[:10]
        raise ValueError(
            f"Found {len(contradictions)} unordered protein pairs with "
            f"contradictory labels. Examples: {examples}")

    canonical_pairs = deduped.sort_values(
        ["label", "protein_a", "protein_b"],
    ).reset_index(drop=True)
    metadata = {
        "n_self_pairs_removed": n_self_pairs_removed,
        "n_duplicate_pairs_removed": n_duplicate_pairs_removed,
        "n_contradictory_pairs": 0,
        "n_pairs_after_canonicalization": int(len(canonical_pairs)),
    }

    return PairProcessingResult(pairs=canonical_pairs, metadata=metadata)


def filter_pairs_missing_sequences(
        pairs: pd.DataFrame, sequences: dict[str, str],
    ) -> PairProcessingResult:
    """
    Drop pairs where either protein is absent from the FASTA sequences.
    """
    has_sequence_mask = (
        pairs["protein_a"].isin(sequences)
        & pairs["protein_b"].isin(sequences)
    )
    filtered_pairs = pairs.loc[has_sequence_mask].copy().reset_index(drop=True)
    metadata = {
        "n_pairs_dropped_missing_sequence": int(
            len(pairs) - len(filtered_pairs)),
    }

    return PairProcessingResult(pairs=filtered_pairs, metadata=metadata)


def assign_canonical_pair_ids(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Sort final pairs deterministically and assign fresh canonical pair IDs.
    """
    final_pairs = pairs.sort_values(
        ["label", "protein_a", "protein_b"],
    ).reset_index(drop=True)
    final_pairs.insert(
        0,
        "pair_id",
        [f"pair_{index}" for index in range(len(final_pairs))],
    )
    final_pairs = final_pairs[CANONICAL_PAIR_COLUMNS]

    return final_pairs


def final_pair_counts(pairs: pd.DataFrame) -> dict[str, int]:
    """
    Return final output counts for canonical pairs.
    """
    proteins = set(pairs["protein_a"]) | set(pairs["protein_b"])
    counts = {
        "n_positive_output": int((pairs["label"] == 1).sum()),
        "n_negative_output": int((pairs["label"] == 0).sum()),
        "n_pairs_output": int(len(pairs)),
        "n_unique_proteins_output": int(len(proteins)),
    }

    return counts


def input_protein_count(pairs: pd.DataFrame) -> int:
    """
    Return the number of unique proteins in a raw pair table.
    """
    if pairs.empty:
        return 0

    prepared = prepare_pair_columns(pairs)
    proteins = set(prepared["protein_a"]) | set(prepared["protein_b"])

    return int(len(proteins))


def observed_pair_keys(pairs: pd.DataFrame) -> set[tuple[str, str]]:
    """
    Return non-self unordered pair keys from a pair table.
    """
    prepared = prepare_pair_columns(pairs)
    keys = {
        unordered_pair_key(protein_a, protein_b)
        for protein_a, protein_b in zip(
            prepared["protein_a"],
            prepared["protein_b"],
        )
        if protein_a != protein_b
    }

    return keys


def complement_index_from_rank(
        allowed_rank: int, forbidden_indexes: list[int],
    ) -> int:
    """
    Map a rank in a complement to its full candidate-space index.
    """
    lower = allowed_rank
    upper = allowed_rank + len(forbidden_indexes)
    while lower < upper:
        midpoint = (lower + upper) // 2
        n_allowed_through_midpoint = (
            midpoint + 1 - bisect_right(forbidden_indexes, midpoint)
        )
        if n_allowed_through_midpoint <= allowed_rank:
            lower = midpoint + 1
        else:
            upper = midpoint

    return lower


def sample_complement_indexes(
        n_candidates: int, forbidden_indexes: Iterable[int],
        n_samples: int, rng: random.Random,
    ) -> list[int]:
    """
    Sample indexes without materializing the allowed candidate complement.
    """
    forbidden_indexes = sorted(set(forbidden_indexes))
    n_available = n_candidates - len(forbidden_indexes)
    if n_samples > n_available:
        raise ValueError(
            f"Requested {n_samples} samples from {n_available} candidates.")

    sampled_ranks = rng.sample(range(n_available), n_samples)

    return [
        complement_index_from_rank(rank, forbidden_indexes)
        for rank in sampled_ranks
    ]


class IndexedPairSpace:
    """
    Indexed combinations or Cartesian products of protein IDs.
    """
    def __init__(
            self, protein_a_ids: list[str],
            protein_b_ids: list[str] | None = None,
        ) -> None:
        self.protein_a_ids = protein_a_ids
        self.protein_b_ids = protein_b_ids
        self.protein_a_indexes = {
            protein_id: index
            for index, protein_id in enumerate(protein_a_ids)
        }
        if protein_b_ids is None:
            n_proteins = len(protein_a_ids)
            self.offsets = [
                index * (2 * n_proteins - index - 1) // 2
                for index in range(n_proteins)
            ]
            self.protein_b_indexes = None
            self.n_candidates = n_proteins * (n_proteins - 1) // 2
        else:
            self.offsets = None
            self.protein_b_indexes = {
                protein_id: index
                for index, protein_id in enumerate(protein_b_ids)
            }
            self.n_candidates = len(protein_a_ids) * len(protein_b_ids)

    def pair_index(self, protein_a: str, protein_b: str) -> int:
        """
        Encode one candidate pair as an integer index.
        """
        if self.protein_b_ids is None:
            item_a_index = self.protein_a_indexes[protein_a]
            item_b_index = self.protein_a_indexes[protein_b]
            if item_a_index > item_b_index:
                item_a_index, item_b_index = item_b_index, item_a_index
            return (
                self.offsets[item_a_index]
                + item_b_index
                - item_a_index
                - 1
            )

        if protein_a in self.protein_a_indexes:
            left_id, right_id = protein_a, protein_b
        else:
            left_id, right_id = protein_b, protein_a

        return (
            self.protein_a_indexes[left_id] * len(self.protein_b_ids)
            + self.protein_b_indexes[right_id]
        )

    def pair_from_index(self, pair_index: int) -> tuple[str, str]:
        """
        Decode one integer index as a canonical protein pair.
        """
        if self.protein_b_ids is None:
            item_a_index = bisect_right(self.offsets, pair_index) - 1
            item_b_index = (
                item_a_index + 1 + pair_index - self.offsets[item_a_index]
            )
            return (
                self.protein_a_ids[item_a_index],
                self.protein_a_ids[item_b_index],
            )

        item_a_index, item_b_index = divmod(
            pair_index,
            len(self.protein_b_ids),
        )

        return unordered_pair_key(
            self.protein_a_ids[item_a_index],
            self.protein_b_ids[item_b_index],
        )


def sample_pair_space(
        pair_space: IndexedPairSpace,
        forbidden_pairs: set[tuple[str, str]],
        n_samples: int, rng: random.Random,
    ) -> tuple[list[tuple[str, str]], int]:
    """
    Sample from an indexed pair space while excluding observed pairs.
    """
    forbidden_indexes = [
        pair_space.pair_index(*pair)
        for pair in forbidden_pairs
    ]
    sampled_indexes = sample_complement_indexes(
        n_candidates=pair_space.n_candidates,
        forbidden_indexes=forbidden_indexes,
        n_samples=n_samples,
        rng=rng,
    )

    return (
        [pair_space.pair_from_index(index) for index in sampled_indexes],
        pair_space.n_candidates - len(forbidden_pairs),
    )


def allocate_stratified_samples(
        weights: dict[tuple[str, str], int],
        capacities: dict[tuple[str, str], int],
        target_count: int,
    ) -> dict[tuple[str, str], int]:
    """
    Allocate an exact sample target proportionally across bounded strata.
    """
    allocations = {key: 0 for key in weights}
    remaining = target_count
    while remaining:
        active = [
            key for key in sorted(weights)
            if allocations[key] < capacities[key]
        ]
        if not active:
            raise ValueError(
                f"Cannot allocate {target_count} samples across the available "
                "strata.")

        total_weight = sum(weights[key] for key in active)
        quotas = {
            key: remaining * weights[key] / total_weight
            for key in active
        }
        grants = {
            key: min(
                capacities[key] - allocations[key],
                int(quotas[key]),
            )
            for key in active
        }
        n_granted = sum(grants.values())
        if n_granted == 0:
            ranked_keys = sorted(
                active,
                key=lambda key: (-quotas[key], key),
            )
            for key in ranked_keys[:remaining]:
                grants[key] = 1
            n_granted = sum(grants.values())

        for key, count in grants.items():
            allocations[key] += count
        remaining -= n_granted

    return allocations


def taxon_pair_key(taxon_a: str, taxon_b: str) -> tuple[str, str]:
    """
    Return a canonical unordered taxonomy-pair key.
    """
    return tuple(sorted((taxon_a, taxon_b)))


def taxon_pair_name(taxon_pair: tuple[str, str]) -> str:
    """
    Return a compact JSON key for one taxonomy-pair stratum.
    """
    return "|".join(taxon_pair)


def sample_taxon_stratified_negative_keys(
        protein_ids: list[str], positive_keys: set[tuple[str, str]],
        protein_taxa: dict[str, str], target_count: int,
        rng: random.Random,
    ) -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    """
    Sample within taxonomy-pair strata represented by positive interactions.
    """
    missing_taxa = sorted(
        protein_id for protein_id in protein_ids
        if protein_id not in protein_taxa
    )
    if missing_taxa:
        raise ValueError(
            "Species-aware negative sampling requires a taxon_id for every "
            f"eligible protein. Missing {len(missing_taxa)}; examples: "
            f"{missing_taxa[:10]}")

    proteins_by_taxon: dict[str, list[str]] = {}
    for protein_id in protein_ids:
        proteins_by_taxon.setdefault(
            protein_taxa[protein_id],
            [],
        ).append(protein_id)

    positives_by_stratum: dict[
        tuple[str, str], set[tuple[str, str]]
    ] = {}
    for protein_a, protein_b in positive_keys:
        stratum = taxon_pair_key(
            protein_taxa[protein_a],
            protein_taxa[protein_b],
        )
        positives_by_stratum.setdefault(stratum, set()).add(
            (protein_a, protein_b))

    capacities: dict[tuple[str, str], int] = {}
    weights: dict[tuple[str, str], int] = {}
    for stratum, stratum_positives in positives_by_stratum.items():
        taxon_a, taxon_b = stratum
        if taxon_a == taxon_b:
            n_proteins = len(proteins_by_taxon[taxon_a])
            n_candidates = n_proteins * (n_proteins - 1) // 2
        else:
            n_candidates = (
                len(proteins_by_taxon[taxon_a])
                * len(proteins_by_taxon[taxon_b])
            )
        capacities[stratum] = n_candidates - len(stratum_positives)
        weights[stratum] = len(stratum_positives)

    if sum(capacities.values()) < target_count:
        raise ValueError(
            "Not enough possible negative pairs in the observed taxon-pair "
            f"strata. Requested {target_count}, available "
            f"{sum(capacities.values())}.")
    allocations = allocate_stratified_samples(
        weights=weights,
        capacities=capacities,
        target_count=target_count,
    )

    sampled_keys = []
    for stratum in sorted(positives_by_stratum):
        taxon_a, taxon_b = stratum
        protein_a_ids = proteins_by_taxon[taxon_a]
        if taxon_a == taxon_b:
            protein_b_ids = None
        else:
            protein_b_ids = proteins_by_taxon[taxon_b]
        stratum_samples, _ = sample_pair_space(
            pair_space=IndexedPairSpace(protein_a_ids, protein_b_ids),
            forbidden_pairs=positives_by_stratum[stratum],
            n_samples=allocations[stratum],
            rng=rng,
        )
        sampled_keys.extend(stratum_samples)

    metadata = {
        "positive_pairs_by_taxon_pair": {
            taxon_pair_name(key): weights[key]
            for key in sorted(weights)
        },
        "sampled_negatives_by_taxon_pair": {
            taxon_pair_name(key): allocations[key]
            for key in sorted(allocations)
        },
        "available_negatives_by_taxon_pair": {
            taxon_pair_name(key): capacities[key]
            for key in sorted(capacities)
        },
    }

    return sampled_keys, metadata


def sample_negative_pairs(
        positive_pairs: pd.DataFrame, negative_ratio: float,
        seed: int, allowed_protein_ids: Iterable[str] | None = None,
        protein_taxa: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Sample unobserved pairs globally or within observed taxon-pair strata.
    """
    prepared = prepare_pair_columns(positive_pairs)
    prepared = prepared.loc[prepared["protein_a"] != prepared["protein_b"]]
    if allowed_protein_ids is not None:
        allowed_protein_ids = set(allowed_protein_ids)
        allowed_mask = (
            prepared["protein_a"].isin(allowed_protein_ids)
            & prepared["protein_b"].isin(allowed_protein_ids)
        )
        prepared = prepared.loc[allowed_mask]
    if prepared.empty:
        raise ValueError(
            "Cannot sample negatives without positive pairs whose proteins "
            "have FASTA sequences.")

    protein_ids = sorted(set(prepared["protein_a"]) | set(prepared["protein_b"]))
    positive_keys = observed_pair_keys(prepared)
    target_n_negatives = target_negative_count(
        len(positive_keys),
        negative_ratio,
    )
    n_proteins = len(protein_ids)
    rng = random.Random(seed)
    taxon_metadata: dict[str, Any] = {}
    if protein_taxa:
        sampled_keys, taxon_metadata = sample_taxon_stratified_negative_keys(
            protein_ids=protein_ids,
            positive_keys=positive_keys,
            protein_taxa=protein_taxa,
            target_count=target_n_negatives,
            rng=rng,
        )
        n_available_negatives = sum(
            taxon_metadata["available_negatives_by_taxon_pair"].values()
        )
    else:
        n_possible_pairs = n_proteins * (n_proteins - 1) // 2
        n_available_negatives = n_possible_pairs - len(positive_keys)
        if n_available_negatives < target_n_negatives:
            raise ValueError(
                "Not enough possible negative pairs to satisfy "
                f"negative_ratio={negative_ratio}. Requested "
                f"{target_n_negatives}, available {n_available_negatives}.")
        sampled_keys, n_available_negatives = sample_pair_space(
            pair_space=IndexedPairSpace(protein_ids),
            forbidden_pairs=positive_keys,
            n_samples=target_n_negatives,
            rng=rng,
        )

    sampled_pairs = pd.DataFrame(sampled_keys, columns=["protein_a", "protein_b"])
    sampled_pairs["label"] = 0
    metadata = {
        "target_n_negatives": int(target_n_negatives),
        "n_sampled_negatives": int(len(sampled_pairs)),
        "n_positive_pairs_for_sampling": int(len(positive_keys)),
        "n_proteins_for_sampling": int(n_proteins),
        "n_available_negative_pairs": int(n_available_negatives),
        "species_aware_sampling": bool(protein_taxa),
        **taxon_metadata,
    }

    return sampled_pairs, metadata


def prepare_output_dir(
        out_dir: str | Path, dataset_name: str, overwrite: bool,
    ) -> Path:
    """
    Create or validate a dataset output directory.
    """
    dataset_dir = Path(out_dir) / dataset_name
    if dataset_dir.exists() and not overwrite:
        raise ValueError(
            f"Output directory already exists: {dataset_dir}. Pass "
            "--overwrite to replace canonical outputs.")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    return dataset_dir


def input_path_metadata(input_paths: dict[str, str | Path | None]) -> tuple[
        dict[str, str], dict[str, str], dict[str, int]]:
    """
    Return paths, SHA256 hashes, and sizes keyed by input role.
    """
    paths = {}
    sha256 = {}
    sizes = {}
    for input_name, input_path in input_paths.items():
        if input_path is None:
            continue
        input_path = Path(input_path)
        paths[input_name] = str(input_path)
        sha256[input_name] = file_sha256(input_path)
        sizes[input_name] = file_size_bytes(input_path)

    return paths, sha256, sizes


def final_output_metadata(
        dataset_dir: Path,
    ) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    """
    Return canonical output paths plus path metadata.
    """
    pairs_path = dataset_dir / PAIRS_FILENAME
    fasta_path = dataset_dir / FASTA_FILENAME
    protein_metadata_path = dataset_dir / PROTEIN_METADATA_FILENAME
    metadata_path = dataset_dir / METADATA_FILENAME
    path_metadata = {
        "output_pairs_path": str(pairs_path),
        "output_fasta_path": str(fasta_path),
        "output_protein_metadata_path": str(protein_metadata_path),
        "output_metadata_path": str(metadata_path),
    }

    return (
        pairs_path,
        fasta_path,
        protein_metadata_path,
        metadata_path,
        path_metadata,
    )


def canonicalize_filter_and_assign(
        raw_pairs: pd.DataFrame, sequences: dict[str, str],
    ) -> PairProcessingResult:
    """
    Run pair canonicalization, FASTA filtering, and final pair ID assignment.
    """
    canonicalized = canonicalize_pairs(raw_pairs)
    filtered = filter_pairs_missing_sequences(canonicalized.pairs, sequences)
    if filtered.pairs.empty:
        raise ValueError("No pairs remain after missing-sequence filtering.")

    final_pairs = assign_canonical_pair_ids(filtered.pairs)
    metadata = {
        **canonicalized.metadata,
        **filtered.metadata,
        **final_pair_counts(final_pairs),
    }

    return PairProcessingResult(pairs=final_pairs, metadata=metadata)


def write_prepared_dataset(
        args: Any, loader_name: str, raw_pairs: pd.DataFrame,
        sequences: dict[str, str], input_paths: dict[str, str | Path | None],
        loader_metadata: dict[str, Any],
        loader_specific_options: dict[str, Any],
        protein_taxa: dict[str, str] | None = None,
    ) -> dict[str, Any]:
    """
    Write canonical dataset outputs and return dataset metadata.
    """
    processed = canonicalize_filter_and_assign(raw_pairs, sequences)
    observed_labels = set(processed.pairs["label"])
    if observed_labels != {0, 1}:
        raise ValueError(
            "Prepared dataset must contain both labels 0 and 1 after "
            f"sequence filtering. Found: {sorted(observed_labels)}")

    dataset_dir = prepare_output_dir(
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
    )
    (
        pairs_path,
        fasta_path,
        protein_metadata_path,
        metadata_path,
        output_path_metadata,
    ) = final_output_metadata(dataset_dir)
    proteins = (
        set(processed.pairs["protein_a"])
        | set(processed.pairs["protein_b"])
    )
    protein_taxa = protein_taxa or {}

    processed.pairs.to_csv(pairs_path, index=False)
    write_fasta(sequences, proteins, fasta_path)
    write_protein_metadata(proteins, protein_taxa, protein_metadata_path)
    input_paths_json, input_sha256, input_sizes = input_path_metadata(
        input_paths)
    metadata = {
        "dataset_name": args.dataset_name,
        "loader_name": loader_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": list(sys.argv),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "input_paths": input_paths_json,
        "input_file_sha256": input_sha256,
        "input_file_size_bytes": input_sizes,
        **output_path_metadata,
        "seed": int(args.seed),
        **loader_metadata,
        **processed.metadata,
        "species": protein_taxon_summary(proteins, protein_taxa),
        "loader_specific_options": loader_specific_options,
    }
    write_json(metadata, metadata_path)

    return metadata
