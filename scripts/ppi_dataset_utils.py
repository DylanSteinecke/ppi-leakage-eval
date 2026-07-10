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
METADATA_FILENAME = "dataset_metadata.json"
CANONICAL_PAIR_COLUMNS = ["pair_id", "protein_a", "protein_b", "label"]
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


def read_fasta(
        fasta_path: str | Path, id_format: str = "first_token",
    ) -> dict[str, str]:
    """
    Read a FASTA file into a sequence dictionary.
    """
    sequences = {}
    current_id = None
    chunks = []

    def save_current_record() -> None:
        """
        Save the current FASTA record if one is active.
        """
        if current_id is None:
            return
        if not chunks:
            raise ValueError(f"FASTA record '{current_id}' has no sequence.")

        sequences[current_id] = "".join(chunks)

    with open_text_auto(fasta_path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                save_current_record()
                current_id = fasta_record_id(line[1:], id_format)
                if current_id in sequences:
                    raise ValueError(
                        f"Duplicate FASTA sequence ID: {current_id}")
                chunks = []
            else:
                if current_id is None:
                    raise ValueError(
                        "FASTA sequence line found before any header.")
                chunks.append(line)

    save_current_record()

    return sequences


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


def sample_negative_pairs(
        positive_pairs: pd.DataFrame, negative_ratio: float,
        seed: int, allowed_protein_ids: Iterable[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """
    Sample unordered negative pairs from proteins present in positives.
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
    n_possible_pairs = n_proteins * (n_proteins - 1) // 2
    n_available_negatives = n_possible_pairs - len(positive_keys)
    if n_available_negatives < target_n_negatives:
        raise ValueError(
            "Not enough possible negative pairs to satisfy "
            f"negative_ratio={negative_ratio}. Requested "
            f"{target_n_negatives}, available {n_available_negatives}.")

    protein_indexes = {
        protein_id: index
        for index, protein_id in enumerate(protein_ids)
    }
    pair_offsets = [
        index * (2 * n_proteins - index - 1) // 2
        for index in range(n_proteins)
    ]
    positive_indexes = sorted(
        pair_offsets[protein_indexes[protein_a]]
        + protein_indexes[protein_b]
        - protein_indexes[protein_a]
        - 1
        for protein_a, protein_b in positive_keys
    )

    def pair_index_from_allowed_rank(allowed_rank: int) -> int:
        """
        Map a rank in the negative complement to the full pair-index space.
        """
        lower = allowed_rank
        upper = allowed_rank + len(positive_indexes)
        while lower < upper:
            midpoint = (lower + upper) // 2
            n_allowed_through_midpoint = (
                midpoint + 1 - bisect_right(positive_indexes, midpoint)
            )
            if n_allowed_through_midpoint <= allowed_rank:
                lower = midpoint + 1
            else:
                upper = midpoint

        return lower

    def pair_from_index(pair_index: int) -> tuple[str, str]:
        """
        Decode one lexicographic combination index into protein IDs.
        """
        protein_a_index = bisect_right(pair_offsets, pair_index) - 1
        protein_b_index = (
            protein_a_index
            + 1
            + pair_index
            - pair_offsets[protein_a_index]
        )

        return protein_ids[protein_a_index], protein_ids[protein_b_index]

    rng = random.Random(seed)
    sampled_ranks = rng.sample(
        range(n_available_negatives),
        target_n_negatives,
    )
    sampled_keys = [
        pair_from_index(pair_index_from_allowed_rank(allowed_rank))
        for allowed_rank in sampled_ranks
    ]
    sampled_pairs = pd.DataFrame(sampled_keys, columns=["protein_a", "protein_b"])
    sampled_pairs["label"] = 0
    metadata = {
        "target_n_negatives": int(target_n_negatives),
        "n_sampled_negatives": int(len(sampled_pairs)),
        "n_positive_pairs_for_sampling": int(len(positive_keys)),
        "n_proteins_for_sampling": int(n_proteins),
        "n_available_negative_pairs": int(n_available_negatives),
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
    ) -> tuple[Path, Path, Path, dict[str, str]]:
    """
    Return canonical output paths plus path metadata.
    """
    pairs_path = dataset_dir / PAIRS_FILENAME
    fasta_path = dataset_dir / FASTA_FILENAME
    metadata_path = dataset_dir / METADATA_FILENAME
    path_metadata = {
        "output_pairs_path": str(pairs_path),
        "output_fasta_path": str(fasta_path),
        "output_metadata_path": str(metadata_path),
    }

    return pairs_path, fasta_path, metadata_path, path_metadata


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
    pairs_path, fasta_path, metadata_path, output_path_metadata = (
        final_output_metadata(dataset_dir))
    proteins = (
        set(processed.pairs["protein_a"])
        | set(processed.pairs["protein_b"])
    )

    processed.pairs.to_csv(pairs_path, index=False)
    write_fasta(sequences, proteins, fasta_path)
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
        "loader_specific_options": loader_specific_options,
    }
    write_json(metadata, metadata_path)

    return metadata
