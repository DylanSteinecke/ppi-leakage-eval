"""Task-independent construction and validation of protein grouping artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..reporting.io import output_lock


SEQUENCE_CLUSTER_PROTEIN_COLUMN = "protein_id"
SEQUENCE_CLUSTER_COLUMN = "cluster_id"
SEQUENCE_CLUSTER_METHODS = ("mmseqs2",)
SEQUENCE_CLUSTER_CACHE_SCHEMA_VERSION = 1
SEQUENCE_CLUSTER_PARSER_VERSION = 1
SEQUENCE_CLUSTER_CACHE_IMPLEMENTATION_REVISION = 1
SEQUENCE_CLUSTER_PARSER_IMPLEMENTATION_REVISION = 1
SEQUENCE_CLUSTER_ASSIGNMENTS_FILENAME = "sequence_clusters.csv"
SEQUENCE_CLUSTER_METADATA_FILENAME = "sequence_clusters_metadata.json"
MMSEQS_WORKFLOW = "easy-cluster"
MMSEQS_OUTPUT_SUFFIX = "_cluster.tsv"
MMSEQS_DIAGNOSTIC_TAIL_CHARACTERS = 10_000
MMSEQS_SCIENTIFIC_WARNING = (
    "Partitions built from this artifact are MMseqs2 "
    "sequence-cluster-disjoint. Cluster separation does not prove that every "
    "cross-partition protein pair falls below the configured identity "
    "threshold; that requires a separate all-vs-all audit."
)


@dataclass(frozen=True)
class SequenceClusterParameters:
    """Validated parameters for one automatic sequence-clustering artifact."""

    method: str = "mmseqs2"
    min_seq_id: float = 0.30
    coverage: float = 0.80
    cov_mode: int = 0
    evalue: float = 0.001
    sensitivity: float | None = None
    cluster_mode: int | None = None
    threads: int = 1

    def __post_init__(self) -> None:
        if self.method not in SEQUENCE_CLUSTER_METHODS:
            raise ValueError(
                f"Unknown sequence-cluster method {self.method!r}. "
                f"Expected one of: {', '.join(SEQUENCE_CLUSTER_METHODS)}."
            )
        for field_name in ("min_seq_id", "coverage"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"sequence_cluster_{field_name} must be between 0 and 1."
                )
        if self.cov_mode not in range(6):
            raise ValueError("sequence_cluster_cov_mode must be between 0 and 5.")
        if not math.isfinite(self.evalue) or self.evalue <= 0.0:
            raise ValueError("sequence_cluster_evalue must be greater than 0.")
        if self.sensitivity is not None and (
            not math.isfinite(self.sensitivity) or self.sensitivity <= 0.0
        ):
            raise ValueError(
                "sequence_cluster_sensitivity must be greater than 0."
            )
        if self.cluster_mode is not None and self.cluster_mode not in range(4):
            raise ValueError(
                "sequence_cluster_cluster_mode must be between 0 and 3."
            )
        if self.threads < 1:
            raise ValueError("sequence_cluster_threads must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible effective parameters."""
        return {
            "method": self.method,
            "workflow": MMSEQS_WORKFLOW,
            "min_seq_id": float(self.min_seq_id),
            "coverage": float(self.coverage),
            "cov_mode": int(self.cov_mode),
            "evalue": float(self.evalue),
            "sensitivity": self.sensitivity,
            "sensitivity_mode": (
                "automatic" if self.sensitivity is None else "explicit"
            ),
            "cluster_mode": self.cluster_mode,
            "cluster_mode_setting": (
                "automatic" if self.cluster_mode is None else "explicit"
            ),
            "threads": int(self.threads),
        }


@dataclass(frozen=True)
class SequenceClusterResult:
    """One normalized protein grouping and its complete provenance."""

    protein_to_group: dict[str, str]
    assignments: pd.DataFrame
    metadata: dict[str, Any]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_sequence_cluster_cache_dir() -> Path:
    """Return the shared cache used across benchmark run directories."""
    explicit_path = os.environ.get("PPI_SEQUENCE_CLUSTER_CACHE_DIR")
    if explicit_path:
        return Path(explicit_path).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = (
        Path(xdg_cache_home).expanduser()
        if xdg_cache_home
        else Path.home() / ".cache"
    )
    return cache_root / "ppi-leakage" / "sequence_clusters"


def load_sequence_cluster_mapping(
    mapping_path: str | Path,
    required_proteins: set[str] | None = None,
    *,
    reject_unknown_proteins: bool = False,
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
        mapped_proteins = set(protein_to_group)
        missing_proteins = set(required_proteins) - mapped_proteins
        if missing_proteins:
            examples = sorted(missing_proteins)[:10]
            raise ValueError(
                "Sequence-cluster CSV must map every eligible protein for "
                "grouped splitting. Missing "
                f"{len(missing_proteins)} protein(s): {examples}"
            )
        unknown_proteins = mapped_proteins - set(required_proteins)
        if reject_unknown_proteins and unknown_proteins:
            examples = sorted(unknown_proteins)[:10]
            raise ValueError(
                "Sequence-cluster CSV contains proteins outside the eligible "
                f"universe. Found {len(unknown_proteins)} protein(s): {examples}"
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


def _normalized_sequence_universe(
    sequences: Mapping[str, str],
    eligible_protein_ids: Iterable[str],
) -> tuple[dict[str, str], str]:
    eligible_values = [str(protein_id) for protein_id in eligible_protein_ids]
    if len(eligible_values) != len(set(eligible_values)):
        raise ValueError("Eligible protein IDs must not contain duplicates.")
    if not eligible_values:
        raise ValueError("Sequence clustering requires at least one protein.")

    normalized: dict[str, str] = {}
    for protein_id in sorted(eligible_values):
        if not protein_id or protein_id != protein_id.strip():
            raise ValueError(
                "Eligible protein IDs must be nonblank and already normalized."
            )
        if any(character.isspace() for character in protein_id):
            raise ValueError(
                f"Eligible protein ID {protein_id!r} contains whitespace."
            )
        if protein_id not in sequences:
            raise ValueError(
                f"Eligible protein {protein_id!r} is missing a sequence."
            )
        sequence = str(sequences[protein_id]).strip().upper()
        if not sequence:
            raise ValueError(f"Protein {protein_id!r} has a blank sequence.")
        if any(character.isspace() for character in sequence):
            raise ValueError(
                f"Protein {protein_id!r} contains whitespace in its sequence."
            )
        normalized[protein_id] = sequence

    canonical_universe = json.dumps(
        list(normalized.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    universe_hash = hashlib.sha256(
        canonical_universe.encode("utf-8")
    ).hexdigest()
    return normalized, universe_hash


def _cluster_size_statistics(assignments: pd.DataFrame) -> dict[str, Any]:
    sizes = assignments.groupby(SEQUENCE_CLUSTER_COLUMN).size()
    n_proteins = int(len(assignments))
    return {
        "minimum_cluster_size": int(sizes.min()),
        "median_cluster_size": float(sizes.median()),
        "mean_cluster_size": float(sizes.mean()),
        "maximum_cluster_size": int(sizes.max()),
        "largest_cluster_fraction": float(sizes.max() / n_proteins),
    }


def _mapping_sha256(assignments: pd.DataFrame) -> str:
    canonical_mapping = json.dumps(
        assignments.sort_values(SEQUENCE_CLUSTER_PROTEIN_COLUMN).to_dict(
            "records"
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_mapping.encode("utf-8")).hexdigest()


def _diagnostic_tail(value: str) -> dict[str, Any]:
    return {
        "tail": value[-MMSEQS_DIAGNOSTIC_TAIL_CHARACTERS:],
        "truncated": len(value) > MMSEQS_DIAGNOSTIC_TAIL_CHARACTERS,
    }


def _stable_cluster_id(member_ids: Sequence[str]) -> str:
    canonical_members = json.dumps(
        sorted(member_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_members.encode("utf-8")).hexdigest()
    return f"sequence_cluster_{digest}"


def parse_mmseqs_cluster_output(
    output_path: str | Path,
    eligible_protein_ids: Iterable[str],
) -> pd.DataFrame:
    """Parse and strictly validate MMseqs2 representative/member output."""
    output_path = Path(output_path)
    eligible_values = [str(protein_id) for protein_id in eligible_protein_ids]
    eligible = set(eligible_values)
    if len(eligible_values) != len(eligible):
        raise ValueError("Eligible protein IDs must not contain duplicates.")
    representative_members: dict[str, list[str]] = {}
    seen_members: set[str] = set()

    with output_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.rstrip("\r\n")
            fields = line.split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise ValueError(
                    "Malformed MMseqs2 cluster output at line "
                    f"{line_number}: expected representative and member."
                )
            representative, member = fields
            unknown = {representative, member} - eligible
            if unknown:
                raise ValueError(
                    "MMseqs2 cluster output contains unknown protein IDs: "
                    f"{sorted(unknown)}"
                )
            if member in seen_members:
                raise ValueError(
                    f"MMseqs2 assigned protein {member!r} more than once."
                )
            seen_members.add(member)
            representative_members.setdefault(representative, []).append(member)

    missing = eligible - seen_members
    if missing:
        raise ValueError(
            "MMseqs2 cluster output did not assign every eligible protein. "
            f"Missing {len(missing)} protein(s): {sorted(missing)[:10]}"
        )
    representatives_missing_themselves = sorted(
        representative
        for representative, members in representative_members.items()
        if representative not in members
    )
    if representatives_missing_themselves:
        raise ValueError(
            "Malformed MMseqs2 cluster output: every representative must be "
            "emitted as a member of its own cluster. Missing examples: "
            f"{representatives_missing_themselves[:10]}"
        )
    rows = []
    for members in representative_members.values():
        cluster_id = _stable_cluster_id(members)
        rows.extend(
            {
                SEQUENCE_CLUSTER_PROTEIN_COLUMN: protein_id,
                SEQUENCE_CLUSTER_COLUMN: cluster_id,
            }
            for protein_id in members
        )
    return pd.DataFrame(rows).sort_values(
        SEQUENCE_CLUSTER_PROTEIN_COLUMN
    ).reset_index(drop=True)


def mmseqs_version(executable: str | Path) -> str:
    """Return the exact MMseqs2 version string or raise actionably."""
    command = [str(executable), "version"]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        detail = stderr.strip()
        suffix = f" MMseqs2 stderr: {detail}" if detail else ""
        raise RuntimeError(
            f"Could not determine the MMseqs2 version using "
            f"{shlex.join(command)}.{suffix}"
        ) from exc
    output = completed.stdout.strip() or completed.stderr.strip()
    if not output:
        raise RuntimeError("MMseqs2 returned an empty version string.")
    return output.splitlines()[0].strip()


def build_mmseqs_command(
    executable: str | Path,
    input_fasta: str | Path,
    output_prefix: str | Path,
    temporary_directory: str | Path,
    parameters: SequenceClusterParameters,
) -> list[str]:
    """Build the exact MMseqs2 easy-cluster argument vector."""
    command = [
        str(executable),
        MMSEQS_WORKFLOW,
        str(input_fasta),
        str(output_prefix),
        str(temporary_directory),
        "--min-seq-id",
        str(parameters.min_seq_id),
        "-c",
        str(parameters.coverage),
        "--cov-mode",
        str(parameters.cov_mode),
        "-e",
        str(parameters.evalue),
        "--threads",
        str(parameters.threads),
    ]
    if parameters.sensitivity is not None:
        command.extend(["-s", str(parameters.sensitivity)])
    if parameters.cluster_mode is not None:
        command.extend(["--cluster-mode", str(parameters.cluster_mode)])
    return command


def _write_canonical_fasta(
    sequences: Mapping[str, str], output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        for protein_id, sequence in sequences.items():
            output_file.write(f">{protein_id}\n{sequence}\n")


def _cache_fingerprint(
    universe_hash: str,
    tool_version: str,
    parameters: SequenceClusterParameters,
) -> str:
    payload = {
        "cache_schema_version": SEQUENCE_CLUSTER_CACHE_SCHEMA_VERSION,
        "parser_version": SEQUENCE_CLUSTER_PARSER_VERSION,
        "cache_implementation_revision": (
            SEQUENCE_CLUSTER_CACHE_IMPLEMENTATION_REVISION
        ),
        "parser_implementation_revision": (
            SEQUENCE_CLUSTER_PARSER_IMPLEMENTATION_REVISION
        ),
        "normalized_sequence_universe_sha256": universe_hash,
        "tool": "mmseqs2",
        "tool_version": tool_version,
        "parameters": parameters.to_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_mmseqs(
    executable: str,
    sequences: Mapping[str, str],
    parameters: SequenceClusterParameters,
    work_dir: Path,
) -> tuple[pd.DataFrame, list[str], str, str]:
    input_fasta = work_dir / "eligible_proteins.fasta"
    output_prefix = work_dir / "mmseqs_clusters"
    temporary_directory = work_dir / "mmseqs_tmp"
    _write_canonical_fasta(sequences, input_fasta)
    command = build_mmseqs_command(
        executable=executable,
        input_fasta=input_fasta,
        output_prefix=output_prefix,
        temporary_directory=temporary_directory,
        parameters=parameters,
    )
    environment = os.environ.copy()
    environment.pop("MMSEQS_NUM_THREADS", None)
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        stdout = getattr(exc, "stdout", "") or ""
        detail = stderr.strip() or stdout.strip() or "no tool output"
        raise RuntimeError(
            "MMseqs2 easy-cluster failed. Command: "
            f"{shlex.join(command)}. Output: {detail}"
        ) from exc
    cluster_output_path = Path(f"{output_prefix}{MMSEQS_OUTPUT_SUFFIX}")
    if not cluster_output_path.is_file():
        raise RuntimeError(
            "MMseqs2 easy-cluster completed without producing "
            f"{cluster_output_path}."
        )
    assignments = parse_mmseqs_cluster_output(
        cluster_output_path,
        sequences,
    )
    return assignments, command, completed.stdout, completed.stderr


def _result_from_cache(
    cache_path: Path,
    fingerprint: str,
    universe_hash: str,
) -> SequenceClusterResult:
    assignments_path = cache_path / SEQUENCE_CLUSTER_ASSIGNMENTS_FILENAME
    metadata_path = cache_path / SEQUENCE_CLUSTER_METADATA_FILENAME
    if not assignments_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"Incomplete sequence-cluster cache entry at {cache_path}. "
            "Remove the entry and rerun clustering."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Corrupt sequence-cluster cache metadata at {metadata_path}. "
            "Remove the entry and rerun clustering."
        ) from exc
    expected_metadata = {
        "cache_schema_version": SEQUENCE_CLUSTER_CACHE_SCHEMA_VERSION,
        "parser_version": SEQUENCE_CLUSTER_PARSER_VERSION,
        "cache_implementation_revision": (
            SEQUENCE_CLUSTER_CACHE_IMPLEMENTATION_REVISION
        ),
        "parser_implementation_revision": (
            SEQUENCE_CLUSTER_PARSER_IMPLEMENTATION_REVISION
        ),
        "cache_fingerprint": fingerprint,
        "normalized_sequence_universe_sha256": universe_hash,
    }
    mismatches = [
        key
        for key, expected_value in expected_metadata.items()
        if metadata.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Sequence-cluster cache metadata mismatch at {cache_path}: "
            f"{mismatches}. Remove the entry and rerun clustering."
        )
    try:
        protein_to_group, assignments, counts = load_sequence_cluster_mapping(
            assignments_path,
            required_proteins=set(metadata["eligible_protein_ids"]),
            reject_unknown_proteins=True,
        )
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            f"Corrupt sequence-cluster assignments at {assignments_path}. "
            "Remove the entry and rerun clustering."
        ) from exc
    observed_hash = _file_sha256(assignments_path)
    if observed_hash != metadata.get("mapping_file_sha256"):
        raise ValueError(
            f"Sequence-cluster mapping hash mismatch at {assignments_path}. "
            "Remove the entry and rerun clustering."
        )
    if _mapping_sha256(assignments) != metadata.get("mapping_sha256"):
        raise ValueError(
            f"Sequence-cluster mapping content mismatch at {assignments_path}. "
            "Remove the entry and rerun clustering."
        )
    if counts["n_sequence_clusters"] != metadata.get("n_sequence_clusters"):
        raise ValueError(
            f"Sequence-cluster count mismatch at {cache_path}. Remove the "
            "entry and rerun clustering."
        )
    run_metadata = dict(metadata)
    # The complete ID list is needed to validate the cache entry, but the
    # universe hash and assignment snapshot are sufficient run provenance.
    # Avoid copying a potentially enormous list into every benchmark run.
    run_metadata.pop("eligible_protein_ids", None)
    run_metadata.update({
        "path": str(assignments_path),
        "file_size_bytes": assignments_path.stat().st_size,
        "file_sha256": observed_hash,
        "cache_hit": True,
    })
    return SequenceClusterResult(
        protein_to_group=protein_to_group,
        assignments=assignments,
        metadata=run_metadata,
    )


def _generate_cached_result(
    sequences: Mapping[str, str],
    universe_hash: str,
    executable: str,
    tool_version: str,
    parameters: SequenceClusterParameters,
    cache_root: Path,
    fingerprint: str,
) -> SequenceClusterResult:
    cache_path = cache_root / fingerprint
    lock_target = cache_root / f"{fingerprint}.cache"
    with output_lock(lock_target):
        if cache_path.exists():
            if not cache_path.is_dir():
                raise ValueError(
                    f"Sequence-cluster cache path is not a directory: "
                    f"{cache_path}"
                )
            return _result_from_cache(cache_path, fingerprint, universe_hash)

        temporary_path = Path(tempfile.mkdtemp(
            dir=cache_root,
            prefix=f".{fingerprint}.",
        ))
        try:
            work_path = temporary_path / "work"
            work_path.mkdir()
            assignments, command, stdout, stderr = _run_mmseqs(
                executable=executable,
                sequences=sequences,
                parameters=parameters,
                work_dir=work_path,
            )
            shutil.rmtree(work_path)
            assignments_path = (
                temporary_path / SEQUENCE_CLUSTER_ASSIGNMENTS_FILENAME
            )
            assignments.to_csv(assignments_path, index=False)
            mapping_hash = _file_sha256(assignments_path)
            normalized_mapping_hash = _mapping_sha256(assignments)
            counts = {
                "n_mapped_proteins": int(len(assignments)),
                "n_sequence_clusters": int(
                    assignments[SEQUENCE_CLUSTER_COLUMN].nunique()
                ),
            }
            metadata = {
                "grouping_kind": "sequence_cluster",
                "grouping_source": "generated",
                "method": "mmseqs2",
                "workflow": MMSEQS_WORKFLOW,
                "tool_version": tool_version,
                "executable": executable,
                "version_command": shlex.join([executable, "version"]),
                "version_command_argv": [executable, "version"],
                "effective_command": shlex.join(command),
                "effective_command_argv": command,
                "diagnostics": {
                    "stdout": _diagnostic_tail(stdout),
                    "stderr": _diagnostic_tail(stderr),
                },
                "parameters": parameters.to_dict(),
                "cache_schema_version": SEQUENCE_CLUSTER_CACHE_SCHEMA_VERSION,
                "parser_version": SEQUENCE_CLUSTER_PARSER_VERSION,
                "cache_implementation_revision": (
                    SEQUENCE_CLUSTER_CACHE_IMPLEMENTATION_REVISION
                ),
                "parser_implementation_revision": (
                    SEQUENCE_CLUSTER_PARSER_IMPLEMENTATION_REVISION
                ),
                "cache_fingerprint": fingerprint,
                "cache_path": str(cache_path),
                "cache_hit": False,
                "normalized_sequence_universe_sha256": universe_hash,
                "mapping_file_sha256": mapping_hash,
                "mapping_sha256": normalized_mapping_hash,
                "eligible_protein_ids": list(sequences),
                "missing_entity_policy": "error",
                "composite_grouping_policy": None,
                "representative_policy": "mmseqs2_easy_cluster",
                "transitivity_policy": "mmseqs2_workflow_defined",
                "scientific_warning": MMSEQS_SCIENTIFIC_WARNING,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **counts,
                **_cluster_size_statistics(assignments),
            }
            metadata_path = temporary_path / SEQUENCE_CLUSTER_METADATA_FILENAME
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)

    result = _result_from_cache(cache_path, fingerprint, universe_hash)
    miss_metadata = dict(result.metadata)
    miss_metadata["cache_hit"] = False
    return SequenceClusterResult(
        protein_to_group=result.protein_to_group,
        assignments=result.assignments,
        metadata=miss_metadata,
    )


def resolve_sequence_clusters(
    sequences: Mapping[str, str],
    eligible_protein_ids: Iterable[str],
    *,
    supplied_mapping_path: str | Path | None = None,
    parameters: SequenceClusterParameters | None = None,
    cache_dir: str | Path | None = None,
) -> SequenceClusterResult | None:
    """Resolve a supplied or generated protein grouping without task semantics."""
    if supplied_mapping_path is not None and parameters is not None:
        raise ValueError(
            "A supplied sequence-cluster mapping and automatic clustering "
            "are mutually exclusive."
        )
    if supplied_mapping_path is None and parameters is None:
        return None

    normalized_sequences, universe_hash = _normalized_sequence_universe(
        sequences,
        eligible_protein_ids,
    )
    required_proteins = set(normalized_sequences)
    if supplied_mapping_path is not None:
        mapping_path = Path(supplied_mapping_path)
        protein_to_group, assignments, counts = load_sequence_cluster_mapping(
            mapping_path,
            required_proteins=required_proteins,
        )
        mapping_hash = _file_sha256(mapping_path)
        metadata = {
            "grouping_kind": "sequence_cluster",
            "grouping_source": "supplied_csv",
            "method": None,
            "workflow": None,
            "tool_version": None,
            "executable": None,
            "version_command": None,
            "version_command_argv": None,
            "effective_command": None,
            "effective_command_argv": None,
            "diagnostics": None,
            "parameters": None,
            "cache_fingerprint": None,
            "cache_path": None,
            "cache_hit": None,
            "normalized_sequence_universe_sha256": universe_hash,
            "mapping_file_sha256": mapping_hash,
            "mapping_sha256": _mapping_sha256(assignments),
            "path": str(mapping_path),
            "file_size_bytes": mapping_path.stat().st_size,
            "file_sha256": mapping_hash,
            "applied_to_split": True,
            "missing_entity_policy": "error",
            "composite_grouping_policy": None,
            "representative_policy": None,
            "transitivity_policy": None,
            "scientific_warning": (
                "Cluster separation reflects the supplied artifact. Its "
                "biological relation is only as strong as the artifact's "
                "external provenance."
            ),
            **counts,
            **_cluster_size_statistics(assignments),
        }
        return SequenceClusterResult(
            protein_to_group=protein_to_group,
            assignments=assignments,
            metadata=metadata,
        )

    assert parameters is not None
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            "MMseqs2 automatic clustering requires the 'mmseqs' executable "
            "on PATH. Install MMseqs2 or use --sequence-clusters PATH."
        )
    executable = str(Path(executable).resolve())
    tool_version = mmseqs_version(executable)
    fingerprint = _cache_fingerprint(
        universe_hash=universe_hash,
        tool_version=tool_version,
        parameters=parameters,
    )
    cache_root = (
        default_sequence_cluster_cache_dir()
        if cache_dir is None
        else Path(cache_dir).expanduser()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    result = _generate_cached_result(
        sequences=normalized_sequences,
        universe_hash=universe_hash,
        executable=executable,
        tool_version=tool_version,
        parameters=parameters,
        cache_root=cache_root,
        fingerprint=fingerprint,
    )
    metadata = dict(result.metadata)
    metadata["applied_to_split"] = True
    return SequenceClusterResult(
        protein_to_group=result.protein_to_group,
        assignments=result.assignments,
        metadata=metadata,
    )
