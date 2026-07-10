"""
Split audit helpers for PPI pipeline runs.

This module owns source-row tracking, split assignment outputs, split metadata,
and append-mode split compatibility checks.
"""

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ppi_dataset_utils import file_sha256
from ppi_inputs import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT, protein_ids_in_pairs
from ppi_results import output_lock
from split_diagnostics import (
    compute_ppi_split_diagnostics,
    label_counts,
    native_value,
)


SOURCE_ROW_INDEX_COLUMN = "source_row_index"
PAIR_ID_COLUMN = "pair_id"
SPLIT_COLUMN = "split"
SPLITS_DIRNAME = "splits"
SPLIT_ASSIGNMENTS_FILENAME = "split_assignments.csv"
DROPPED_PAIRS_FILENAME = "dropped_pairs.csv"
SPLIT_METADATA_FILENAME = "split_metadata.json"
INVOCATIONS_FILENAME = "invocations.jsonl"


#######################
# Source row identity #
#######################
def add_source_row_index(protein_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Add stable source row IDs from the original pairs.csv row order.
    """
    if SOURCE_ROW_INDEX_COLUMN in protein_pairs.columns:
        raise ValueError(
            "pairs.csv already contains source_row_index; this column is "
            "reserved for the pipeline's original row number.")

    indexed_pairs = protein_pairs.copy()
    indexed_pairs.insert(0, SOURCE_ROW_INDEX_COLUMN, range(len(indexed_pairs)))

    return indexed_pairs


#####################
# Split assignments #
#####################
def split_assignment_frame(
        split_df: pd.DataFrame, split_name: str,
    ) -> pd.DataFrame:
    """
    Return minimal source-row assignments for one split.
    """
    columns = [SOURCE_ROW_INDEX_COLUMN]
    if PAIR_ID_COLUMN in split_df.columns:
        columns.append(PAIR_ID_COLUMN)

    assignment_df = split_df[columns].copy()
    assignment_df.insert(1, SPLIT_COLUMN, split_name)
    columns = [SOURCE_ROW_INDEX_COLUMN, SPLIT_COLUMN]
    if PAIR_ID_COLUMN in assignment_df.columns:
        columns.append(PAIR_ID_COLUMN)
    assignment_df = assignment_df[columns]

    return assignment_df


def make_split_assignments(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Return minimal train/validation/test split assignments.
    """
    assignment_frames = [
        split_assignment_frame(train_df, TRAIN_SPLIT),
    ]
    if val_df is not None and not val_df.empty:
        assignment_frames.append(split_assignment_frame(val_df, VAL_SPLIT))
    assignment_frames.append(split_assignment_frame(test_df, TEST_SPLIT))

    assignments_df = pd.concat(assignment_frames, ignore_index=True)
    assignments_df = assignments_df.sort_values(
        SOURCE_ROW_INDEX_COLUMN,
    ).reset_index(drop=True)

    return assignments_df


def run_git_command(args: list[str], working_directory: Path) -> str | None:
    """
    Return stdout for one git command, or None if git metadata is unavailable.
    """
    try:
        completed_process = subprocess.run(
            ["git", *args],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        output = None
    else:
        output = completed_process.stdout.strip()

    return output


def get_git_metadata(working_directory: str | Path) -> dict[str, Any]:
    """
    Return best-effort git commit, branch, and dirty-state metadata.
    """
    working_directory = Path(working_directory)
    git_commit = run_git_command(["rev-parse", "HEAD"], working_directory)
    git_branch = run_git_command(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        working_directory,
    )
    git_status = run_git_command(["status", "--porcelain"], working_directory)
    git_is_dirty = None if git_status is None else bool(git_status)

    git_metadata = {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_is_dirty": git_is_dirty,
    }

    return git_metadata


#######################
# Invocation metadata #
#######################
def resolved_args_dict(args: Any) -> dict[str, Any]:
    """
    Return the post-parse CLI namespace as JSON-friendly resolved settings.
    """
    resolved_args = {
        str(arg_name): native_value(arg_value)
        for arg_name, arg_value in sorted(vars(args).items())
    }

    return resolved_args


def invocation_log_entry(
        args: Any, output_paths: Any, execution_id: str,
    ) -> dict[str, Any]:
    """
    Return one JSONL record describing this CLI invocation.
    """
    working_directory = Path.cwd()
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_id": execution_id,
        "run_dir": str(output_paths.run_dir),
        "append_results": bool(args.append_results),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": list(sys.argv),
        "python_executable": sys.executable,
        "working_directory": str(working_directory),
        **get_git_metadata(working_directory),
        "resolved_args": resolved_args_dict(args),
    }

    return native_value(entry)


def append_invocation_log(entry: dict[str, Any], output_path: Path) -> None:
    """
    Append one invocation record to a run-level JSONL log.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output_path):
        with output_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(entry, sort_keys=True) + "\n")


def split_protein_ids(split_df: pd.DataFrame | None) -> set[str]:
    """
    Return protein IDs in one optional split dataframe.
    """
    if split_df is None or split_df.empty:
        protein_ids = set()
    else:
        protein_ids = protein_ids_in_pairs(split_df)

    return protein_ids


def path_string(output_path: Path | None) -> str | None:
    """
    Return a string path or None for optional output paths.
    """
    path_value = None if output_path is None else str(output_path)

    return path_value


def compute_split_metadata(
        args: Any, output_paths: Any, protein_pairs: pd.DataFrame,
        dropped_pairs: pd.DataFrame, train_df: pd.DataFrame,
        val_df: pd.DataFrame | None, test_df: pd.DataFrame,
        execution_id: str, n_input_pairs_before_filtering: int,
    ) -> dict[str, Any]:
    """
    Return reproducibility and audit metadata for one train/val/test split.
    """
    pairs_path = Path(args.pairs)
    fasta_path = Path(args.fasta)
    n_val = 0 if val_df is None else len(val_df)
    n_split_pairs = len(train_df) + n_val + len(test_df)
    actual_train_size = len(train_df) / n_split_pairs
    actual_val_size = n_val / n_split_pairs
    actual_test_size = len(test_df) / n_split_pairs
    target_test_size = 1.0 - args.train_size - args.val_size

    total_proteins = protein_ids_in_pairs(protein_pairs)
    train_proteins = split_protein_ids(train_df)
    val_proteins = split_protein_ids(val_df)
    test_proteins = split_protein_ids(test_df)
    working_directory = Path.cwd()
    diagnostics = compute_ppi_split_diagnostics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        args=args,
        protein_pairs=protein_pairs,
    )

    metadata = {
        "execution_id": execution_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(output_paths.run_dir),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": sys.argv,
        "resolved_args": resolved_args_dict(args),
        "python_executable": sys.executable,
        "working_directory": str(working_directory),
        **get_git_metadata(working_directory),
        "pairs_path": str(pairs_path),
        "fasta_path": str(fasta_path),
        "pairs_file_size_bytes": pairs_path.stat().st_size,
        "fasta_file_size_bytes": fasta_path.stat().st_size,
        "pairs_file_sha256": file_sha256(pairs_path),
        "fasta_file_sha256": file_sha256(fasta_path),
        "n_input_pairs_before_filtering": n_input_pairs_before_filtering,
        "n_pairs_after_filtering": len(protein_pairs),
        "n_dropped_pairs": len(dropped_pairs),
        "split_strategy": args.effective_split_strategy,
        "split_name": args.split_name,
        "split_col": args.split_col,
        "split_seed": args.seed,
        "target_train_size": args.train_size,
        "target_val_size": args.val_size,
        "target_test_size": target_test_size,
        "eval_test_set": args.evaluate_test_metrics,
        "n_train": len(train_df),
        "n_val": n_val,
        "n_test": len(test_df),
        "actual_train_size": actual_train_size,
        "actual_val_size": actual_val_size,
        "actual_test_size": actual_test_size,
        "label_counts_total_after_filtering": label_counts(protein_pairs),
        "label_counts_train": label_counts(train_df),
        "label_counts_val": label_counts(val_df),
        "label_counts_test": label_counts(test_df),
        "n_unique_proteins_total_after_filtering": len(total_proteins),
        "n_unique_proteins_train": len(train_proteins),
        "n_unique_proteins_val": len(val_proteins),
        "n_unique_proteins_test": len(test_proteins),
        "n_shared_proteins_train_val": len(train_proteins & val_proteins),
        "n_shared_proteins_train_test": len(train_proteins & test_proteins),
        "n_shared_proteins_val_test": len(val_proteins & test_proteins),
        "n_connected_components": args.n_connected_components,
        "n_pruned_pairs": args.n_pruned_pairs,
        "pruned_pair_fraction": args.pruned_pair_fraction,
        "diagnostics": diagnostics,
        "train_metrics_path": path_string(output_paths.train_metrics_path),
        "split_assignments_path": path_string(
            output_paths.split_assignments_path),
        "dropped_pairs_path": path_string(output_paths.dropped_pairs_path),
        "split_metadata_path": path_string(output_paths.split_metadata_path),
        "invocations_path": path_string(output_paths.invocations_path),
    }
    if output_paths.val_metrics_path is not None:
        metadata["val_metrics_path"] = path_string(
            output_paths.val_metrics_path)
    if output_paths.test_metrics_path is not None:
        metadata["test_metrics_path"] = path_string(
            output_paths.test_metrics_path)
    if output_paths.predictions_path is not None:
        metadata["predictions_path"] = path_string(
            output_paths.predictions_path)

    return metadata


####################
# Artifact writing #
####################
def write_metadata_json(metadata: dict[str, Any], output_path: Path) -> None:
    """
    Write metadata as stable, human-readable JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_append_split_assignments(
        new_assignments: pd.DataFrame, existing_path: Path,
    ) -> None:
    """
    Fail if append mode would mix metrics from different splits.
    """
    existing_assignments = pd.read_csv(existing_path)
    try:
        pd.testing.assert_frame_equal(
            existing_assignments,
            new_assignments,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError(
            f"Cannot append results to {existing_path.parent.parent}: "
            "existing split_assignments.csv does not match the split for "
            "this run. The same run-dir is being reused with a different "
            "split. Use a new --run-dir or rerun without --append-results."
        ) from exc


def validate_append_input_files(
        new_metadata: dict[str, Any], existing_path: Path,
    ) -> None:
    """
    Fail if append mode would mix results from different input contents.
    """
    if not existing_path.exists():
        return

    with existing_path.open("r", encoding="utf-8") as fin:
        existing_metadata = json.load(fin)
    hash_fields = ("pairs_file_sha256", "fasta_file_sha256")
    changed_fields = [
        field
        for field in hash_fields
        if existing_metadata.get(field) != new_metadata.get(field)
    ]
    if changed_fields:
        raise ValueError(
            f"Cannot append results to {existing_path.parent.parent}: input "
            f"file contents changed ({changed_fields}). Use a new --run-dir "
            "or rerun without --append-results.")


def write_split_artifacts(
        split_assignments: pd.DataFrame, dropped_pairs: pd.DataFrame,
        split_metadata: dict[str, Any], output_paths: Any,
        append_results: bool,
    ) -> None:
    """
    Write split assignments, dropped rows, and split metadata.
    """
    split_assignments_path = output_paths.split_assignments_path
    dropped_pairs_path = output_paths.dropped_pairs_path
    split_metadata_path = output_paths.split_metadata_path

    if append_results and split_assignments_path.exists():
        validate_append_split_assignments(
            new_assignments=split_assignments,
            existing_path=split_assignments_path,
        )
        validate_append_input_files(
            new_metadata=split_metadata,
            existing_path=split_metadata_path,
        )
        if not dropped_pairs_path.exists():
            dropped_pairs.to_csv(dropped_pairs_path, index=False)
        if not split_metadata_path.exists():
            write_metadata_json(split_metadata, split_metadata_path)
        return

    split_assignments_path.parent.mkdir(parents=True, exist_ok=True)
    split_assignments.to_csv(split_assignments_path, index=False)
    dropped_pairs.to_csv(dropped_pairs_path, index=False)
    write_metadata_json(split_metadata, split_metadata_path)
