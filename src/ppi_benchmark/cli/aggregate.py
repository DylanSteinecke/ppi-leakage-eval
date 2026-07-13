#!/usr/bin/env python3

"""
Aggregate canonical PPI run directories into benchmark-level CSVs.

The aggregator derives benchmark bookkeeping from run outputs instead of asking
the shell runner to maintain a second source of truth. Each run contributes one
manifest row from ``splits/split_metadata.json`` and zero or more summary rows
from the train/val/test metrics summary files that exist.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METADATA_RELATIVE_PATH = Path("splits") / "split_metadata.json"
INVOCATIONS_FILENAME = "invocations.jsonl"
MANIFEST_FILENAME = "benchmark_manifest.csv"
SUMMARY_FILENAME = "benchmark_summary.csv"
SUMMARY_FILES = (
    ("train", "train_metrics_summary.csv"),
    ("val", "val_metrics_summary.csv"),
    ("test", "test_metrics_summary.csv"),
)
MANIFEST_COLUMNS = (
    "run_dir",
    "execution_id",
    "timestamp_utc",
    "split_strategy",
    "split_name",
    "split_col",
    "train_size",
    "val_size",
    "test_size",
    "actual_train_size",
    "actual_val_size",
    "actual_test_size",
    "split_seed",
    "num_reruns",
    "features_run",
    "classifiers_run",
    "pairs",
    "fasta",
    "pairs_file_sha256",
    "fasta_file_sha256",
    "git_commit",
    "git_is_dirty",
    "n_train",
    "n_val",
    "n_test",
    "n_input_pairs_before_filtering",
    "n_pairs_after_filtering",
    "n_pairs_in_sampled_cohort",
    "sampling_seed",
    "sampling_applied",
    "n_dropped_pairs",
    "eval_test_set",
    "has_validation_split",
    "n_discarded_edges",
    "discarded_edge_fraction",
    "n_invocations",
)
SUMMARY_CONTEXT_COLUMNS = (
    "run_dir",
    "execution_id",
    "timestamp_utc",
    "pairs",
    "fasta",
    "pairs_file_sha256",
    "fasta_file_sha256",
    "git_commit",
    "git_is_dirty",
    "features_run",
    "classifiers_run",
    "n_invocations",
    "n_input_pairs_before_filtering",
    "n_pairs_after_filtering",
    "n_pairs_in_sampled_cohort",
    "sampling_seed",
    "sampling_applied",
    "n_dropped_pairs",
)
DIAGNOSTIC_COLUMNS = (
    "n_shared_proteins_train_val",
    "n_shared_proteins_train_test",
    "n_shared_proteins_val_test",
    "n_exact_ordered_pair_overlaps_train_val",
    "n_exact_ordered_pair_overlaps_train_test",
    "n_exact_ordered_pair_overlaps_val_test",
    "n_unordered_pair_overlaps_train_val",
    "n_unordered_pair_overlaps_train_test",
    "n_unordered_pair_overlaps_val_test",
    "n_duplicate_ordered_pairs_train",
    "n_duplicate_ordered_pairs_val",
    "n_duplicate_ordered_pairs_test",
    "n_duplicate_unordered_pairs_train",
    "n_duplicate_unordered_pairs_val",
    "n_duplicate_unordered_pairs_test",
)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate PPI run directories into benchmark_manifest.csv and "
            "benchmark_summary.csv."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Directory containing one or more canonical run directories.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional output path for the run-level manifest CSV.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional output path for the combined metrics summary CSV.",
    )

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """
    Read a JSON object from disk.
    """
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)

    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read a JSONL file, returning an empty list when it is absent.
    """
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def csv_value(value: Any) -> Any:
    """
    Convert nested values to stable CSV cell values.
    """
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""

    return value


def ordered_unique(values: list[Any]) -> list[Any]:
    """
    Return unique non-empty values while preserving first-seen order.
    """
    seen = set()
    unique_values = []
    for value in values:
        if isinstance(value, (dict, list, tuple)):
            key = json.dumps(value, sort_keys=True)
        else:
            key = value
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(value)

    return unique_values


def flatten_invocation_arg_values(
        invocations: list[dict[str, Any]], arg_name: str,
    ) -> list[Any]:
    """
    Return unique values for one resolved arg across invocation records.
    """
    values = []
    for invocation in invocations:
        resolved_args = invocation.get("resolved_args", {})
        arg_value = resolved_args.get(arg_name)
        if isinstance(arg_value, list):
            values.extend(arg_value)
        elif arg_value not in (None, ""):
            values.append(arg_value)

    return ordered_unique(values)


def fallback_arg_values(metadata: dict[str, Any], arg_name: str) -> list[Any]:
    """
    Return one metadata resolved-arg value as a list.
    """
    arg_value = metadata.get("resolved_args", {}).get(arg_name)
    if isinstance(arg_value, list):
        values = arg_value
    elif arg_value in (None, ""):
        values = []
    else:
        values = [arg_value]

    return values


def run_directories(benchmark_dir: Path) -> list[Path]:
    """
    Return child directories that contain split metadata.
    """
    metadata_paths = sorted(benchmark_dir.glob(f"*/{METADATA_RELATIVE_PATH}"))
    directories = [
        metadata_path.parent.parent
        for metadata_path in metadata_paths
    ]

    return directories


def diagnostic_context(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Return selected split diagnostics with a column prefix.
    """
    diagnostics = metadata.get("diagnostics", {})
    context = {
        f"diagnostics_{key}": csv_value(diagnostics.get(key, ""))
        for key in DIAGNOSTIC_COLUMNS
    }

    return context


def manifest_row(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Return one benchmark manifest row for a canonical run directory.
    """
    invocations = read_jsonl(run_dir / INVOCATIONS_FILENAME)
    features_run = flatten_invocation_arg_values(invocations, "features")
    classifiers_run = flatten_invocation_arg_values(invocations, "classifiers")
    if not features_run:
        features_run = fallback_arg_values(metadata, "features")
    if not classifiers_run:
        classifiers_run = fallback_arg_values(metadata, "classifiers")

    resolved_args = metadata.get("resolved_args", {})
    sampling = metadata.get("sampling") or {}
    row = {
        "run_dir": str(run_dir),
        "execution_id": metadata.get("execution_id", ""),
        "timestamp_utc": metadata.get("timestamp_utc", ""),
        "split_strategy": metadata.get("split_strategy", ""),
        "split_name": metadata.get("split_name", ""),
        "split_col": metadata.get("split_col", ""),
        "train_size": metadata.get("target_train_size", ""),
        "val_size": metadata.get("target_val_size", ""),
        "test_size": metadata.get("target_test_size", ""),
        "actual_train_size": metadata.get("actual_train_size", ""),
        "actual_val_size": metadata.get("actual_val_size", ""),
        "actual_test_size": metadata.get("actual_test_size", ""),
        "split_seed": metadata.get("split_seed", ""),
        "num_reruns": resolved_args.get("num_reruns", ""),
        "features_run": csv_value(features_run),
        "classifiers_run": csv_value(classifiers_run),
        "pairs": metadata.get("pairs_path", ""),
        "fasta": metadata.get("fasta_path", ""),
        "pairs_file_sha256": metadata.get("pairs_file_sha256", ""),
        "fasta_file_sha256": metadata.get("fasta_file_sha256", ""),
        "git_commit": metadata.get("git_commit", ""),
        "git_is_dirty": metadata.get("git_is_dirty", ""),
        "n_train": metadata.get("n_train", ""),
        "n_val": metadata.get("n_val", ""),
        "n_test": metadata.get("n_test", ""),
        "n_input_pairs_before_filtering": metadata.get(
            "n_input_pairs_before_filtering", ""),
        "n_pairs_after_filtering": metadata.get("n_pairs_after_filtering", ""),
        "n_pairs_in_sampled_cohort": metadata.get(
            "n_pairs_in_sampled_cohort", ""),
        "sampling_seed": sampling.get("seed", ""),
        "sampling_applied": sampling.get("applied", ""),
        "n_dropped_pairs": metadata.get("n_dropped_pairs", ""),
        "eval_test_set": metadata.get("eval_test_set", ""),
        "has_validation_split": resolved_args.get("has_validation_split", ""),
        "n_discarded_edges": metadata.get("n_discarded_edges", ""),
        "discarded_edge_fraction": metadata.get(
            "discarded_edge_fraction",
            "",
        ),
        "n_invocations": len(invocations),
        **diagnostic_context(metadata),
    }

    return {key: csv_value(value) for key, value in row.items()}


def summary_context(row: dict[str, Any]) -> dict[str, Any]:
    """
    Return run-level context columns for benchmark summary rows.
    """
    context = {
        key: row.get(key, "")
        for key in SUMMARY_CONTEXT_COLUMNS
    }
    context.update({
        key: value
        for key, value in row.items()
        if key.startswith("diagnostics_")
    })

    return context


def run_summary_rows(
        run_dir: Path, row: dict[str, Any],
    ) -> list[pd.DataFrame]:
    """
    Return available train/val/test summary dataframes for one run.
    """
    frames = []
    context = summary_context(row)
    for expected_split, filename in SUMMARY_FILES:
        summary_path = run_dir / filename
        if not summary_path.exists():
            continue

        summary_df = pd.read_csv(summary_path)
        if summary_df.empty:
            continue

        for column_name, column_value in reversed(list(context.items())):
            if column_name not in summary_df.columns:
                summary_df.insert(0, column_name, column_value)
        if "summary_file" not in summary_df.columns:
            summary_df.insert(0, "summary_file", filename)
        if "summary_split" not in summary_df.columns:
            summary_df.insert(0, "summary_split", expected_split)
        frames.append(summary_df)

    return frames


def aggregate_benchmark_results(
        benchmark_dir: Path | str,
        manifest_out: Path | str | None = None,
        summary_out: Path | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write benchmark manifest and combined summary CSVs.
    """
    benchmark_dir = Path(benchmark_dir)
    manifest_out = (
        benchmark_dir / MANIFEST_FILENAME
        if manifest_out is None else Path(manifest_out)
    )
    summary_out = (
        benchmark_dir / SUMMARY_FILENAME
        if summary_out is None else Path(summary_out)
    )

    manifest_rows = []
    summary_frames = []
    for run_dir in run_directories(benchmark_dir):
        metadata = read_json(run_dir / METADATA_RELATIVE_PATH)
        row = manifest_row(run_dir, metadata)
        manifest_rows.append(row)
        summary_frames.extend(run_summary_rows(run_dir, row))

    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.empty:
        manifest_df = pd.DataFrame(columns=MANIFEST_COLUMNS)
    else:
        leading_columns = [
            column
            for column in MANIFEST_COLUMNS
            if column in manifest_df.columns
        ]
        other_columns = [
            column
            for column in manifest_df.columns
            if column not in leading_columns
        ]
        manifest_df = manifest_df[leading_columns + other_columns]

    if summary_frames:
        summary_df = pd.concat(summary_frames, ignore_index=True)
    else:
        summary_df = pd.DataFrame()

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    return manifest_df, summary_df


def main() -> None:
    """
    Run the benchmark aggregator.
    """
    args = parse_args()
    manifest_df, summary_df = aggregate_benchmark_results(
        benchmark_dir=args.benchmark_dir,
        manifest_out=args.manifest_out,
        summary_out=args.summary_out,
    )
    manifest_out = args.manifest_out or (
        Path(args.benchmark_dir) / MANIFEST_FILENAME)
    summary_out = args.summary_out or (
        Path(args.benchmark_dir) / SUMMARY_FILENAME)
    print(f"Wrote {len(manifest_df)} run rows to {manifest_out}")
    print(f"Wrote {len(summary_df)} summary rows to {summary_out}")


if __name__ == "__main__":
    main()
