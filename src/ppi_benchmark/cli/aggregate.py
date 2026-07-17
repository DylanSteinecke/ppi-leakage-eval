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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..reporting.benchmark_plots import plot_benchmark_train_val_f1
from ..reporting.degree import (
    BENCHMARK_DEGREE_CONTROL_SELECTION_FILENAME,
    BENCHMARK_DEGREE_LIFT_FILENAME,
    BENCHMARK_DEGREE_SUMMARY_FILENAME,
    aggregate_degree_diagnostics,
    protocol_lift_table,
)
from ..splitting.protocols import C2_SPLIT_STRATEGY, C3_SPLIT_STRATEGY


METADATA_RELATIVE_PATH = Path("splits") / "split_metadata.json"
INVOCATIONS_FILENAME = "invocations.jsonl"
MANIFEST_FILENAME = "benchmark_manifest.csv"
SUMMARY_FILENAME = "benchmark_summary.csv"
TRAIN_VAL_F1_PLOT_FILENAME = "benchmark_train_val_f1.png"
SUMMARY_FILES = (
    ("train", "train_metrics_summary.csv"),
    ("val", "val_metrics_summary.csv"),
    ("test", "test_metrics_summary.csv"),
)


@dataclass(frozen=True)
class BenchmarkAggregationResult:
    """Primary and additive benchmark aggregation outputs."""

    manifest: pd.DataFrame
    summary: pd.DataFrame
    degree_summary: pd.DataFrame
    degree_lift: pd.DataFrame
    degree_selection: dict[str, Any]

    def __iter__(self):
        """Preserve the historical two-value unpacking contract."""
        yield self.manifest
        yield self.summary


DATASET_IDENTITY_COLUMNS = (
    "pairs",
    "fasta",
    "pairs_file_sha256",
    "fasta_file_sha256",
    "dataset_metadata_path",
    "dataset_metadata_file_sha256",
    "dataset_metadata_pairs_binding",
)
NEGATIVE_CONSTRUCTION_COLUMNS = (
    "negative_label_meaning",
    "negative_sampling_policy",
    "negative_ratio_requested",
    "negative_ratio_realized",
    "negative_sampling_seed",
    "negative_construction_timing",
)
NEGATIVE_CONSTRUCTION_FIELD_MAP = {
    "negative_label_meaning": "label_meaning",
    "negative_sampling_policy": "policy",
    "negative_ratio_requested": "negative_ratio_requested",
    "negative_ratio_realized": "negative_ratio_realized",
    "negative_sampling_seed": "seed",
    "negative_construction_timing": "timing",
}
MANIFEST_COLUMNS = (
    "run_dir",
    "evaluation_schema_version",
    "task",
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
    "model_seed",
    "model_seeds",
    "num_reruns",
    "features_run",
    "classifiers_run",
    *DATASET_IDENTITY_COLUMNS,
    *NEGATIVE_CONSTRUCTION_COLUMNS,
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
    "split_grouping_kind",
    "split_grouping_instance",
    "split_grouping_label",
    "sequence_clusters_path",
    "sequence_clusters_file_sha256",
    "sequence_cluster_grouping_applied",
    "sequence_cluster_grouping_kind",
    "sequence_cluster_grouping_source",
    "sequence_cluster_method",
    "sequence_cluster_workflow",
    "sequence_cluster_tool_version",
    "sequence_cluster_min_seq_id",
    "sequence_cluster_coverage",
    "sequence_cluster_cov_mode",
    "sequence_cluster_evalue",
    "sequence_cluster_sensitivity",
    "sequence_cluster_cluster_mode",
    "sequence_cluster_threads",
    "sequence_cluster_cache_fingerprint",
    "sequence_cluster_cache_hit",
    "n_sequence_clusters",
    "n_dropped_pairs",
    "eval_test_set",
    "has_validation_split",
    "n_discarded_edges",
    "discarded_edge_fraction",
    "n_invocations",
)
SUMMARY_CONTEXT_COLUMNS = (
    "run_dir",
    "evaluation_schema_version",
    "task",
    "execution_id",
    "timestamp_utc",
    "split_strategy",
    "split_name",
    *DATASET_IDENTITY_COLUMNS,
    *NEGATIVE_CONSTRUCTION_COLUMNS,
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
    "split_grouping_kind",
    "split_grouping_instance",
    "split_grouping_label",
    "sequence_clusters_path",
    "sequence_clusters_file_sha256",
    "sequence_cluster_grouping_applied",
    "sequence_cluster_grouping_kind",
    "sequence_cluster_grouping_source",
    "sequence_cluster_method",
    "sequence_cluster_workflow",
    "sequence_cluster_tool_version",
    "sequence_cluster_min_seq_id",
    "sequence_cluster_coverage",
    "sequence_cluster_cov_mode",
    "sequence_cluster_evalue",
    "sequence_cluster_sensitivity",
    "sequence_cluster_cluster_mode",
    "sequence_cluster_threads",
    "sequence_cluster_cache_fingerprint",
    "sequence_cluster_cache_hit",
    "n_sequence_clusters",
    "n_dropped_pairs",
)
DIAGNOSTIC_COLUMNS = (
    "has_shared_proteins_across_splits",
    "has_exact_ordered_pair_overlap_across_splits",
    "has_unordered_pair_overlap_across_splits",
    "has_pair_leakage_across_splits",
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
    parser.add_argument(
        "--degree-summary-out",
        default=None,
        help="Optional output path for benchmark degree-stratified summaries.",
    )
    parser.add_argument(
        "--degree-lift-out",
        default=None,
        help="Optional output path for hash-gated degree-control lift rows.",
    )
    parser.add_argument(
        "--degree-control-selection-out",
        default=None,
        help="Optional path for the immutable degree-control selection lock.",
    )
    parser.add_argument(
        "--train-val-plot-out",
        default=None,
        help="Optional output path for the benchmark train/validation F1 PNG.",
    )
    parser.add_argument(
        "--plot-execution-id-prefix",
        default=None,
        help=(
            "Only include execution IDs with this prefix in the benchmark "
            "train/validation plot. CSV aggregation is unchanged."
        ),
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
    Return descendant run directories that contain split metadata.
    """
    metadata_paths = sorted(benchmark_dir.rglob(str(METADATA_RELATIVE_PATH)))
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


def negative_construction_context(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return curated negative-construction provenance for aggregation."""
    construction = metadata.get("negative_construction") or {}
    if not isinstance(construction, dict):
        construction = {}

    context = {
        key: metadata.get(key, "")
        for key in DATASET_IDENTITY_COLUMNS[4:]
    }
    context.update({
        output_field: construction.get(source_field, "")
        for output_field, source_field in (
            NEGATIVE_CONSTRUCTION_FIELD_MAP.items()
        )
    })
    return context


def split_grouping_context(
    metadata: dict[str, Any],
    sequence_clusters: dict[str, Any],
) -> dict[str, str]:
    """Return a stable grouping identity for cross-run comparison."""
    strategy = metadata.get("split_strategy", "")
    if strategy not in {C2_SPLIT_STRATEGY, C3_SPLIT_STRATEGY}:
        return {
            "split_grouping_kind": "",
            "split_grouping_instance": "not_applicable",
            "split_grouping_label": "",
        }

    split_audit = metadata.get("split_audit") or {}
    grouping_kind = split_audit.get("grouping_kind")
    if not grouping_kind:
        grouping_kind = (
            sequence_clusters.get("grouping_kind")
            if sequence_clusters.get("applied_to_split") is True
            else "protein_identity"
        )

    if grouping_kind == "protein_identity":
        return {
            "split_grouping_kind": grouping_kind,
            "split_grouping_instance": grouping_kind,
            "split_grouping_label": "Protein identity",
        }

    if grouping_kind == "sequence_cluster":
        artifact_id = (
            sequence_clusters.get("cache_fingerprint")
            or sequence_clusters.get("mapping_sha256")
            or metadata.get("sequence_clusters_file_sha256")
            or "unspecified"
        )
        artifact_suffix = (
            "" if artifact_id == "unspecified" else f" [{str(artifact_id)[:8]}]"
        )
        if sequence_clusters.get("method") == "mmseqs2":
            parameters = sequence_clusters.get("parameters") or {}
            label = (
                "MMseqs2 "
                f"id={parameters.get('min_seq_id', '?')} "
                f"cov={parameters.get('coverage', '?')} "
                f"mode={parameters.get('cov_mode', '?')}"
                f"{artifact_suffix}"
            )
        elif sequence_clusters.get("grouping_source") == "supplied_csv":
            label = f"Supplied sequence clusters{artifact_suffix}"
        else:
            label = f"Sequence clusters{artifact_suffix}"
        return {
            "split_grouping_kind": grouping_kind,
            "split_grouping_instance": f"{grouping_kind}:{artifact_id}",
            "split_grouping_label": label,
        }

    grouping_kind = str(grouping_kind)
    return {
        "split_grouping_kind": grouping_kind,
        "split_grouping_instance": grouping_kind,
        "split_grouping_label": grouping_kind.replace("_", " ").title(),
    }


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
    sequence_clusters = metadata.get("sequence_clusters") or {}
    sequence_cluster_parameters = sequence_clusters.get("parameters") or {}
    grouping_context = split_grouping_context(metadata, sequence_clusters)
    row = {
        "run_dir": str(run_dir),
        "evaluation_schema_version": metadata.get(
            "evaluation_schema_version", ""),
        "task": metadata.get("task", "ppi"),
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
        "model_seed": metadata.get("model_seed", ""),
        "model_seeds": csv_value(
            metadata.get(
                "model_seeds",
                resolved_args.get("model_seeds", []),
            )
        ),
        "num_reruns": resolved_args.get("num_reruns", ""),
        "features_run": csv_value(features_run),
        "classifiers_run": csv_value(classifiers_run),
        "pairs": metadata.get("pairs_path", ""),
        "fasta": metadata.get("fasta_path", ""),
        "pairs_file_sha256": metadata.get("pairs_file_sha256", ""),
        "fasta_file_sha256": metadata.get("fasta_file_sha256", ""),
        **negative_construction_context(metadata),
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
        **grouping_context,
        "sequence_clusters_path": metadata.get(
            "sequence_clusters_path", ""),
        "sequence_clusters_file_sha256": metadata.get(
            "sequence_clusters_file_sha256", ""),
        "sequence_cluster_grouping_applied": sequence_clusters.get(
            "applied_to_split", ""),
        "sequence_cluster_grouping_kind": sequence_clusters.get(
            "grouping_kind", ""),
        "sequence_cluster_grouping_source": sequence_clusters.get(
            "grouping_source", ""),
        "sequence_cluster_method": sequence_clusters.get("method", ""),
        "sequence_cluster_workflow": sequence_clusters.get(
            "workflow", ""),
        "sequence_cluster_tool_version": sequence_clusters.get(
            "tool_version", ""),
        "sequence_cluster_min_seq_id": sequence_cluster_parameters.get(
            "min_seq_id", ""),
        "sequence_cluster_coverage": sequence_cluster_parameters.get(
            "coverage", ""),
        "sequence_cluster_cov_mode": sequence_cluster_parameters.get(
            "cov_mode", ""),
        "sequence_cluster_evalue": sequence_cluster_parameters.get(
            "evalue", ""),
        "sequence_cluster_sensitivity": sequence_cluster_parameters.get(
            "sensitivity", ""),
        "sequence_cluster_cluster_mode": sequence_cluster_parameters.get(
            "cluster_mode", ""),
        "sequence_cluster_threads": sequence_cluster_parameters.get(
            "threads", ""),
        "sequence_cluster_cache_fingerprint": sequence_clusters.get(
            "cache_fingerprint", ""),
        "sequence_cluster_cache_hit": sequence_clusters.get(
            "cache_hit", ""),
        "n_sequence_clusters": sequence_clusters.get(
            "n_sequence_clusters", ""),
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
        degree_summary_out: Path | str | None = None,
        degree_lift_out: Path | str | None = None,
        degree_control_selection_out: Path | str | None = None,
    ) -> BenchmarkAggregationResult:
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
    degree_summary_out = (
        benchmark_dir / BENCHMARK_DEGREE_SUMMARY_FILENAME
        if degree_summary_out is None else Path(degree_summary_out)
    )
    degree_lift_out = (
        benchmark_dir / BENCHMARK_DEGREE_LIFT_FILENAME
        if degree_lift_out is None else Path(degree_lift_out)
    )
    degree_control_selection_out = (
        benchmark_dir / BENCHMARK_DEGREE_CONTROL_SELECTION_FILENAME
        if degree_control_selection_out is None
        else Path(degree_control_selection_out)
    )

    manifest_rows = []
    summary_frames = []
    directories = run_directories(benchmark_dir)
    for run_dir in directories:
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
    degree_summary, degree_lift, degree_selection = (
        aggregate_degree_diagnostics(
            directories,
            summary_out=degree_summary_out,
            lift_out=degree_lift_out,
            selection_out=degree_control_selection_out,
        )
    )

    return BenchmarkAggregationResult(
        manifest=manifest_df,
        summary=summary_df,
        degree_summary=degree_summary,
        degree_lift=degree_lift,
        degree_selection=degree_selection,
    )


def main() -> None:
    """
    Run the benchmark aggregator.
    """
    args = parse_args()
    result = aggregate_benchmark_results(
        benchmark_dir=args.benchmark_dir,
        manifest_out=args.manifest_out,
        summary_out=args.summary_out,
        degree_summary_out=args.degree_summary_out,
        degree_lift_out=args.degree_lift_out,
        degree_control_selection_out=args.degree_control_selection_out,
    )
    manifest_df = result.manifest
    summary_df = result.summary
    manifest_out = args.manifest_out or (
        Path(args.benchmark_dir) / MANIFEST_FILENAME)
    summary_out = args.summary_out or (
        Path(args.benchmark_dir) / SUMMARY_FILENAME)
    degree_summary_out = args.degree_summary_out or (
        Path(args.benchmark_dir) / BENCHMARK_DEGREE_SUMMARY_FILENAME)
    degree_lift_out = args.degree_lift_out or (
        Path(args.benchmark_dir) / BENCHMARK_DEGREE_LIFT_FILENAME)
    degree_control_selection_out = args.degree_control_selection_out or (
        Path(args.benchmark_dir) / BENCHMARK_DEGREE_CONTROL_SELECTION_FILENAME
    )
    train_val_plot_out = args.train_val_plot_out or (
        Path(args.benchmark_dir) / TRAIN_VAL_F1_PLOT_FILENAME)
    train_val_plot_out = Path(train_val_plot_out)
    if train_val_plot_out.exists():
        train_val_plot_out.unlink()
    print(f"Wrote {len(manifest_df)} run rows to {manifest_out}")
    print(f"Wrote {len(summary_df)} summary rows to {summary_out}")
    degree_summary = result.degree_summary
    degree_lift = result.degree_lift
    print(
        f"Wrote {len(degree_summary)} degree summary rows to "
        f"{degree_summary_out}"
    )
    selected_controls = sorted({
        context["selected_control"]
        for context in result.degree_selection.get("selections", [])
    } or {
        result.degree_selection["default_on_tie_or_missing"]
    })
    print(
        "Locked validation-selected fitted degree control: "
        + ", ".join(map(str, selected_controls))
        + f" ({degree_control_selection_out})"
    )
    print(f"Wrote {len(degree_lift)} degree lift rows to {degree_lift_out}")
    lift_table = protocol_lift_table(degree_lift)
    if lift_table.empty:
        print("Degree lift table unavailable: no matched real/control rows")
    else:
        print("Protocol-level residual_over_degree_control (AUPRC)")
        print(lift_table.to_string(index=False))
    plot_written = plot_benchmark_train_val_f1(
        summary=summary_df,
        plot_path=train_val_plot_out,
        execution_id_prefix=args.plot_execution_id_prefix,
    )
    if plot_written:
        print(f"Wrote train/validation F1 plot to {train_val_plot_out}")
    else:
        print("Skipped train/validation F1 plot: matching summaries not found")


if __name__ == "__main__":
    main()
