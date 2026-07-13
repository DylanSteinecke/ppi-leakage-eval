#!/usr/bin/env python3

"""
Train and evaluate PPI predictors from the command line.

This script owns CLI parsing and high-level orchestration. Domain-specific
input validation, feature extraction, model evaluation, and result writing live
in focused helper modules so future pipeline variants can reuse them.
"""

import argparse
import logging
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..datasets.common import (
    discover_protein_metadata_path,
    FASTA_ID_FORMAT_CHOICES,
    read_fasta_with_taxa,
)
from ..features import (
    FEATURE_CHOICES,
    build_feature_matrices,
    make_feature_name,
    normalize_feature_types,
)
from ..inputs import (
    PROVIDED_SPLIT_STRATEGY,
    load_or_make_split,
    normalized_split_values,
    prepare_input_data,
    PROTEIN_DISJOINT_SPLIT_STRATEGIES,
    RANDOM_SPLIT_STRATEGY,
    SPLIT_STRATEGY_CHOICES,
    validate_splits,
)
from ..models import (
    BASELINE_CLASSIFIER_CHOICES,
    CLASSIFIER_CHOICES,
    get_metrics,
    get_scores_and_predictions,
    make_classifier,
)
from ..plots import (
    legacy_f1_heatmap_output_paths,
    plot_metrics_summary,
    plot_train_test_f1_heatmap,
    plot_train_test_metrics_summary,
    plot_train_test_metrics_summary_png,
)
from ..results import (
    append_dataframe,
    reset_output_file,
    summarize_metrics,
    write_dataframe_threadsafe,
)
from ..sampling import (
    SAMPLING_DIRNAME,
    SELECTED_EXAMPLES_FILENAME,
    SamplingSpec,
    select_examples,
    write_selection_manifest,
)
from ..splits import (
    add_source_row_index,
    append_invocation_log,
    compute_split_metadata,
    DROPPED_PAIRS_FILENAME,
    invocation_log_entry,
    INVOCATIONS_FILENAME,
    make_split_assignments,
    SOURCE_ROW_INDEX_COLUMN,
    SPLIT_ASSIGNMENTS_FILENAME,
    SPLIT_METADATA_FILENAME,
    SPLITS_DIRNAME,
    write_split_artifacts,
)

FEATURELESS_FEATURE = "none"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOGGER = logging.getLogger(__name__)
PLOTS_DIRNAME = "plots"
PREDICTIONS_FILENAME = "predictions.csv"
TRAIN_METRICS_FILENAME = "train_metrics.csv"
VAL_METRICS_FILENAME = "val_metrics.csv"
TEST_METRICS_FILENAME = "test_metrics.csv"
TRAIN_SUMMARY_FILENAME = "train_metrics_summary.csv"
VAL_SUMMARY_FILENAME = "val_metrics_summary.csv"
TEST_SUMMARY_FILENAME = "test_metrics_summary.csv"
TRAIN_PLOT_FILENAME = "train_metrics_summary.svg"
VAL_PLOT_FILENAME = "val_metrics_summary.svg"
TEST_PLOT_FILENAME = "test_metrics_summary.svg"
TRAIN_VAL_PLOT_FILENAME = "train_val_metrics_summary.svg"
TRAIN_VAL_PNG_FILENAME = "train_val_metrics_summary.png"
TRAIN_VAL_F1_HEATMAP_FILENAME = "train_val_f1_heatmap.png"
TRAIN_TEST_PLOT_FILENAME = "train_test_metrics_summary.svg"
TRAIN_TEST_PNG_FILENAME = "train_test_metrics_summary.png"
TRAIN_TEST_F1_HEATMAP_FILENAME = "train_test_f1_heatmap.png"


#######
# CLI #
#######
def positive_int(value: str) -> int:
    """
    Parse a positive integer argparse value.
    """
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def finite_float(value: str) -> float:
    """
    Parse a finite float argparse value.
    """
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise argparse.ArgumentTypeError("value must be finite")

    return parsed_value


def positive_float(value: str) -> float:
    """
    Parse a positive float argparse value.
    """
    parsed_value = finite_float(value)
    if parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")

    return parsed_value


def unit_interval(value: str) -> float:
    """
    Parse a float value between 0 and 1, inclusive.
    """
    parsed_value = finite_float(value)
    if (parsed_value < 0.0) or (parsed_value > 1.0):
        raise argparse.ArgumentTypeError("value must be between 0 and 1")

    return parsed_value


def proportion(value: str) -> float:
    """
    Parse a float proportion between 0 and 1.
    """
    parsed_value = finite_float(value)
    if (parsed_value <= 0.0) or (parsed_value >= 1.0):
        raise argparse.ArgumentTypeError("value must be between 0 and 1")

    return parsed_value


def nonnegative_proportion(value: str) -> float:
    """
    Parse a float proportion between 0 and 1, inclusive of 0 only.
    """
    parsed_value = finite_float(value)
    if (parsed_value < 0.0) or (parsed_value >= 1.0):
        raise argparse.ArgumentTypeError("value must be at least 0 and less than 1")

    return parsed_value


def non_empty_string(value: str) -> str:
    """
    Parse a non-empty argparse string after trimming whitespace.
    """
    parsed_value = value.strip()
    if not parsed_value:
        raise argparse.ArgumentTypeError("value must not be empty")

    return parsed_value


def argument_parser() -> argparse.Namespace:
    """
    Argument parser for protein-protein interaction prediction.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate protein-protein interaction classifiers over "
            "one or more feature/model configurations."))

    # Input data args
    input_group = parser.add_argument_group("Input data")
    input_group.add_argument(
        "--pairs", required=True,
        help="CSV with columns for protein_a, protein_b, label")
    input_group.add_argument(
        "--fasta", required=True,
        help="FASTA file of protein sequences")
    input_group.add_argument(
        "--fasta-id-format",
        choices=FASTA_ID_FORMAT_CHOICES,
        default="first_token",
        help="How sequence IDs are parsed from FASTA headers.")
    input_group.add_argument(
        "--protein-metadata",
        default=None,
        help=(
            "Optional CSV with protein_id and taxon_id columns. A canonical "
            "sidecar beside pairs.csv is discovered automatically."
        ))
    input_group.add_argument(
        "--taxon-id",
        default=None,
        help="Optional NCBI taxonomy ID applied to every FASTA record.")

    # Whole-cohort sampling args
    sampling_group = parser.add_argument_group("Cohort sampling")
    sampling_size_group = sampling_group.add_mutually_exclusive_group()
    sampling_size_group.add_argument(
        "--max-pairs", type=positive_int, default=None,
        help=(
            "Maximum number of eligible pairs retained before splitting. "
            "Sampling is deterministic and stratified by label."
        ))
    sampling_size_group.add_argument(
        "--sample-fraction", type=proportion, default=None,
        help=(
            "Fraction of eligible pairs retained before splitting. Sampling "
            "is deterministic and stratified by label."
        ))
    sampling_group.add_argument(
        "--sampling-seed", type=int, default=0,
        help="Seed for cohort sampling, independent of the split seed.")

    # Output data args
    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument(
        "--run-dir", required=True,
        help="Directory where canonical metrics, predictions, and plots are "
            "written.")
    output_group.add_argument(
        "--append-results", action="store_true",
        help="Append to existing metric/prediction CSVs in --run-dir instead "
            "of starting fresh.")
    output_group.add_argument(
        "--execution-id", default=None,
        help="Optional identifier stored with each model run row.")
    output_group.add_argument(
        "--log-level", choices=LOG_LEVEL_CHOICES, default="INFO",
        help="Logging verbosity.")

    # Plot args
    plot_group = parser.add_argument_group("Plots")
    plot_group.add_argument(
        "--no-metrics-plots", action="store_true",
        help="Do not create canonical metrics summary plots under "
            "--run-dir/plots.")

    # Feature extraction args
    feature_group = parser.add_argument_group("Features")
    feature_group.add_argument(
        "--features", choices=FEATURE_CHOICES, nargs="+", default=["tfidf"],
        help="One or more feature types to concatenate")
    feature_group.add_argument("--k", type=positive_int, default=3,
                               help="k-mer size")
    feature_group.add_argument("--bm25-k1", type=positive_float, default=1.5)
    feature_group.add_argument("--bm25-b", type=unit_interval, default=0.75)

    # Model args
    model_group = parser.add_argument_group("Models")
    model_group.add_argument(
        "--classifier", "--classifiers", dest="classifiers",
        choices=CLASSIFIER_CHOICES, nargs="+", default=["logistic"],
        help="One or more classifier model types")

    # Training args
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--max-iter", type=positive_int, default=1000)
    training_group.add_argument(
        "--train-size", type=proportion, default=0.80,
        help="Fraction of pairs assigned to the training set")
    training_group.add_argument(
        "--val-size", type=nonnegative_proportion, default=0.0,
        help="Fraction of pairs assigned to the validation set")
    training_group.add_argument(
        "--eval-test-set", action="store_true",
        help="Evaluate on the test split. In validation runs, omit this to "
            "keep test held out.")
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument(
        "--split-col", default=None,
        help="Optional column with train, validation, and test labels")
    training_group.add_argument(
        "--split-name", type=non_empty_string, default=None,
        help="Optional display name for a provided --split-col split")
    training_group.add_argument(
        "--split-strategy", choices=SPLIT_STRATEGY_CHOICES, default=None,
        help="Optional strategy for creating train/validation/test splits")
    training_group.add_argument(
        "--n-split-trials", type=positive_int, default=100,
        help="Candidate assignments evaluated for C1/C2/C3 splitting")
    training_group.add_argument(
        "--num-reruns", type=positive_int, default=1,
        help="Number of times to rerun each classifier with consecutive seeds")

    args = parser.parse_args()
    if args.split_col and args.split_strategy:
        parser.error("--split-col and --split-strategy cannot both be set.")
    if args.split_name and not args.split_col:
        parser.error("--split-name can only be used with --split-col.")
    if args.train_size + args.val_size >= 1.0:
        parser.error("--train-size + --val-size must be less than 1.")

    if args.split_col:
        args.effective_split_strategy = PROVIDED_SPLIT_STRATEGY
    else:
        args.effective_split_strategy = (
            args.split_strategy or RANDOM_SPLIT_STRATEGY)
    if (
            args.effective_split_strategy in PROTEIN_DISJOINT_SPLIT_STRATEGIES
            and args.val_size > 0.0
            ):
        parser.error("C1/C2/C3 split strategies require --val-size 0.")
    args.n_discarded_edges = 0
    args.discarded_edge_fraction = 0.0
    args.split_audit = None
    args.sampling_requested = (
        args.max_pairs is not None or args.sample_fraction is not None)

    try:
        args.features = normalize_feature_types(args.features)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def configure_logging(args: argparse.Namespace) -> None:
    """
    Configure command-line logging.
    """
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )


def sample_ppi_cohort(
        protein_pairs: pd.DataFrame, args: argparse.Namespace,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """
    Select the whole PPI cohort before constructing benchmark splits.

    Labels are always strata. For provided splits, the normalized split value
    is included as a joint stratum so every split/class combination is retained.
    """
    sampling_strata = pd.DataFrame({
        "label": protein_pairs["label"].to_numpy(),
    })
    if args.split_col:
        sampling_strata.insert(
            0,
            "split",
            normalized_split_values(protein_pairs, args.split_col).to_numpy(),
        )
        minimum_per_stratum = 1
    else:
        n_requested_splits = 3 if args.val_size > 0.0 else 2
        minimum_per_stratum = n_requested_splits

    sampling_result = select_examples(
        example_ids=protein_pairs[SOURCE_ROW_INDEX_COLUMN],
        spec=SamplingSpec(
            max_examples=args.max_pairs,
            fraction=args.sample_fraction,
            seed=args.sampling_seed,
            minimum_per_stratum=minimum_per_stratum,
        ),
        strata=sampling_strata,
    )
    selected_pairs = protein_pairs.iloc[
        sampling_result.selected_positions
    ].copy()

    selection_manifest = None
    if args.sampling_requested:
        manifest_columns = [SOURCE_ROW_INDEX_COLUMN]
        if "pair_id" in selected_pairs.columns:
            manifest_columns.append("pair_id")
        selection_manifest = selected_pairs[manifest_columns].copy()
        selection_manifest.insert(
            1,
            "sampling_rank",
            sampling_result.sampling_ranks,
        )
        for column in sampling_strata.columns:
            values = sampling_strata.iloc[
                sampling_result.selected_positions
            ][column].to_numpy()
            selection_manifest[column] = values
        selection_manifest = selection_manifest.sort_values(
            SOURCE_ROW_INDEX_COLUMN,
        ).reset_index(drop=True)

    selected_pairs = selected_pairs.reset_index(drop=True)
    return selected_pairs, selection_manifest, sampling_result.metadata


####################
# Pipeline helpers #
####################
@dataclass(frozen=True)
class OutputPaths:
    """
    Pipeline output paths grouped by artifact type.
    """
    run_dir: Path
    plots_dir: Path
    splits_dir: Path
    sampling_dir: Path
    selected_examples_path: Path
    predictions_path: Path | None
    train_metrics_path: Path
    val_metrics_path: Path | None
    test_metrics_path: Path | None
    split_assignments_path: Path
    dropped_pairs_path: Path
    split_metadata_path: Path
    invocations_path: Path
    train_summary_path: Path
    val_summary_path: Path | None
    test_summary_path: Path | None
    train_plot_path: Path | None
    val_plot_path: Path | None
    test_plot_path: Path | None
    train_val_plot_path: Path | None
    train_val_png_path: Path | None
    train_val_f1_heatmap_path: Path | None
    train_test_plot_path: Path | None
    train_test_png_path: Path | None
    train_test_f1_heatmap_path: Path | None


def make_model_name(feature_name: str, classifier_name: str) -> str:
    """
    Return a stable, machine-readable model configuration name.
    """
    if feature_name == FEATURELESS_FEATURE:
        model_name = classifier_name
    else:
        model_name = f"{feature_name}__{classifier_name}"

    return model_name


def is_baseline_classifier(classifier_name: str) -> bool:
    """
    Return whether a classifier ignores feature matrices.
    """
    is_baseline = classifier_name in BASELINE_CLASSIFIER_CHOICES

    return is_baseline


def feature_metadata(
        feature_name: str, args: argparse.Namespace
    ) -> dict[str, float | int | str]:
    """
    Return feature metadata stored with each result row.
    """
    if feature_name == FEATURELESS_FEATURE:
        feature_metadata = {
            "features": FEATURELESS_FEATURE,
            "k": np.nan,
            "bm25_k1": np.nan,
            "bm25_b": np.nan,
        }
    else:
        feature_metadata = {
            "features": feature_name,
            "k": args.k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
        }

    return feature_metadata


def make_metrics_df(
        metadata: dict[str, Any], metrics: dict[str, float], split_name: str,
    ) -> pd.DataFrame:
    """
    Return one split-specific metrics row.
    """
    metrics_df = pd.DataFrame(
        [
            {
                **metadata,
                "split": split_name,
                **metrics,
            }
        ]
    )

    return metrics_df


def reset_output_files(
        output_paths: list[Path | None], append_results: bool,
    ) -> None:
    """
    Reset all non-empty output paths with one append policy.
    """
    for output_path in filter(None, output_paths):
        reset_output_file(
            output_path,
            append_results=append_results,
        )


def train_and_evaluate_model_run(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, x_train: Any, x_val: Any | None,
        x_test: Any | None, feature_name: str, classifier_name: str,
        run_number: int,
        execution_id: str, args: argparse.Namespace
    ) -> tuple[pd.DataFrame, pd.DataFrame | None,
               pd.DataFrame | None, pd.DataFrame | None]:
    """
    Train and evaluate one model configuration for one run number.
    """
    # Define model metadata
    model_name = make_model_name(feature_name, classifier_name)
    is_baseline = is_baseline_classifier(classifier_name)
    run_seed = args.seed + run_number - 1
    model_seed = np.nan if is_baseline else run_seed
    max_iter = np.nan if is_baseline else args.max_iter
    feature_meta = feature_metadata(feature_name, args)

    # Define training labels
    y_train = train_df["label"].to_numpy()

    # Define model
    model = make_classifier(
        classifier_name=classifier_name,
        max_iter=args.max_iter,
        random_state=run_seed,
    )
    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Define the evaluate metrics tables
    n_val = 0 if val_df is None else len(val_df)
    n_total = len(train_df) + n_val + len(test_df)
    actual_train_size = len(train_df) / n_total
    actual_val_size = n_val / n_total
    actual_test_size = len(test_df) / n_total
    metrics_metadata = {
        "execution_id": execution_id,
        "model_name": model_name,
        "run_number": run_number,
        "model_seed": model_seed,
        "split_seed": args.seed,
        "n_train": len(train_df),
        "n_val": n_val,
        "n_test": len(test_df),
        **feature_meta,
        "classifier": classifier_name,
        "max_iter": max_iter,
        "split_strategy": args.effective_split_strategy,
        "split_name": args.split_name,
        "target_train_size": args.train_size,
        "target_val_size": args.val_size,
        "actual_train_size": actual_train_size,
        "actual_val_size": actual_val_size,
        "actual_test_size": actual_test_size,
        "n_discarded_edges": args.n_discarded_edges,
        "discarded_edge_fraction": args.discarded_edge_fraction,
    }

    def evaluate_split(
            split_df: pd.DataFrame, x_split: Any, split_name: str,
        ) -> tuple[pd.DataFrame, Any, Any]:
        """
        Evaluate one split and return metrics plus raw predictions.
        """
        y_true = split_df["label"].to_numpy()
        y_score, y_pred = get_scores_and_predictions(model, x_split)
        metrics = get_metrics(
            y_true=y_true,
            y_score=y_score,
            y_pred=y_pred,
        )
        metrics_df = make_metrics_df(
            metadata=metrics_metadata,
            metrics=metrics,
            split_name=split_name,
        )

        return metrics_df, y_score, y_pred

    train_metrics_df, _, _ = evaluate_split(train_df, x_train, "train")
    val_metrics_df = None
    if val_df is not None and x_val is not None:
        val_metrics_df, _, _ = evaluate_split(val_df, x_val, "val")

    test_metrics_df = None
    predictions_df = None
    if args.evaluate_test_metrics:
        if x_test is None:
            raise ValueError("Test evaluation requested without test features.")
        test_metrics_df, y_test_score, y_test_pred = evaluate_split(
            test_df,
            x_test,
            "test",
        )

        # Define the predictions table
        predictions_df = test_df[[
            SOURCE_ROW_INDEX_COLUMN,
            "protein_a",
            "protein_b",
            "label",
        ]].copy()
        predictions_df.insert(0, "execution_id", execution_id)
        predictions_df.insert(1, "model_name", model_name)
        predictions_df.insert(2, "run_number", run_number)
        predictions_df.insert(3, "model_seed", model_seed)
        predictions_df.insert(5, "features", feature_meta["features"])
        predictions_df.insert(6, "classifier", classifier_name)
        predictions_df.insert(7, "k", feature_meta["k"])
        predictions_df["pred_score"] = y_test_score
        predictions_df["pred_label"] = y_test_pred

    return train_metrics_df, val_metrics_df, test_metrics_df, predictions_df


def run_model_reruns(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, x_train: Any, x_val: Any | None,
        x_test: Any | None, feature_name: str, classifier_name: str,
        execution_id: str, args: argparse.Namespace,
        output_paths: OutputPaths,
    ) -> None:
    """
    Run one model configuration repeatedly and append each result.
    """
    model_name = make_model_name(feature_name, classifier_name)

    # Re-run the model
    for run_number in range(1, args.num_reruns + 1):
        (
            train_metrics_df,
            val_metrics_df,
            test_metrics_df,
            predictions_df,
        ) = (
            train_and_evaluate_model_run(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                feature_name=feature_name,
                classifier_name=classifier_name,
                run_number=run_number,
                execution_id=execution_id,
                args=args,
            )
        )
        # Save performance and prediction results
        output_dfs = (
            (train_metrics_df, output_paths.train_metrics_path),
            (val_metrics_df, output_paths.val_metrics_path),
            (test_metrics_df, output_paths.test_metrics_path),
            (predictions_df, output_paths.predictions_path),
        )
        for output_df, output_path in output_dfs:
            if output_df is not None and output_path is not None:
                append_dataframe(output_df, output_path)
        LOGGER.info(
            f"Finished model={model_name} "
            f"run={run_number}/{args.num_reruns}"
        )


def prepare_outputs(args: argparse.Namespace) -> OutputPaths:
    """
    Resolve canonical output paths inside --run-dir.
    """
    run_dir = Path(args.run_dir)
    plots_dir = run_dir / PLOTS_DIRNAME
    splits_dir = run_dir / SPLITS_DIRNAME
    sampling_dir = run_dir / SAMPLING_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_predictions_path = run_dir / PREDICTIONS_FILENAME
    train_metrics_path = run_dir / TRAIN_METRICS_FILENAME
    all_val_metrics_path = run_dir / VAL_METRICS_FILENAME
    all_test_metrics_path = run_dir / TEST_METRICS_FILENAME
    split_assignments_path = splits_dir / SPLIT_ASSIGNMENTS_FILENAME
    dropped_pairs_path = splits_dir / DROPPED_PAIRS_FILENAME
    split_metadata_path = splits_dir / SPLIT_METADATA_FILENAME
    selected_examples_path = sampling_dir / SELECTED_EXAMPLES_FILENAME
    invocations_path = run_dir / INVOCATIONS_FILENAME
    train_summary_path = run_dir / TRAIN_SUMMARY_FILENAME
    all_val_summary_path = run_dir / VAL_SUMMARY_FILENAME
    all_test_summary_path = run_dir / TEST_SUMMARY_FILENAME
    all_train_plot_path = plots_dir / TRAIN_PLOT_FILENAME
    all_val_plot_path = plots_dir / VAL_PLOT_FILENAME
    all_test_plot_path = plots_dir / TEST_PLOT_FILENAME
    all_train_val_plot_path = plots_dir / TRAIN_VAL_PLOT_FILENAME
    all_train_val_png_path = plots_dir / TRAIN_VAL_PNG_FILENAME
    all_train_val_f1_heatmap_path = plots_dir / TRAIN_VAL_F1_HEATMAP_FILENAME
    all_train_test_plot_path = plots_dir / TRAIN_TEST_PLOT_FILENAME
    all_train_test_png_path = plots_dir / TRAIN_TEST_PNG_FILENAME
    all_train_test_f1_heatmap_path = (
        plots_dir / TRAIN_TEST_F1_HEATMAP_FILENAME)

    predictions_path = (
        all_predictions_path if args.evaluate_test_metrics else None)
    val_metrics_path = (
        all_val_metrics_path if args.has_validation_split else None)
    test_metrics_path = (
        all_test_metrics_path if args.evaluate_test_metrics else None)
    val_summary_path = (
        all_val_summary_path if args.has_validation_split else None)
    test_summary_path = (
        all_test_summary_path if args.evaluate_test_metrics else None)
    train_plot_path = all_train_plot_path if not args.no_metrics_plots else None
    val_plot_path = None
    test_plot_path = None
    train_val_plot_path = None
    train_val_png_path = None
    train_val_f1_heatmap_path = None
    train_test_plot_path = None
    train_test_png_path = None
    train_test_f1_heatmap_path = None
    if not args.no_metrics_plots:
        if args.has_validation_split:
            val_plot_path = all_val_plot_path
            train_val_plot_path = all_train_val_plot_path
            train_val_png_path = all_train_val_png_path
            train_val_f1_heatmap_path = all_train_val_f1_heatmap_path
        if args.evaluate_test_metrics:
            test_plot_path = all_test_plot_path
            train_test_plot_path = all_train_test_plot_path
            train_test_png_path = all_train_test_png_path
            train_test_f1_heatmap_path = all_train_test_f1_heatmap_path

    output_paths = OutputPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        splits_dir=splits_dir,
        sampling_dir=sampling_dir,
        selected_examples_path=selected_examples_path,
        predictions_path=predictions_path,
        train_metrics_path=train_metrics_path,
        val_metrics_path=val_metrics_path,
        test_metrics_path=test_metrics_path,
        split_assignments_path=split_assignments_path,
        dropped_pairs_path=dropped_pairs_path,
        split_metadata_path=split_metadata_path,
        invocations_path=invocations_path,
        train_summary_path=train_summary_path,
        val_summary_path=val_summary_path,
        test_summary_path=test_summary_path,
        train_plot_path=train_plot_path,
        val_plot_path=val_plot_path,
        test_plot_path=test_plot_path,
        train_val_plot_path=train_val_plot_path,
        train_val_png_path=train_val_png_path,
        train_val_f1_heatmap_path=train_val_f1_heatmap_path,
        train_test_plot_path=train_test_plot_path,
        train_test_png_path=train_test_png_path,
        train_test_f1_heatmap_path=train_test_f1_heatmap_path,
    )

    return output_paths


def initialize_output_files(
        args: argparse.Namespace, output_paths: OutputPaths,
    ) -> None:
    """
    Clear stale outputs after append compatibility has been validated.
    """
    run_dir = output_paths.run_dir
    plots_dir = output_paths.plots_dir
    metric_prediction_paths = [
        run_dir / filename
        for filename in (
            PREDICTIONS_FILENAME,
            TRAIN_METRICS_FILENAME,
            VAL_METRICS_FILENAME,
            TEST_METRICS_FILENAME,
        )
    ]
    summary_paths = [
        run_dir / filename
        for filename in (
            TRAIN_SUMMARY_FILENAME,
            VAL_SUMMARY_FILENAME,
            TEST_SUMMARY_FILENAME,
        )
    ]
    plot_paths = [
        plots_dir / filename
        for filename in (
            TRAIN_PLOT_FILENAME,
            VAL_PLOT_FILENAME,
            TEST_PLOT_FILENAME,
            TRAIN_VAL_PLOT_FILENAME,
            TRAIN_VAL_PNG_FILENAME,
            TRAIN_VAL_F1_HEATMAP_FILENAME,
            TRAIN_TEST_PLOT_FILENAME,
            TRAIN_TEST_PNG_FILENAME,
            TRAIN_TEST_F1_HEATMAP_FILENAME,
        )
    ]
    legacy_heatmap_paths = [
        *legacy_f1_heatmap_output_paths(
            plots_dir / TRAIN_VAL_F1_HEATMAP_FILENAME,
            comparison_split_name="val",
        ).values(),
        *legacy_f1_heatmap_output_paths(
            plots_dir / TRAIN_TEST_F1_HEATMAP_FILENAME,
            comparison_split_name="test",
        ).values(),
    ]

    if not args.append_results:
        reset_output_files(
            metric_prediction_paths
            + summary_paths
            + plot_paths
            + legacy_heatmap_paths
            + [output_paths.invocations_path],
            append_results=False,
        )
        return

    active_derived_paths = [
        output_paths.train_summary_path,
        plots_dir / TRAIN_PLOT_FILENAME,
    ]
    if args.has_validation_split:
        active_derived_paths.extend([
            output_paths.val_summary_path,
            plots_dir / VAL_PLOT_FILENAME,
            plots_dir / TRAIN_VAL_PLOT_FILENAME,
            plots_dir / TRAIN_VAL_PNG_FILENAME,
            plots_dir / TRAIN_VAL_F1_HEATMAP_FILENAME,
        ])
    if args.evaluate_test_metrics:
        active_derived_paths.extend([
            output_paths.test_summary_path,
            plots_dir / TEST_PLOT_FILENAME,
            plots_dir / TRAIN_TEST_PLOT_FILENAME,
            plots_dir / TRAIN_TEST_PNG_FILENAME,
            plots_dir / TRAIN_TEST_F1_HEATMAP_FILENAME,
        ])
    reset_output_files(
        active_derived_paths + legacy_heatmap_paths,
        append_results=False,
    )


def summarize_model_outputs(
        output_paths: OutputPaths,
    ) -> dict[str, pd.DataFrame]:
    """
    Write and return summaries for every evaluated split.
    """
    summary_specs = (
        ("train", output_paths.train_metrics_path,
         output_paths.train_summary_path),
        ("val", output_paths.val_metrics_path, output_paths.val_summary_path),
        ("test", output_paths.test_metrics_path,
         output_paths.test_summary_path),
    )
    summaries = {}
    for split_name, metrics_path, summary_path in summary_specs:
        if metrics_path is None or summary_path is None:
            continue
        summary_df = summarize_metrics(metrics_path)
        write_dataframe_threadsafe(summary_df, summary_path)
        summaries[split_name] = summary_df

    return summaries


def plot_model_outputs(
        output_paths: OutputPaths,
    ) -> dict[str, dict[str, Path]]:
    """
    Write configured split and train/comparison summary plots.
    """
    split_plot_specs = (
        ("train", output_paths.train_summary_path,
         output_paths.train_plot_path),
        ("val", output_paths.val_summary_path, output_paths.val_plot_path),
        ("test", output_paths.test_summary_path, output_paths.test_plot_path),
    )
    for split_name, summary_path, plot_path in split_plot_specs:
        if summary_path is not None and plot_path is not None:
            plot_metrics_summary(
                summary_path=summary_path,
                plot_path=plot_path,
                split_name=split_name,
            )

    comparison_specs = (
        (
            "val",
            output_paths.val_summary_path,
            (
                (plot_train_test_metrics_summary,
                 output_paths.train_val_plot_path),
                (plot_train_test_metrics_summary_png,
                 output_paths.train_val_png_path),
                (plot_train_test_f1_heatmap,
                 output_paths.train_val_f1_heatmap_path),
            ),
        ),
        (
            "test",
            output_paths.test_summary_path,
            (
                (plot_train_test_metrics_summary,
                 output_paths.train_test_plot_path),
                (plot_train_test_metrics_summary_png,
                 output_paths.train_test_png_path),
                (plot_train_test_f1_heatmap,
                 output_paths.train_test_f1_heatmap_path),
            ),
        ),
    )
    heatmap_paths = {}
    for split_name, summary_path, plot_specs in comparison_specs:
        if summary_path is None:
            continue
        for plot_function, plot_path in plot_specs:
            if plot_path is None:
                continue
            result = plot_function(
                train_summary_path=output_paths.train_summary_path,
                test_summary_path=summary_path,
                plot_path=plot_path,
                comparison_split_name=split_name,
            )
            if result:
                heatmap_paths[split_name] = result

    return heatmap_paths


def log_model_outputs(
        summaries: dict[str, pd.DataFrame], output_paths: OutputPaths,
        heatmap_paths: dict[str, dict[str, Path]],
    ) -> None:
    """
    Log metric summaries and the canonical artifacts written for this run.
    """
    summary_specs = (
        (
            "train", "Train", output_paths.train_metrics_path,
            output_paths.train_summary_path, None,
        ),
        (
            "val", "Validation", output_paths.val_metrics_path,
            output_paths.val_summary_path, None,
        ),
        (
            "test", "Test", output_paths.test_metrics_path,
            output_paths.test_summary_path, output_paths.predictions_path,
        ),
    )
    sections = []
    for split_name, display_name, metrics_path, summary_path, pred_path in (
            summary_specs):
        if split_name not in summaries:
            continue
        lines = [
            f"{display_name} metric summary",
            summaries[split_name].to_string(index=False),
        ]
        if pred_path is not None:
            lines.append(f"Saved predictions to: {pred_path}")
        lines.extend([
            f"Saved {display_name.lower()} per-run metrics to: {metrics_path}",
            f"Saved {display_name.lower()} metric summary to: {summary_path}",
        ])
        sections.append("\n".join(lines))

    artifact_specs = (
        (output_paths.train_plot_path, "train metric plot"),
        (output_paths.val_plot_path, "validation metric plot"),
        (output_paths.test_plot_path, "test metric plot"),
        (output_paths.train_val_plot_path, "train/validation metric plot"),
        (output_paths.train_val_png_path, "train/validation metric PNG"),
        (output_paths.train_test_plot_path, "train/test metric plot"),
        (output_paths.train_test_png_path, "train/test metric PNG"),
    )
    artifact_lines = [
        f"Saved {description} to: {path}"
        for path, description in artifact_specs
        if path is not None
    ]
    for split_name, paths in heatmap_paths.items():
        display_name = "validation" if split_name == "val" else split_name
        artifact_lines.append(
            f"Saved train/{display_name} F1 heatmap stack to: "
            + ", ".join(str(path) for path in paths.values())
        )

    log_message = "\n\n".join(sections)
    if artifact_lines:
        log_message = f"{log_message}\n" + "\n".join(artifact_lines)
    LOGGER.info(f"\n{log_message}")


def main() -> None:
    """
    Run the full CLI pipeline.
    """
    args = argument_parser()
    configure_logging(args)

    # Load and process input data
    protein_pairs = pd.read_csv(
        args.pairs,
        dtype={"protein_a": "string", "protein_b": "string"},
    )
    protein_pairs = add_source_row_index(protein_pairs)
    n_input_pairs_before_filtering = len(protein_pairs)
    protein_metadata_path = discover_protein_metadata_path(
        pairs_path=args.pairs,
        explicit_path=args.protein_metadata,
    )
    args.protein_metadata = (
        None if protein_metadata_path is None else str(protein_metadata_path)
    )
    fasta_data = read_fasta_with_taxa(
        fasta_path=args.fasta,
        id_format=args.fasta_id_format,
        protein_metadata_path=protein_metadata_path,
        taxon_id=args.taxon_id,
    )
    sequences = fasta_data.sequences
    eligible_protein_pairs, dropped_pairs = prepare_input_data(
        protein_pairs,
        sequences,
    )

    # Select the complete benchmark cohort before constructing any split.
    (
        protein_pairs,
        selection_manifest,
        sampling_metadata,
    ) = sample_ppi_cohort(eligible_protein_pairs, args)
    LOGGER.info(
        "Cohort sampling: eligible pairs=%s; selected pairs=%s; "
        "excluded pairs=%s; applied=%s",
        sampling_metadata["n_eligible"],
        sampling_metadata["n_selected"],
        sampling_metadata["n_excluded"],
        sampling_metadata["applied"],
    )

    # Split into train/validation/test sets
    train_df, val_df, test_df = load_or_make_split(protein_pairs, args)
    args.has_validation_split = val_df is not None and not val_df.empty
    args.evaluate_test_metrics = (
        args.eval_test_set or not args.has_validation_split)
    validate_splits(train_df=train_df, val_df=val_df, test_df=test_df)

    # Prepare to run the models
    execution_id = args.execution_id or uuid.uuid4().hex
    output_paths = prepare_outputs(args)
    sampling_metadata["selected_examples_path"] = (
        str(output_paths.selected_examples_path)
        if selection_manifest is not None else None
    )
    split_assignments = make_split_assignments(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    split_metadata = compute_split_metadata(
        args=args,
        output_paths=output_paths,
        protein_pairs=protein_pairs,
        dropped_pairs=dropped_pairs,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        execution_id=execution_id,
        n_input_pairs_before_filtering=n_input_pairs_before_filtering,
        eligible_protein_pairs=eligible_protein_pairs,
        sampling_metadata=sampling_metadata,
        protein_taxa=fasta_data.taxon_ids,
        protein_metadata_path=protein_metadata_path,
    )
    try:
        write_selection_manifest(
            selection_manifest=selection_manifest,
            output_path=output_paths.selected_examples_path,
            append_results=args.append_results,
        )
        write_split_artifacts(
            split_assignments=split_assignments,
            dropped_pairs=dropped_pairs,
            split_metadata=split_metadata,
            output_paths=output_paths,
            append_results=args.append_results,
        )
        initialize_output_files(args, output_paths)
        append_invocation_log(
            invocation_log_entry(
                args=args,
                output_paths=output_paths,
                execution_id=execution_id,
            ),
            output_paths.invocations_path,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    # Define the model configurations to run
    baseline_classifiers = [
        classifier_name
        for classifier_name in args.classifiers
        if is_baseline_classifier(classifier_name)
    ]
    learned_classifiers = [
        classifier_name
        for classifier_name in args.classifiers
        if not is_baseline_classifier(classifier_name)
    ]

    # Run the baseline models
    for classifier_name in baseline_classifiers:
        run_model_reruns(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            x_train=train_df,
            x_val=val_df,
            x_test=test_df if args.evaluate_test_metrics else None,
            feature_name=FEATURELESS_FEATURE,
            classifier_name=classifier_name,
            execution_id=execution_id,
            args=args,
            output_paths=output_paths,
        )

    # Extract the combined feature set
    if learned_classifiers:
        feature_name = make_feature_name(args.features)
        x_train, x_val, x_test = build_feature_matrices(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df if args.evaluate_test_metrics else None,
            sequences=sequences,
            feature_types=args.features,
            args=args,
        )

    # Run the learned classifiers
    for classifier_name in learned_classifiers:
        run_model_reruns(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            feature_name=feature_name,
            classifier_name=classifier_name,
            execution_id=execution_id,
            args=args,
            output_paths=output_paths,
        )

    # Summarize and plot model performance
    summaries = summarize_model_outputs(output_paths)
    heatmap_paths = plot_model_outputs(output_paths)
    log_model_outputs(summaries, output_paths, heatmap_paths)


if __name__ == "__main__":
    main()
