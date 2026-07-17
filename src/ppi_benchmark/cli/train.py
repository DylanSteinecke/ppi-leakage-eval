#!/usr/bin/env python3

"""
Train and evaluate PPI predictors from the command line.

This script owns CLI parsing and high-level orchestration. Domain-specific
input validation, feature extraction, model evaluation, and result writing live
in focused helper modules so future pipeline variants can reuse them.
"""

import argparse
import json
import logging
import re
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifact_io import (
    append_dataframe,
    reset_output_file,
    write_dataframe_threadsafe,
)
from ..backends import (
    TORCH_BACKEND,
    backend_name_for_classifier,
    make_model_backend,
)
from ..backends.models import (
    CLASSIFIER_CHOICES,
    is_baseline_classifier,
)
from ..datasets.common import (
    discover_dataset_metadata_path,
    discover_protein_metadata_path,
    FASTA_ID_FORMAT_CHOICES,
    read_dataset_provenance,
    read_fasta_with_taxa,
)
from ..evaluation import (
    BinaryClassificationPolicy,
    THRESHOLD_SELECTION_CHOICES,
    VALIDATION_F1_THRESHOLD,
)
from ..features import (
    FEATURE_CHOICES,
    PLM_FEATURE,
    build_feature_matrices,
    compose_split_feature_matrices,
    make_feature_name,
    normalize_feature_types,
    unique_protein_ids,
)
from ..reporting.performance import (
    PERFORMANCE_FILENAME,
    PerformanceTracker,
    append_performance_report,
    peak_memory_bytes,
)
from ..reporting.plot_common import legacy_f1_heatmap_output_paths
from ..reporting.run_plots import (
    plot_metrics_summary,
    plot_train_test_f1_heatmap,
    plot_train_test_metrics_summary,
    plot_train_test_metrics_summary_png,
)
from ..reporting.summaries import summarize_metrics
from ..splitting.artifacts import (
    add_source_row_index,
    append_invocation_log,
    compute_split_metadata,
    DROPPED_PAIRS_FILENAME,
    invocation_log_entry,
    INVOCATIONS_FILENAME,
    make_split_assignments,
    SEQUENCE_CLUSTER_ASSIGNMENTS_FILENAME,
    SOURCE_ROW_INDEX_COLUMN,
    SPLIT_ASSIGNMENTS_FILENAME,
    SPLIT_METADATA_FILENAME,
    SPLITS_DIRNAME,
    write_metadata_json,
    write_split_artifacts,
)
from ..splitting.cohort import (
    SAMPLING_DIRNAME,
    SELECTED_EXAMPLES_FILENAME,
    SamplingSpec,
    select_examples,
    write_selection_manifest,
)
from ..splitting.dispatch import (
    load_or_make_split,
    normalized_split_values,
    protein_ids_in_pairs,
    validate_splits,
)
from ..splitting.grouping import (
    SEQUENCE_CLUSTER_METHODS,
    SequenceClusterParameters,
    resolve_sequence_clusters,
)
from ..splitting.negative_sampling import normalize_negative_construction
from ..splitting.preparation import prepare_input_data
from ..splitting.protocols import (
    PROVIDED_SPLIT_STRATEGY,
    RANDOM_SPLIT_STRATEGY,
    SPLIT_STRATEGY_CHOICES,
    get_split_strategy,
)
from ..protein_encoders import (
    DEFAULT_ESM2_MODEL,
    DEFAULT_PROTEIN_ENCODER_ADAPTER,
    PLM_POOLING_CHOICES,
    PLM_PRECISION_CHOICES,
    PLM_TRUNCATION_CHOICES,
    PROTEIN_ENCODER_ADAPTER_CHOICES,
    PROTEIN_ENCODER_PRESET_CHOICES,
    EmbeddingCache,
    FrozenProteinEncoder,
    create_protein_encoder,
    get_protein_encoder_preset,
)
from ..schema import EVALUATION_SCHEMA_VERSION
from ..tasks import PPI_TASK
from ..torch_utils import TORCH_DEVICE_CHOICES, TORCH_TRAINING_PRECISIONS
from ..training import TaskSplitData, fit_and_evaluate_task
from .arg_types import (
    auto_or_cluster_mode,
    auto_or_positive_float,
    non_empty_string,
    nonnegative_float,
    nonnegative_proportion,
    positive_float,
    positive_int,
    proportion,
    unit_interval,
)

FEATURELESS_FEATURE = "none"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOGGER = logging.getLogger(__name__)
PLOTS_DIRNAME = "plots"
PREDICTIONS_FILENAME = "predictions.csv"
TRAINING_HISTORY_FILENAME = "training_history.csv"
PROTEIN_ENCODER_METADATA_FILENAME = "protein_encoder.json"
CHECKPOINTS_DIRNAME = "checkpoints"
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
TRAINING_HISTORY_COLUMNS = (
    "evaluation_schema_version",
    "task",
    "execution_id",
    "model_name",
    "classifier",
    "backend",
    "features",
    "run_number",
    "model_seed",
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_auprc",
    "monitor_metric",
    "monitor_value",
    "improved",
    "learning_rate",
    "epoch_seconds",
    "train_batches",
    "validation_batches",
)


#######
# CLI #
#######
def argument_parser(
        argv: Sequence[str] | None = None,
    ) -> argparse.Namespace:
    """
    Argument parser for protein-protein interaction prediction.

    ``argv`` is injectable so parser and end-to-end orchestration tests do not
    need to start a new Python process. Console entry points continue to parse
    ``sys.argv`` when it is omitted.
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
        "--dataset-metadata",
        default=None,
        help=(
            "Optional ppi-prepare dataset_metadata.json. A canonical sidecar "
            "beside pairs.csv is discovered automatically."
        ),
    )
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
    input_group.add_argument(
        "--sequence-clusters",
        default=None,
        help=(
            "Supplied CSV with protein_id and cluster_id columns for C2/C3. "
            "Mutually exclusive with automatic sequence clustering."
        ))

    grouping_group = parser.add_argument_group("Automatic sequence grouping")
    grouping_group.add_argument(
        "--sequence-cluster-method",
        choices=SEQUENCE_CLUSTER_METHODS,
        default=None,
        help="Generate task-independent protein groups for a C2/C3 split.",
    )
    grouping_group.add_argument(
        "--sequence-cluster-min-seq-id",
        type=unit_interval,
        default=argparse.SUPPRESS,
        help="MMseqs2 minimum sequence identity (default: 0.30).",
    )
    grouping_group.add_argument(
        "--sequence-cluster-coverage",
        type=unit_interval,
        default=argparse.SUPPRESS,
        help="MMseqs2 coverage threshold (default: 0.80).",
    )
    grouping_group.add_argument(
        "--sequence-cluster-cov-mode",
        type=int,
        choices=range(6),
        default=argparse.SUPPRESS,
        help="MMseqs2 coverage mode, from 0 through 5 (default: 0).",
    )
    grouping_group.add_argument(
        "--sequence-cluster-evalue",
        type=positive_float,
        default=argparse.SUPPRESS,
        help="MMseqs2 E-value threshold (default: 0.001).",
    )
    grouping_group.add_argument(
        "--sequence-cluster-sensitivity",
        type=auto_or_positive_float,
        default=argparse.SUPPRESS,
        metavar="AUTO_OR_FLOAT",
        help=(
            "MMseqs2 sensitivity. 'auto' omits -s so MMseqs2 derives it "
            "from sequence identity (default: auto)."
        ),
    )
    grouping_group.add_argument(
        "--sequence-cluster-cluster-mode",
        type=auto_or_cluster_mode,
        default=argparse.SUPPRESS,
        metavar="AUTO_OR_0_TO_3",
        help=(
            "MMseqs2 cluster mode. 'auto' omits --cluster-mode so MMseqs2 "
            "derives it from coverage mode (default: auto)."
        ),
    )
    grouping_group.add_argument(
        "--sequence-cluster-threads",
        type=positive_int,
        default=argparse.SUPPRESS,
        help="MMseqs2 worker threads (default: 1).",
    )
    grouping_group.add_argument(
        "--sequence-cluster-cache-dir",
        default=argparse.SUPPRESS,
        help=(
            "Content-addressed grouping cache. Defaults to "
            "$PPI_SEQUENCE_CLUSTER_CACHE_DIR, then "
            "$XDG_CACHE_HOME/ppi-leakage/sequence_clusters, then "
            "~/.cache/ppi-leakage/sequence_clusters."
        ),
    )

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
        help=argparse.SUPPRESS)
    output_group.add_argument(
        "--execution-id", default=None,
        help=argparse.SUPPRESS)
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

    # Frozen protein language-model args
    plm_group = parser.add_argument_group("Frozen protein encoder")
    plm_group.add_argument(
        "--plm-preset",
        choices=PROTEIN_ENCODER_PRESET_CHOICES,
        default=None,
        help="Approved self-supervised checkpoint and safe runtime defaults")
    plm_group.add_argument(
        "--plm-adapter",
        choices=PROTEIN_ENCODER_ADAPTER_CHOICES,
        default=None,
        help=(
            "Model-family adapter controlling tokenization, pooling, and "
            "residue alignment"
        ))
    plm_group.add_argument(
        "--plm-model",
        default=None,
        help="Hugging Face model ID or local model directory")
    plm_group.add_argument(
        "--plm-revision",
        default=None,
        help=(
            "Exact model revision. Remote models require an immutable "
            "40-character Hugging Face commit hash."
        ))
    plm_group.add_argument(
        "--plm-tokenizer-revision",
        default=None,
        help="Exact tokenizer revision; defaults to --plm-revision")
    plm_group.add_argument(
        "--plm-pooling",
        choices=PLM_POOLING_CHOICES,
        default=None,
        help="Pool residue representations by masked mean or CLS token")
    plm_group.add_argument(
        "--plm-max-length",
        type=positive_int,
        default=None,
        help="Maximum tokenized length including model special tokens")
    plm_group.add_argument(
        "--plm-truncation-policy",
        choices=PLM_TRUNCATION_CHOICES,
        default=None,
        help="Fail on overlength proteins or truncate them explicitly")
    plm_group.add_argument(
        "--plm-precision",
        choices=PLM_PRECISION_CHOICES,
        default=None,
        help="Frozen encoder compute precision and cache namespace")
    plm_group.add_argument(
        "--plm-device",
        choices=TORCH_DEVICE_CHOICES,
        default=None,
        help="Device for frozen PLM inference")
    plm_group.add_argument(
        "--plm-max-batch-tokens",
        type=positive_int,
        default=None,
        help="Maximum padded tokens per length-bucketed encoder batch")
    plm_group.add_argument(
        "--plm-max-batch-sequences",
        type=positive_int,
        default=None,
        help="Safety cap on sequences per token-budgeted encoder batch")
    plm_group.add_argument(
        "--embedding-cache-dir",
        default=None,
        help=(
            "Shared frozen-embedding cache directory. Defaults to "
            "$PPI_EMBEDDING_CACHE_DIR or the user cache directory."
        ))

    # Model args
    model_group = parser.add_argument_group("Models")
    model_group.add_argument(
        "--classifier", dest="classifiers",
        choices=CLASSIFIER_CHOICES, nargs="+", default=["logistic"],
        help="One or more classifier model types")
    model_group.add_argument(
        "--classifiers", dest="classifiers",
        choices=CLASSIFIER_CHOICES, nargs="+", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS)
    model_group.add_argument(
        "--threshold-selection",
        choices=THRESHOLD_SELECTION_CHOICES,
        default=VALIDATION_F1_THRESHOLD,
        help=(
            "Select learned-model decision thresholds by validation F1, or "
            "retain each backend's fixed default threshold."
        ))

    # Training args
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--max-iter", type=positive_int, default=1000)
    training_group.add_argument(
        "--train-size", type=proportion, default=0.80,
        help="Fraction of pairs assigned to the training set")
    training_group.add_argument(
        "--val-size", type=nonnegative_proportion, default=0.10,
        help="Fraction of pairs assigned to the validation set")
    training_group.add_argument(
        "--eval-test-set", action="store_true",
        help="Evaluate on the test split. In validation runs, omit this to "
            "keep test held out.")
    training_group.add_argument(
        "--seed", type=int, default=None, help=argparse.SUPPRESS)
    training_group.add_argument(
        "--split-seed", type=int, default=None,
        help="Seed controlling generated train/validation/test assignments")
    training_group.add_argument(
        "--model-seeds", type=int, nargs="+", default=None,
        help="Explicit random seeds for independent model fits")
    training_group.add_argument(
        "--model-seed", type=int, default=None, help=argparse.SUPPRESS)
    training_group.add_argument(
        "--split-col", default=None,
        help="Optional column with train, validation, and test labels")
    training_group.add_argument(
        "--split-name", type=non_empty_string, default=None,
        help=argparse.SUPPRESS)
    training_group.add_argument(
        "--split-strategy", choices=SPLIT_STRATEGY_CHOICES, default=None,
        help="Optional strategy for creating train/validation/test splits")
    training_group.add_argument(
        "--n-split-trials", type=positive_int, default=100,
        help="Candidate assignments evaluated for C1/C2/C3 splitting")
    training_group.add_argument(
        "--num-reruns", type=positive_int, default=None,
        help=argparse.SUPPRESS)

    # Torch MLP args
    torch_group = parser.add_argument_group("Torch MLP")
    torch_group.add_argument(
        "--torch-max-epochs", type=positive_int, default=50,
        help="Maximum epochs for torch_mlp training")
    torch_group.add_argument(
        "--torch-batch-size", type=positive_int, default=256,
        help="Sparse row-batch size for torch_mlp training and prediction")
    torch_group.add_argument(
        "--torch-hidden-dim", type=positive_int, default=64,
        help="Hidden-layer width for torch_mlp")
    torch_group.add_argument(
        "--torch-learning-rate", type=positive_float, default=1e-3)
    torch_group.add_argument(
        "--torch-weight-decay", type=nonnegative_float, default=1e-4)
    torch_group.add_argument(
        "--torch-dropout", type=nonnegative_proportion, default=0.1)
    torch_group.add_argument(
        "--torch-patience", type=positive_int, default=5,
        help="Validation epochs without improvement before early stopping")
    torch_group.add_argument(
        "--torch-min-delta", type=nonnegative_float, default=1e-4,
        help="Minimum validation-monitor improvement counted as progress")
    torch_group.add_argument(
        "--torch-device", choices=TORCH_DEVICE_CHOICES,
        default="auto",
        help="Device for torch_mlp; auto prefers CUDA, then MPS, then CPU")
    torch_group.add_argument(
        "--torch-precision", choices=TORCH_TRAINING_PRECISIONS,
        default="float32",
        help="Autocast precision used by the shared Torch trainer")
    torch_group.add_argument(
        "--torch-validation-monitor", choices=("auprc", "loss"),
        default="auprc",
        help="Validation quantity used for early stopping")
    torch_group.add_argument(
        "--torch-resume-from", default=None,
        help=(
            "Resume torch_mlp from a prior *.last.pt checkpoint. The target "
            "--torch-max-epochs is the total epoch count."
        ))

    args = parser.parse_args(argv)
    preset = (
        None
        if args.plm_preset is None
        else get_protein_encoder_preset(args.plm_preset)
    )
    if preset is not None:
        identity_values = {
            "plm_adapter": preset.adapter,
            "plm_model": preset.model_name,
            "plm_revision": preset.model_revision,
        }
        for argument_name, expected_value in identity_values.items():
            supplied_value = getattr(args, argument_name)
            if supplied_value is not None and supplied_value != expected_value:
                parser.error(
                    f"--plm-preset {preset.name} conflicts with "
                    f"--{argument_name.replace('_', '-')}={supplied_value}."
                )
            setattr(args, argument_name, expected_value)
        if args.plm_tokenizer_revision is None:
            args.plm_tokenizer_revision = preset.model_revision
    plm_defaults = {
        "plm_adapter": (
            DEFAULT_PROTEIN_ENCODER_ADAPTER
            if preset is None else preset.adapter
        ),
        "plm_model": DEFAULT_ESM2_MODEL if preset is None else preset.model_name,
        "plm_pooling": "mean" if preset is None else preset.pooling,
        "plm_max_length": 1024 if preset is None else preset.maximum_length,
        "plm_truncation_policy": (
            "error" if preset is None else preset.truncation_policy
        ),
        "plm_precision": "float32" if preset is None else preset.precision,
        "plm_device": "auto" if preset is None else preset.device,
        "plm_max_batch_tokens": (
            4096 if preset is None else preset.max_batch_tokens
        ),
        "plm_max_batch_sequences": (
            32 if preset is None else preset.max_batch_sequences
        ),
    }
    for argument_name, default_value in plm_defaults.items():
        if getattr(args, argument_name) is None:
            setattr(args, argument_name, default_value)
    args.plm_preset_metadata = (
        None if preset is None else preset.to_dict()
    )
    if args.split_col and args.split_strategy:
        parser.error("--split-col and --split-strategy cannot both be set.")
    if args.split_name and not args.split_col:
        parser.error("--split-name can only be used with --split-col.")
    if args.train_size + args.val_size >= 1.0:
        parser.error("--train-size + --val-size must be less than 1.")

    args.cli_deprecation_warnings = []
    if args.seed is not None:
        args.cli_deprecation_warnings.append(
            "--seed is deprecated; use --split-seed and --model-seeds."
        )
    if args.model_seed is not None or args.num_reruns is not None:
        args.cli_deprecation_warnings.append(
            "--model-seed/--num-reruns are deprecated; pass every seed "
            "explicitly with --model-seeds."
        )
    if args.append_results:
        args.cli_deprecation_warnings.append(
            "--append-results is deprecated; use one immutable run "
            "directory per configuration and aggregate run directories."
        )
    if args.execution_id is not None:
        args.cli_deprecation_warnings.append(
            "--execution-id is reserved for legacy orchestration and is "
            "deprecated for direct use."
        )
    if args.split_name is not None:
        args.cli_deprecation_warnings.append(
            "--split-name is deprecated; use the provided split column name "
            "or experiment configuration metadata."
        )

    if args.split_col:
        args.effective_split_strategy = PROVIDED_SPLIT_STRATEGY
    else:
        args.effective_split_strategy = (
            args.split_strategy or RANDOM_SPLIT_STRATEGY)

    automatic_grouping_option_names = (
        "sequence_cluster_min_seq_id",
        "sequence_cluster_coverage",
        "sequence_cluster_cov_mode",
        "sequence_cluster_evalue",
        "sequence_cluster_sensitivity",
        "sequence_cluster_cluster_mode",
        "sequence_cluster_threads",
        "sequence_cluster_cache_dir",
    )
    explicit_automatic_options = [
        option_name
        for option_name in automatic_grouping_option_names
        if hasattr(args, option_name)
    ]
    if args.sequence_clusters is not None and (
        args.sequence_cluster_method is not None
        or explicit_automatic_options
    ):
        parser.error(
            "--sequence-clusters cannot be combined with automatic "
            "sequence-cluster options."
        )
    if (
        args.sequence_cluster_method is None
        and explicit_automatic_options
    ):
        option = explicit_automatic_options[0].replace("_", "-")
        parser.error(
            f"--{option} requires --sequence-cluster-method mmseqs2."
        )
    grouping_requested = (
        args.sequence_clusters is not None
        or args.sequence_cluster_method is not None
    )
    split_spec = get_split_strategy(
        PPI_TASK.name,
        args.effective_split_strategy,
    )
    if (
        grouping_requested
        and "sequence_cluster" not in split_spec.allowed_grouping_kinds
    ):
        parser.error(
            "Sequence-cluster grouping is incompatible with split strategy "
            f"{args.effective_split_strategy!r}; use C2 or C3."
        )

    sequence_cluster_defaults = SequenceClusterParameters()
    sequence_cluster_default_values = {
        "sequence_cluster_min_seq_id": sequence_cluster_defaults.min_seq_id,
        "sequence_cluster_coverage": sequence_cluster_defaults.coverage,
        "sequence_cluster_cov_mode": sequence_cluster_defaults.cov_mode,
        "sequence_cluster_evalue": sequence_cluster_defaults.evalue,
        "sequence_cluster_sensitivity": sequence_cluster_defaults.sensitivity,
        "sequence_cluster_cluster_mode": (
            sequence_cluster_defaults.cluster_mode
        ),
        "sequence_cluster_threads": sequence_cluster_defaults.threads,
        "sequence_cluster_cache_dir": None,
    }
    for option_name, default_value in sequence_cluster_default_values.items():
        if not hasattr(args, option_name):
            setattr(args, option_name, default_value)
    args.n_discarded_edges = 0
    args.discarded_edge_fraction = 0.0
    args.split_audit = None
    args.sampling_requested = (
        args.max_pairs is not None or args.sample_fraction is not None)
    args.split_seed = (
        args.split_seed
        if args.split_seed is not None
        else (0 if args.seed is None else args.seed)
    )
    if args.model_seeds is not None:
        if args.model_seed is not None or args.num_reruns is not None:
            parser.error(
                "--model-seeds cannot be combined with deprecated "
                "--model-seed or --num-reruns."
            )
        model_seeds = tuple(args.model_seeds)
    else:
        first_model_seed = (
            args.model_seed
            if args.model_seed is not None
            else (0 if args.seed is None else args.seed)
        )
        n_model_seeds = 1 if args.num_reruns is None else args.num_reruns
        model_seeds = tuple(
            first_model_seed + offset
            for offset in range(n_model_seeds)
        )
    if len(set(model_seeds)) != len(model_seeds):
        parser.error("--model-seeds cannot contain duplicate values.")
    args.model_seeds = model_seeds
    args.model_seed = model_seeds[0]
    args.num_reruns = len(model_seeds)
    if args.torch_resume_from and len(args.model_seeds) != 1:
        parser.error("--torch-resume-from requires exactly one model seed.")
    if args.torch_resume_from and "torch_mlp" not in args.classifiers:
        parser.error("--torch-resume-from requires --classifier torch_mlp.")

    try:
        args.features = normalize_feature_types(args.features)
    except ValueError as exc:
        parser.error(str(exc))
    if PLM_FEATURE in args.features and len(args.features) != 1:
        parser.error(
            "--features plm cannot currently be concatenated with k-mer "
            "features without inefficient dense/sparse conversion."
        )
    if PLM_FEATURE in args.features and args.plm_revision is None:
        parser.error("--features plm requires --plm-revision.")
    if args.torch_resume_from:
        resume_path = Path(args.torch_resume_from).expanduser().resolve()
        run_dir = Path(args.run_dir).expanduser().resolve()
        if resume_path == run_dir or run_dir in resume_path.parents:
            parser.error(
                "--torch-resume-from must be outside --run-dir. Resume into "
                "a new output directory so the source checkpoint remains "
                "immutable."
            )

    args.protein_encoder_metadata = None

    return args


def configure_logging(args: argparse.Namespace) -> None:
    """
    Configure command-line logging.
    """
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )
    for warning_message in args.cli_deprecation_warnings:
        LOGGER.warning("Deprecated CLI: %s", warning_message)


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
    checkpoints_dir: Path
    selected_examples_path: Path
    sequence_cluster_assignments_path: Path
    protein_encoder_metadata_path: Path
    training_history_path: Path
    predictions_path: Path | None
    train_metrics_path: Path
    val_metrics_path: Path | None
    test_metrics_path: Path | None
    split_assignments_path: Path
    dropped_pairs_path: Path
    split_metadata_path: Path
    invocations_path: Path
    performance_path: Path
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


def safe_artifact_component(value: str) -> str:
    """Return a path-safe component for model-specific artifacts."""
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe_value or "unnamed"


def model_checkpoint_paths(
        checkpoints_dir: Path, execution_id: str, model_name: str,
        run_number: int,
    ) -> tuple[Path, Path]:
    """Return deterministic best/last paths for one model rerun."""
    execution_component = safe_artifact_component(execution_id)
    model_component = safe_artifact_component(model_name)
    checkpoint_prefix = (
        checkpoints_dir / execution_component
        / f"{model_component}__run_{run_number}"
    )
    return (
        checkpoint_prefix.with_suffix(".best.pt"),
        checkpoint_prefix.with_suffix(".last.pt"),
    )


def feature_metadata(
        feature_name: str, args: argparse.Namespace
    ) -> dict[str, Any]:
    """
    Return feature metadata stored with each result row.
    """
    is_featureless = feature_name == FEATURELESS_FEATURE
    is_plm = (
        not is_featureless
        and getattr(args, "protein_encoder_metadata", None) is not None
    )
    feature_metadata = {
        "features": FEATURELESS_FEATURE if is_featureless else feature_name,
        "k": np.nan if is_featureless or is_plm else args.k,
        "bm25_k1": np.nan if is_featureless or is_plm else args.bm25_k1,
        "bm25_b": np.nan if is_featureless or is_plm else args.bm25_b,
        "encoder_adapter": np.nan,
        "encoder_fingerprint": np.nan,
        "encoder_model": np.nan,
        "encoder_revision": np.nan,
        "encoder_pooling": np.nan,
        "encoder_maximum_length": np.nan,
        "encoder_precision": np.nan,
        "encoder_label_independent": np.nan,
        "encoder_checkpoint_sha256": np.nan,
        "encoder_training_split_sha256": np.nan,
    }
    if is_plm:
        encoder_metadata = args.protein_encoder_metadata
        encoder_spec = encoder_metadata["encoder_spec"]
        feature_metadata.update({
            "encoder_adapter": encoder_metadata["adapter"],
            "encoder_fingerprint": encoder_metadata[
                "encoder_fingerprint"],
            "encoder_model": encoder_spec["model_name"],
            "encoder_revision": encoder_spec["model_revision"],
            "encoder_pooling": encoder_spec["pooling"],
            "encoder_maximum_length": encoder_spec["maximum_length"],
            "encoder_precision": encoder_spec["precision"],
            "encoder_label_independent": encoder_spec[
                "label_independent"],
            "encoder_checkpoint_sha256": encoder_spec[
                "checkpoint_sha256"],
            "encoder_training_split_sha256": encoder_spec[
                "training_split_sha256"],
        })

    return feature_metadata


def frozen_plm_feature_name(encoder_metadata: dict[str, Any]) -> str:
    """Return a model-specific feature name that cannot mix PLM revisions."""
    encoder_spec = encoder_metadata["encoder_spec"]
    model_label = Path(str(encoder_spec["model_name"])).name
    model_label = safe_artifact_component(model_label)
    fingerprint = str(encoder_metadata["encoder_fingerprint"])
    return f"plm-{model_label}-{fingerprint[:12]}"


def build_frozen_plm_feature_matrices(
        cohort_df: pd.DataFrame, train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, sequences: dict[str, str],
        evaluate_test_metrics: bool, args: argparse.Namespace,
    ) -> tuple[Any, Any | None, Any | None, str, dict[str, Any]]:
    """Encode unique proteins once and compose dense symmetric pair rows."""
    encoder = create_protein_encoder(
        adapter=args.plm_adapter,
        model_name=args.plm_model,
        model_revision=args.plm_revision,
        tokenizer_revision=args.plm_tokenizer_revision,
        pooling=args.plm_pooling,
        maximum_length=args.plm_max_length,
        precision=args.plm_precision,
        truncation_policy=args.plm_truncation_policy,
        device=args.plm_device,
    )
    all_protein_ids = unique_protein_ids((cohort_df,))
    with EmbeddingCache(
            encoder_spec=encoder.spec,
            cache_dir=args.embedding_cache_dir,
        ) as cache:
        cached_encoder = FrozenProteinEncoder(
            encoder=encoder,
            cache=cache,
            max_batch_tokens=args.plm_max_batch_tokens,
            max_batch_sequences=args.plm_max_batch_sequences,
        )
        embedding_table = cached_encoder.encode(
            protein_sequences=sequences,
            protein_ids=all_protein_ids.tolist(),
        )
    model_loaded_for_cache_misses = encoder.is_loaded
    encoder.release_model()
    protein_ids = pd.Index(embedding_table.protein_ids)
    x_train, x_val, x_test = compose_split_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df if evaluate_test_metrics else None,
        protein_ids=protein_ids,
        protein_features=embedding_table.embeddings,
    )
    encoder_metadata = {
        **embedding_table.metadata,
        "adapter": args.plm_adapter,
        "sequence_preprocessing": encoder.sequence_preprocessing,
        "preset": args.plm_preset_metadata,
        "model_loaded_for_cache_misses": model_loaded_for_cache_misses,
        "model_released_after_encoding": model_loaded_for_cache_misses,
    }
    return (
        x_train,
        x_val,
        x_test,
        frozen_plm_feature_name(encoder_metadata),
        encoder_metadata,
    )


def write_protein_encoder_metadata(
        metadata: dict[str, Any], output_path: Path,
        append_results: bool,
    ) -> None:
    """Write encoder/cache provenance without mixing incompatible specs."""
    if append_results and output_path.exists():
        with output_path.open("r", encoding="utf-8") as input_file:
            existing_metadata = json.load(input_file)
        if existing_metadata.get("encoder_fingerprint") != metadata.get(
                "encoder_fingerprint"):
            raise ValueError(
                f"Cannot append a different frozen encoder to "
                f"{output_path.parent}. Use a new --run-dir."
            )
    write_metadata_json(metadata, output_path)


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
        execution_id: str, args: argparse.Namespace,
        best_checkpoint_path: Path | None = None,
        last_checkpoint_path: Path | None = None,
        resume_from: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None,
               pd.DataFrame | None, pd.DataFrame | None,
               pd.DataFrame | None, dict[str, Any]]:
    """
    Train and evaluate one model configuration for one run number.
    """
    # Define model metadata
    model_name = make_model_name(feature_name, classifier_name)
    is_baseline = is_baseline_classifier(classifier_name)
    run_seed = args.model_seeds[run_number - 1]
    model_seed = np.nan if is_baseline else run_seed
    backend_name = backend_name_for_classifier(classifier_name)
    backend_max_iter = (
        args.torch_max_epochs
        if backend_name == TORCH_BACKEND
        else args.max_iter
    )
    max_iter = np.nan if is_baseline else backend_max_iter
    feature_meta = feature_metadata(feature_name, args)
    run_identity = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "task": PPI_TASK.name,
        "execution_id": execution_id,
        "model_name": model_name,
        "run_number": run_number,
        "model_seed": model_seed,
    }

    # Define model backend
    backend_options = None
    if backend_name == TORCH_BACKEND:
        backend_options = {
            "batch_size": args.torch_batch_size,
            "hidden_dim": args.torch_hidden_dim,
            "learning_rate": args.torch_learning_rate,
            "weight_decay": args.torch_weight_decay,
            "dropout": args.torch_dropout,
            "patience": args.torch_patience,
            "min_delta": args.torch_min_delta,
            "device": args.torch_device,
            "precision": args.torch_precision,
            "validation_monitor": args.torch_validation_monitor,
        }
    backend = make_model_backend(
        classifier_name=classifier_name,
        max_iter=backend_max_iter,
        random_state=run_seed,
        backend_name=backend_name,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        resume_from=resume_from,
        backend_options=backend_options,
        task_name=PPI_TASK.name,
        task_schema_version=PPI_TASK.schema_version,
    )
    train_split = TaskSplitData("train", train_df, x_train)
    validation_split = None
    if val_df is not None and x_val is not None:
        validation_split = TaskSplitData("val", val_df, x_val)
    test_split = None
    if args.evaluate_test_metrics:
        if x_test is None:
            raise ValueError("Test evaluation requested without test features.")
        test_split = TaskSplitData("test", test_df, x_test)

    model_run_result = fit_and_evaluate_task(
        backend=backend,
        task=PPI_TASK,
        train=train_split,
        validation=validation_split,
        test=test_split,
        evaluation_policy=BinaryClassificationPolicy(
            threshold_strategy=args.threshold_selection,
            force_fixed=is_baseline,
        ),
    )
    fit_result = model_run_result.fit_result
    fit_seconds = fit_result.fit_seconds
    iteration_report = fit_result.iteration_report
    training_metadata = fit_result.metadata
    solver_iterations = (
        np.nan if iteration_report is None else iteration_report["maximum"])
    threshold_selection = model_run_result.operating_point

    # Define the evaluate metrics tables
    n_val = 0 if val_df is None else len(val_df)
    n_total = len(train_df) + n_val + len(test_df)
    actual_train_size = len(train_df) / n_total
    actual_val_size = n_val / n_total
    actual_test_size = len(test_df) / n_total
    metrics_metadata = {
        **run_identity,
        "split_seed": args.split_seed,
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
        "target_test_size": 1.0 - args.train_size - args.val_size,
        "actual_train_size": actual_train_size,
        "actual_val_size": actual_val_size,
        "actual_test_size": actual_test_size,
        "n_discarded_edges": args.n_discarded_edges,
        "discarded_edge_fraction": args.discarded_edge_fraction,
        "fit_seconds": fit_seconds,
        "solver_iterations": solver_iterations,
        "decision_threshold": threshold_selection.threshold,
        "default_decision_threshold": (
            threshold_selection.default_threshold),
        "threshold_selection": threshold_selection.strategy,
        "threshold_metric": threshold_selection.metric_name,
        "threshold_metric_value": (
            np.nan
            if threshold_selection.metric_value is None
            else threshold_selection.metric_value
        ),
    }

    training_history_df = None
    if fit_result.training_history:
        history_metadata = {
            **run_identity,
            "classifier": classifier_name,
            "backend": backend.backend_name,
            "features": feature_meta["features"],
        }
        training_history_df = pd.DataFrame([
            {**history_metadata, **history_row}
            for history_row in fit_result.training_history
        ]).reindex(columns=TRAINING_HISTORY_COLUMNS)

    def metrics_frame(split_name: str) -> pd.DataFrame | None:
        evaluation = model_run_result.split_evaluations.get(split_name)
        if evaluation is None:
            return None
        frame = make_metrics_df(
            metadata=metrics_metadata,
            metrics=dict(evaluation.metrics),
            split_name=split_name,
        )
        frame["evaluation_seconds"] = evaluation.evaluation_seconds
        return frame

    train_metrics_df = metrics_frame("train")
    if train_metrics_df is None:
        raise RuntimeError("Training evaluation was not produced.")
    val_metrics_df = metrics_frame("val")
    test_metrics_df = metrics_frame("test")
    predictions_df = None
    if args.evaluate_test_metrics:
        test_evaluation = model_run_result.split_evaluations.get("test")
        if test_evaluation is None:
            raise RuntimeError("Test evaluation was not produced.")
        predictions_df = PPI_TASK.prediction_frame(
            examples=test_df,
            evaluation=test_evaluation,
            model_metadata={
                "execution_id": run_identity["execution_id"],
                "model_name": run_identity["model_name"],
                "run_number": run_identity["run_number"],
                "model_seed": run_identity["model_seed"],
                "features": feature_meta["features"],
                "classifier": classifier_name,
                "k": feature_meta["k"],
            },
        )
        predictions_df["decision_threshold"] = (
            threshold_selection.threshold)
        predictions_df["threshold_selection"] = (
            threshold_selection.strategy)
        predictions_df["encoder_fingerprint"] = feature_meta[
            "encoder_fingerprint"]
        predictions_df["encoder_adapter"] = feature_meta[
            "encoder_adapter"]

    model_performance = {
        "evaluation_schema_version": run_identity[
            "evaluation_schema_version"],
        "task": run_identity["task"],
        "model_name": run_identity["model_name"],
        "feature_name": feature_name,
        "classifier": classifier_name,
        "backend": backend.backend_name,
        "run_number": run_identity["run_number"],
        "model_seed": None if is_baseline else run_seed,
        "fit_seconds": float(fit_seconds),
        "evaluation_seconds": {
            split_name: evaluation.evaluation_seconds
            for split_name, evaluation in (
                model_run_result.split_evaluations.items()
            )
        },
        "total_seconds": model_run_result.total_seconds,
        "solver_iterations": iteration_report,
        "training": training_metadata,
        "protein_encoder": (
            args.protein_encoder_metadata
            if feature_name != FEATURELESS_FEATURE
            else None
        ),
        "threshold_selection_seconds": (
            model_run_result.operating_point_selection_seconds),
        "decision_threshold": {
            "selected": threshold_selection.threshold,
            "default": threshold_selection.default_threshold,
            "strategy": threshold_selection.strategy,
            "metric": threshold_selection.metric_name,
            "metric_value": threshold_selection.metric_value,
            "n_candidates": threshold_selection.n_candidates,
        },
        "peak_memory_bytes_after_run": peak_memory_bytes(),
    }
    return (
        train_metrics_df,
        val_metrics_df,
        test_metrics_df,
        predictions_df,
        training_history_df,
        model_performance,
    )


def run_model_reruns(
        train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        test_df: pd.DataFrame, x_train: Any, x_val: Any | None,
        x_test: Any | None, feature_name: str, classifier_name: str,
        execution_id: str, args: argparse.Namespace,
        output_paths: OutputPaths,
    ) -> list[dict[str, Any]]:
    """
    Run one model configuration repeatedly and append each result.
    """
    model_name = make_model_name(feature_name, classifier_name)
    model_performance_records = []

    # Re-run the model
    for run_number, _ in enumerate(args.model_seeds, start=1):
        if backend_name_for_classifier(classifier_name) == TORCH_BACKEND:
            best_checkpoint_path, last_checkpoint_path = (
                model_checkpoint_paths(
                    checkpoints_dir=output_paths.checkpoints_dir,
                    execution_id=execution_id,
                    model_name=model_name,
                    run_number=run_number,
                )
            )
            resume_from = (
                None
                if args.torch_resume_from is None
                else Path(args.torch_resume_from)
            )
        else:
            best_checkpoint_path = None
            last_checkpoint_path = None
            resume_from = None
        (
            train_metrics_df,
            val_metrics_df,
            test_metrics_df,
            predictions_df,
            training_history_df,
            model_performance,
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
                best_checkpoint_path=best_checkpoint_path,
                last_checkpoint_path=last_checkpoint_path,
                resume_from=resume_from,
            )
        )
        model_performance_records.append(model_performance)
        # Save performance and prediction results
        output_dfs = (
            (train_metrics_df, output_paths.train_metrics_path),
            (val_metrics_df, output_paths.val_metrics_path),
            (test_metrics_df, output_paths.test_metrics_path),
            (predictions_df, output_paths.predictions_path),
            (training_history_df, output_paths.training_history_path),
        )
        for output_df, output_path in output_dfs:
            if output_df is not None and output_path is not None:
                append_dataframe(output_df, output_path)
        LOGGER.info(
            f"Finished model={model_name} "
            f"run={run_number}/{args.num_reruns}"
        )

    return model_performance_records


def prepare_outputs(args: argparse.Namespace) -> OutputPaths:
    """
    Resolve canonical output paths inside --run-dir.
    """
    run_dir = Path(args.run_dir)
    plots_dir = run_dir / PLOTS_DIRNAME
    splits_dir = run_dir / SPLITS_DIRNAME
    sampling_dir = run_dir / SAMPLING_DIRNAME
    checkpoints_dir = run_dir / CHECKPOINTS_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_predictions_path = run_dir / PREDICTIONS_FILENAME
    training_history_path = run_dir / TRAINING_HISTORY_FILENAME
    train_metrics_path = run_dir / TRAIN_METRICS_FILENAME
    all_val_metrics_path = run_dir / VAL_METRICS_FILENAME
    all_test_metrics_path = run_dir / TEST_METRICS_FILENAME
    split_assignments_path = splits_dir / SPLIT_ASSIGNMENTS_FILENAME
    dropped_pairs_path = splits_dir / DROPPED_PAIRS_FILENAME
    split_metadata_path = splits_dir / SPLIT_METADATA_FILENAME
    sequence_cluster_assignments_path = (
        splits_dir / SEQUENCE_CLUSTER_ASSIGNMENTS_FILENAME)
    protein_encoder_metadata_path = (
        run_dir / PROTEIN_ENCODER_METADATA_FILENAME)
    selected_examples_path = sampling_dir / SELECTED_EXAMPLES_FILENAME
    invocations_path = run_dir / INVOCATIONS_FILENAME
    performance_path = run_dir / PERFORMANCE_FILENAME
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
        checkpoints_dir=checkpoints_dir,
        selected_examples_path=selected_examples_path,
        sequence_cluster_assignments_path=(
            sequence_cluster_assignments_path),
        protein_encoder_metadata_path=protein_encoder_metadata_path,
        training_history_path=training_history_path,
        predictions_path=predictions_path,
        train_metrics_path=train_metrics_path,
        val_metrics_path=val_metrics_path,
        test_metrics_path=test_metrics_path,
        split_assignments_path=split_assignments_path,
        dropped_pairs_path=dropped_pairs_path,
        split_metadata_path=split_metadata_path,
        invocations_path=invocations_path,
        performance_path=performance_path,
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
            PROTEIN_ENCODER_METADATA_FILENAME,
            TRAINING_HISTORY_FILENAME,
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
        if output_paths.checkpoints_dir.exists():
            shutil.rmtree(output_paths.checkpoints_dir)
        reset_output_files(
            metric_prediction_paths
            + summary_paths
            + plot_paths
            + legacy_heatmap_paths
            + [output_paths.invocations_path, output_paths.performance_path],
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
        (
            output_paths.training_history_path
            if output_paths.training_history_path.exists()
            else None,
            "training history",
        ),
        (
            output_paths.checkpoints_dir
            if output_paths.checkpoints_dir.exists()
            else None,
            "model checkpoints",
        ),
        (
            output_paths.sequence_cluster_assignments_path
            if output_paths.sequence_cluster_assignments_path.exists()
            else None,
            "sequence-cluster assignments",
        ),
        (
            output_paths.protein_encoder_metadata_path
            if output_paths.protein_encoder_metadata_path.exists()
            else None,
            "protein-encoder metadata",
        ),
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


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the full CLI pipeline.
    """
    performance = PerformanceTracker()
    with performance.stage("argument_parsing"):
        args = argument_parser(argv)
        split_spec = get_split_strategy(
            PPI_TASK.name,
            args.effective_split_strategy,
        )
        configure_logging(args)

    # Load and process input data
    with performance.stage("load_pairs"):
        dataset_metadata_path = discover_dataset_metadata_path(
            pairs_path=args.pairs,
            explicit_path=args.dataset_metadata,
        )
        negative_construction = None
        try:
            dataset_provenance = read_dataset_provenance(
                pairs_path=args.pairs,
                metadata_path=dataset_metadata_path,
            )
            if dataset_provenance.metadata is not None:
                negative_construction = normalize_negative_construction(
                    dataset_provenance.metadata
                )
                if dataset_provenance.pairs_binding == "legacy_path_only":
                    LOGGER.warning(
                        "Dataset metadata is bound to pairs by a legacy path "
                        "only; regenerate it with ppi-prepare to add a "
                        "cryptographic output hash."
                    )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        args.dataset_metadata = (
            None
            if dataset_metadata_path is None
            else str(dataset_metadata_path)
        )
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
            None
            if protein_metadata_path is None
            else str(protein_metadata_path)
        )
    with performance.stage("load_sequences"):
        fasta_data = read_fasta_with_taxa(
            fasta_path=args.fasta,
            id_format=args.fasta_id_format,
            protein_metadata_path=protein_metadata_path,
            taxon_id=args.taxon_id,
        )
        sequences = fasta_data.sequences
    with performance.stage("validate_inputs"):
        eligible_protein_pairs, dropped_pairs = prepare_input_data(
            protein_pairs,
            sequences,
        )
    sequence_cluster_mapping = None
    sequence_cluster_assignments = None
    args.sequence_cluster_metadata = None
    if (
        args.sequence_clusters is not None
        or args.sequence_cluster_method is not None
    ):
        with performance.stage("resolve_sequence_clusters"):
            automatic_parameters = (
                None
                if args.sequence_cluster_method is None
                else SequenceClusterParameters(
                    method=args.sequence_cluster_method,
                    min_seq_id=args.sequence_cluster_min_seq_id,
                    coverage=args.sequence_cluster_coverage,
                    cov_mode=args.sequence_cluster_cov_mode,
                    evalue=args.sequence_cluster_evalue,
                    sensitivity=args.sequence_cluster_sensitivity,
                    cluster_mode=args.sequence_cluster_cluster_mode,
                    threads=args.sequence_cluster_threads,
                )
            )
            sequence_cluster_result = resolve_sequence_clusters(
                sequences=sequences,
                eligible_protein_ids=protein_ids_in_pairs(
                    eligible_protein_pairs
                ),
                supplied_mapping_path=args.sequence_clusters,
                parameters=automatic_parameters,
                cache_dir=args.sequence_cluster_cache_dir,
            )
            assert sequence_cluster_result is not None
            sequence_cluster_mapping = (
                sequence_cluster_result.protein_to_group
            )
            sequence_cluster_assignments = (
                sequence_cluster_result.assignments
            )
            args.sequence_cluster_metadata = (
                sequence_cluster_result.metadata
            )

    # Select the complete benchmark cohort before constructing any split.
    with performance.stage("sample_cohort"):
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
    with performance.stage("split_cohort"):
        train_df, val_df, test_df = load_or_make_split(
            protein_pairs,
            args,
            split_spec=split_spec,
            protein_to_group=sequence_cluster_mapping,
        )
        args.has_validation_split = val_df is not None and not val_df.empty
        args.evaluate_test_metrics = (
            args.eval_test_set or not args.has_validation_split)
        validate_splits(train_df=train_df, val_df=val_df, test_df=test_df)

    # Prepare to run the models
    with performance.stage("prepare_artifacts"):
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
            dataset_provenance=dataset_provenance,
            negative_construction=negative_construction,
            task_name=PPI_TASK.name,
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
                sequence_cluster_assignments=(
                    sequence_cluster_assignments),
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
    with performance.stage("baseline_models"):
        for classifier_name in baseline_classifiers:
            performance.add_model_runs(
                run_model_reruns(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    x_train=train_df,
                    x_val=val_df,
                    x_test=(
                        test_df if args.evaluate_test_metrics else None),
                    feature_name=FEATURELESS_FEATURE,
                    classifier_name=classifier_name,
                    execution_id=execution_id,
                    args=args,
                    output_paths=output_paths,
                )
            )

    # Extract the combined feature set
    if learned_classifiers:
        with performance.stage("feature_extraction"):
            if args.features == (PLM_FEATURE,):
                (
                    x_train,
                    x_val,
                    x_test,
                    feature_name,
                    encoder_metadata,
                ) = build_frozen_plm_feature_matrices(
                    cohort_df=protein_pairs,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    sequences=sequences,
                    evaluate_test_metrics=args.evaluate_test_metrics,
                    args=args,
                )
                args.protein_encoder_metadata = encoder_metadata
                try:
                    write_protein_encoder_metadata(
                        metadata=encoder_metadata,
                        output_path=(
                            output_paths.protein_encoder_metadata_path),
                        append_results=args.append_results,
                    )
                except ValueError as exc:
                    raise SystemExit(str(exc)) from None
                performance.add_observation(
                    "protein_encoder",
                    encoder_metadata,
                )
            else:
                feature_name = make_feature_name(args.features)
                x_train, x_val, x_test = build_feature_matrices(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=(
                        test_df if args.evaluate_test_metrics else None),
                    sequences=sequences,
                    feature_types=args.features,
                    args=args,
                )
            performance.add_matrices({
                "train": x_train,
                "val": x_val,
                "test": x_test,
            })

    # Run the learned classifiers
    with performance.stage("learned_models"):
        for classifier_name in learned_classifiers:
            performance.add_model_runs(
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
            )

    # Summarize and plot model performance
    with performance.stage("summarize_results"):
        summaries = summarize_model_outputs(output_paths)
    with performance.stage("plot_results"):
        heatmap_paths = plot_model_outputs(output_paths)
    log_model_outputs(summaries, output_paths, heatmap_paths)
    append_performance_report(
        performance.report(execution_id, task=PPI_TASK.name),
        output_paths.performance_path,
    )
    LOGGER.info(
        "Saved performance report to: %s",
        output_paths.performance_path,
    )


if __name__ == "__main__":
    main()
