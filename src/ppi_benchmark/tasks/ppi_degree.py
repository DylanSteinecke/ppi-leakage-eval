"""Leakage-safe training-graph degree diagnostics for PPI examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..artifact_io import write_dataframe_threadsafe
from ..evaluation import binary_classification_metrics
from ..features import FeatureIdentity
from ..matrix_provenance import canonicalize_matrix, stable_row_identities
from ..schema import EVALUATION_SCHEMA_VERSION
from ..splitting.artifacts import (
    SOURCE_ROW_INDEX_COLUMN,
    SPLIT_COLUMN,
)
from ..splitting.dispatch import TRAIN_SPLIT
from ..splitting.protocols import (
    C2_SPLIT_STRATEGY,
    C3_SPLIT_STRATEGY,
)


DEGREE_DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_DEGREE_BIN_QUANTILES = (0.5, 0.9)
TRAINING_POSITIVE_DEGREE_FILENAME = "training_positive_degree.csv"
VAL_DEGREE_METRICS_FILENAME = "val_degree_metrics.csv"
TEST_DEGREE_METRICS_FILENAME = "test_degree_metrics.csv"
VAL_DEGREE_SUMMARY_FILENAME = "val_degree_metrics_summary.csv"
TEST_DEGREE_SUMMARY_FILENAME = "test_degree_metrics_summary.csv"
PREFERENTIAL_ATTACHMENT_CLASSIFIER = "preferential_attachment"
DEGREE_MATRIX_SCHEMA_ID = "ppi.training_degree.v1"
DEGREE_FEATURE_IDENTITY = "training_degree_v1"
DEGREE_FEATURE_NAMES = (
    "degree_log_min",
    "degree_log_max",
    "degree_log_product",
)
DEGREE_TRAINING_FIT_POLICY = (
    "positive_graph_from_final_retained_canonical_training_split"
)
DEGREE_SPLIT_TRANSFORMATION_POLICIES = {
    "train": "leave_one_canonical_positive_edge_out",
    "val": "complete_positive_training_graph",
    "test": "complete_positive_training_graph",
}
POSITIVE_DEGREE_BIN_ORDER = ("unseen", "zero", "low", "mid", "high")
EXPOSURE_BIN_ORDER = ("unseen", "low", "mid", "high")
DEGREE_METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auprc",
    "auroc",
)
DEGREE_COUNT_COLUMNS = (
    "count",
    "positives",
    "negatives",
    "prevalence",
    "global_threshold",
)
DEGREE_SUMMARY_IDENTITY_COLUMNS = (
    "evaluation_schema_version",
    "degree_diagnostic_schema_version",
    "task",
    "split",
    "model_name",
    "estimator_id",
    "estimator_params",
    "configuration_id",
    "reporting_group",
    "model_role",
    "features",
    "feature_spec_sha256",
    "feature_identity",
    "fitted_extractor_sha256",
    "matrix_source",
    "matrix_schema_id",
    "pair_composition_schema_id",
    "matrix_contract_sha256",
    "row_identity_sha256",
    "matrix_sha256",
    "matrix_persisted",
    "split_strategy",
    "protocol_id",
    "protocol_version",
    "split_seed",
    "dataset_sha256",
    "positive_graph_sha256",
    "training_positive_edges_sha256",
    "training_examples_sha256",
    "split_assignments_sha256",
    "evaluation_cohort_sha256",
    "negative_construction_sha256",
    "grouping_sha256",
    "protocol_instance_sha256",
    "degree_provenance_sha256",
    "control_selection_context_sha256",
    "stratification_axis",
    "degree_measure",
    "endpoint_selector",
    "stratum",
    "stratification_applicable",
    "stratification_note",
    "endpoint_novelty",
)


@dataclass(frozen=True)
class DegreeDiagnosticContext:
    """Immutable run identity used by every degree-diagnostic model row."""

    task: str
    split_strategy: str
    protocol_id: str
    protocol_version: Any
    split_seed: int
    evaluation_cohort_hashes: Mapping[str, str]
    identity: Mapping[str, Any]

    def metric_metadata(
        self,
        *,
        split_name: str,
        model_name: str,
        estimator_id: str,
        estimator_params: str,
        configuration_id: str,
        reporting_group: str,
        model_role: str,
        feature_metadata: Mapping[str, Any],
        run_number: int,
        model_seed: float | int,
        execution_id: str,
    ) -> dict[str, Any]:
        """Return model and hash identity for one evaluated cohort."""
        try:
            cohort_hash = self.evaluation_cohort_hashes[split_name]
        except KeyError as exc:
            raise ValueError(
                f"No degree-diagnostic cohort was prepared for {split_name!r}."
            ) from exc
        return {
            "task": self.task,
            "execution_id": execution_id,
            "split": split_name,
            "model_name": model_name,
            "estimator_id": estimator_id,
            "estimator_params": estimator_params,
            "configuration_id": configuration_id,
            "reporting_group": reporting_group,
            "model_role": model_role,
            **feature_metadata,
            "run_number": run_number,
            "model_seed": model_seed,
            "split_strategy": self.split_strategy,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "split_seed": self.split_seed,
            "evaluation_cohort_sha256": cohort_hash,
            **self.identity,
        }


@dataclass(frozen=True)
class DegreeStratum:
    """One immutable mask and its reporting identity."""

    stratification_axis: str
    degree_measure: str
    endpoint_selector: str
    stratum: str
    mask: np.ndarray
    stratification_applicable: bool = True
    stratification_note: str = ""

    def metadata(self) -> dict[str, Any]:
        """Return serializable stratum metadata without its mask."""
        return {
            "stratification_axis": self.stratification_axis,
            "degree_measure": self.degree_measure,
            "endpoint_selector": self.endpoint_selector,
            "stratum": self.stratum,
            "stratification_applicable": self.stratification_applicable,
            "stratification_note": self.stratification_note,
        }


@dataclass(frozen=True)
class DegreeEvaluationPlan:
    """Cohort annotations and masks reusable across every evaluated model."""

    split_strategy: str
    targets: np.ndarray
    preferential_attachment_scores: np.ndarray
    strata: tuple[DegreeStratum, ...]
    cohort_sha256: str


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _native(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_native(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def stable_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one canonical representation."""
    encoded = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_degree_feature_specification() -> dict[str, Any]:
    """Return the complete split-independent degree feature contract."""
    return {
        "task": "ppi",
        "matrix_schema_id": DEGREE_MATRIX_SCHEMA_ID,
        "ordered_columns": list(DEGREE_FEATURE_NAMES),
        "graph": "undirected_unweighted_simple_positive_training_graph",
        "degree_transform": "log1p",
        "training_fit_policy": DEGREE_TRAINING_FIT_POLICY,
        "split_transformation_policies": dict(
            DEGREE_SPLIT_TRANSFORMATION_POLICIES),
    }


def training_degree_feature_identity() -> FeatureIdentity:
    """Return the task-owned logical identity of the degree matrix."""
    specification = training_degree_feature_specification()
    return FeatureIdentity(
        features=DEGREE_FEATURE_IDENTITY,
        feature_spec_sha256=stable_json_sha256(specification),
        feature_identity=DEGREE_FEATURE_IDENTITY,
    )


def _canonical_pair(protein_a: Any, protein_b: Any) -> tuple[str, str]:
    return tuple(sorted((str(protein_a), str(protein_b))))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _source_id_key(value: Any) -> str:
    return json.dumps(_native(value), sort_keys=True, separators=(",", ":"))


def validate_training_assignment_rows(
    train_df: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> None:
    """Require exact equality with rows explicitly assigned to training."""
    _require_columns(
        train_df,
        (SOURCE_ROW_INDEX_COLUMN,),
        "training examples",
    )
    _require_columns(
        split_assignments,
        (SOURCE_ROW_INDEX_COLUMN, SPLIT_COLUMN),
        "split assignments",
    )
    if train_df[SOURCE_ROW_INDEX_COLUMN].duplicated().any():
        raise ValueError("Training source_row_index values must be unique.")
    if split_assignments[SOURCE_ROW_INDEX_COLUMN].duplicated().any():
        raise ValueError("Split-assignment source_row_index values must be unique.")

    supplied = {
        _source_id_key(value)
        for value in train_df[SOURCE_ROW_INDEX_COLUMN]
    }
    assigned_train = {
        _source_id_key(value)
        for value in split_assignments.loc[
            split_assignments[SPLIT_COLUMN] == TRAIN_SPLIT,
            SOURCE_ROW_INDEX_COLUMN,
        ]
    }
    missing = sorted(assigned_train - supplied)
    extra = sorted(supplied - assigned_train)
    if missing or extra:
        raise ValueError(
            "Degree computation requires exact source_row_index equality "
            "with training-marked split assignments; "
            f"missing training rows={missing[:10]}, "
            f"non-training/extra rows={extra[:10]}."
        )


def split_assignments_sha256(split_assignments: pd.DataFrame) -> str:
    """Hash source-row split assignments independent of dataframe order."""
    _require_columns(
        split_assignments,
        (SOURCE_ROW_INDEX_COLUMN, SPLIT_COLUMN),
        "split assignments",
    )
    records = [
        {
            SOURCE_ROW_INDEX_COLUMN: _native(source_row_index),
            SPLIT_COLUMN: str(split_name),
        }
        for source_row_index, split_name in zip(
            split_assignments[SOURCE_ROW_INDEX_COLUMN],
            split_assignments[SPLIT_COLUMN],
        )
    ]
    records.sort(key=lambda row: _source_id_key(row[SOURCE_ROW_INDEX_COLUMN]))
    return stable_json_sha256(records)


def evaluation_cohort_sha256(examples: pd.DataFrame) -> str:
    """Hash exact evaluation identities, endpoints, and targets."""
    _require_columns(
        examples,
        ("protein_a", "protein_b", "label"),
        "evaluation examples",
    )
    _, example_ids = stable_row_identities(
        examples,
        task_example_id_columns=("example_id", "pair_id"),
    )
    records = []
    for example_id, protein_a, protein_b, label in zip(
        example_ids,
        examples["protein_a"],
        examples["protein_b"],
        examples["label"],
    ):
        canonical_a, canonical_b = _canonical_pair(protein_a, protein_b)
        records.append({
            "example_id": _native(example_id),
            "protein_a": canonical_a,
            "protein_b": canonical_b,
            "target": int(label),
        })
    records.sort(key=lambda row: _source_id_key(row["example_id"]))
    return stable_json_sha256(records)


def _quantile_cutoffs(
    values: Sequence[int],
    quantiles: tuple[float, float],
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    cutoffs = np.quantile(
        np.asarray(values, dtype=float),
        np.asarray(quantiles, dtype=float),
        method="linear",
    )
    return float(cutoffs[0]), float(cutoffs[1])


def _positive_degree_bin(
    value: int,
    seen: bool,
    cutoffs: tuple[float, float],
) -> str:
    if not seen:
        return "unseen"
    if value == 0:
        return "zero"
    if value <= cutoffs[0]:
        return "low"
    if value <= cutoffs[1]:
        return "mid"
    return "high"


def _exposure_bin(
    value: int,
    seen: bool,
    cutoffs: tuple[float, float],
) -> str:
    if not seen:
        return "unseen"
    if value <= cutoffs[0]:
        return "low"
    if value <= cutoffs[1]:
        return "mid"
    return "high"


@dataclass(frozen=True)
class TrainingDegreeProfile:
    """Training-only PPI degree maps, bins, and provenance."""

    frame: pd.DataFrame
    positive_degree: Mapping[str, int]
    training_exposure: Mapping[str, int]
    positive_edges: frozenset[tuple[str, str]]
    positive_degree_cutoffs: tuple[float, float]
    exposure_cutoffs: tuple[float, float]
    metadata: Mapping[str, Any]

    def is_seen(self, protein_id: Any) -> bool:
        return str(protein_id) in self.training_exposure

    def degree(self, protein_id: Any) -> int:
        return int(self.positive_degree.get(str(protein_id), 0))

    def exposure(self, protein_id: Any) -> int:
        return int(self.training_exposure.get(str(protein_id), 0))

    def degree_bin(self, protein_id: Any) -> str:
        protein_id = str(protein_id)
        return _positive_degree_bin(
            self.degree(protein_id),
            self.is_seen(protein_id),
            self.positive_degree_cutoffs,
        )

    def exposure_bin(self, protein_id: Any) -> str:
        protein_id = str(protein_id)
        return _exposure_bin(
            self.exposure(protein_id),
            self.is_seen(protein_id),
            self.exposure_cutoffs,
        )


def build_training_degree_profile(
    train_df: pd.DataFrame,
    split_assignments: pd.DataFrame,
    quantiles: Sequence[float] = DEFAULT_DEGREE_BIN_QUANTILES,
) -> TrainingDegreeProfile:
    """Build degree and exposure maps from validated training rows only."""
    _require_columns(
        train_df,
        (SOURCE_ROW_INDEX_COLUMN, "protein_a", "protein_b", "label"),
        "training examples",
    )
    quantile_values = tuple(float(value) for value in quantiles)
    if (
        len(quantile_values) != 2
        or not 0.0 < quantile_values[0] < quantile_values[1] < 1.0
    ):
        raise ValueError(
            "Degree bin quantiles must contain two strictly increasing "
            "values between zero and one."
        )
    if not train_df["label"].isin((0, 1)).all():
        raise ValueError("Training degree computation requires 0/1 labels.")
    validate_training_assignment_rows(train_df, split_assignments)

    exposure: dict[str, int] = {}
    positive_partners: dict[str, set[str]] = {}
    positive_edges: set[tuple[str, str]] = set()
    training_records = []
    for source_row_index, protein_a_raw, protein_b_raw, label_raw in zip(
        train_df[SOURCE_ROW_INDEX_COLUMN],
        train_df["protein_a"],
        train_df["protein_b"],
        train_df["label"],
    ):
        protein_a = str(protein_a_raw)
        protein_b = str(protein_b_raw)
        label = int(label_raw)
        if protein_a == protein_b:
            raise ValueError(
                "Training degree graph does not allow self-loop examples: "
                f"{protein_a!r}."
            )
        canonical = _canonical_pair(protein_a, protein_b)
        for protein_id in set(canonical):
            exposure[protein_id] = exposure.get(protein_id, 0) + 1
        if label == 1:
            positive_edges.add(canonical)
            positive_partners.setdefault(protein_a, set()).add(protein_b)
            positive_partners.setdefault(protein_b, set()).add(protein_a)
        training_records.append({
            SOURCE_ROW_INDEX_COLUMN: _native(source_row_index),
            "protein_a": canonical[0],
            "protein_b": canonical[1],
            "target": label,
        })

    degree = {
        protein_id: len(partners)
        for protein_id, partners in sorted(positive_partners.items())
    }
    degree_cutoffs = _quantile_cutoffs(
        [value for value in degree.values() if value > 0],
        quantile_values,
    )
    exposure_cutoffs = _quantile_cutoffs(
        list(exposure.values()),
        quantile_values,
    )
    rows = [
        {
            "degree_diagnostic_schema_version": (
                DEGREE_DIAGNOSTIC_SCHEMA_VERSION
            ),
            "protein_id": protein_id,
            "positive_degree": degree.get(protein_id, 0),
            "training_exposure": exposure[protein_id],
            "seen_in_training": True,
            "positive_degree_bin": _positive_degree_bin(
                degree.get(protein_id, 0), True, degree_cutoffs
            ),
            "training_exposure_bin": _exposure_bin(
                exposure[protein_id], True, exposure_cutoffs
            ),
        }
        for protein_id in sorted(exposure)
    ]
    frame = pd.DataFrame(rows, columns=(
        "degree_diagnostic_schema_version",
        "protein_id",
        "positive_degree",
        "training_exposure",
        "seen_in_training",
        "positive_degree_bin",
        "training_exposure_bin",
    ))
    positive_edge_records = [
        {"protein_a": protein_a, "protein_b": protein_b}
        for protein_a, protein_b in sorted(positive_edges)
    ]
    training_records.sort(
        key=lambda row: _source_id_key(row[SOURCE_ROW_INDEX_COLUMN])
    )
    positive_graph_sha256 = stable_json_sha256(positive_edge_records)
    hashes = {
        "positive_graph_sha256": positive_graph_sha256,
        # Compatibility alias for existing degree-diagnostic joins.
        "training_positive_edges_sha256": positive_graph_sha256,
        "training_examples_sha256": stable_json_sha256(training_records),
        "split_assignments_sha256": split_assignments_sha256(
            split_assignments
        ),
    }
    metadata = {
        "degree_diagnostic_schema_version": DEGREE_DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_filename": TRAINING_POSITIVE_DEGREE_FILENAME,
        "matrix": {
            "matrix_schema_id": DEGREE_MATRIX_SCHEMA_ID,
            "ordered_columns": list(DEGREE_FEATURE_NAMES),
        },
        "definitions": {
            "positive_graph": (
                "undirected, unweighted simple graph containing one edge per "
                "distinct canonical positive pair in the final retained "
                "training split"
            ),
            "positive_degree": (
                "number of distinct canonical positive training partners"
            ),
            "training_exposure": (
                "number of retained training examples containing the protein"
            ),
            "unseen": (
                "protein absent from all retained training examples; absent "
                "from the persisted training-only profile"
            ),
            "training_feature_policy": (
                "leave_one_canonical_positive_edge_out"
            ),
            "held_out_feature_policy": "complete_training_positive_graph",
        },
        "binning": {
            "quantiles": list(quantile_values),
            "quantile_method": "linear",
            "positive_degree_population": (
                "seen training proteins with positive degree greater than zero"
            ),
            "training_exposure_population": "all seen training proteins",
            "positive_degree_cutoffs": list(degree_cutoffs),
            "training_exposure_cutoffs": list(exposure_cutoffs),
            "positive_degree_bins": list(POSITIVE_DEGREE_BIN_ORDER),
            "training_exposure_bins": list(EXPOSURE_BIN_ORDER),
        },
        "n_seen_training_proteins": len(exposure),
        "n_positive_graph_proteins": len(degree),
        "n_distinct_training_positive_edges": len(positive_edges),
        "hashes": hashes,
    }
    metadata["provenance_sha256"] = stable_json_sha256(metadata)
    return TrainingDegreeProfile(
        frame=frame,
        positive_degree=degree,
        training_exposure=exposure,
        positive_edges=frozenset(positive_edges),
        positive_degree_cutoffs=degree_cutoffs,
        exposure_cutoffs=exposure_cutoffs,
        metadata=metadata,
    )


def write_training_degree_profile(
    profile: TrainingDegreeProfile,
    output_path: Path,
    append_results: bool,
) -> None:
    """Write the training-only profile or validate append compatibility."""
    if append_results and output_path.exists():
        existing = pd.read_csv(output_path)
        try:
            pd.testing.assert_frame_equal(
                existing,
                profile.frame,
                check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError(
                f"Cannot append to {output_path.parent.parent}: existing "
                "training_positive_degree.csv does not match this split."
            ) from exc
        return
    write_dataframe_threadsafe(profile.frame, output_path)


def _mapped_int_values(
    values: pd.Series,
    mapping: Mapping[str, int],
) -> np.ndarray:
    """Map protein IDs to integer training statistics with unseen values at zero."""
    return (
        values.astype(str)
        .map(mapping)
        .fillna(0)
        .to_numpy(dtype=np.int64, copy=True)
    )


def _positive_degree_bins(
    values: np.ndarray,
    seen: np.ndarray,
    cutoffs: tuple[float, float],
) -> np.ndarray:
    return np.select(
        (
            ~seen,
            values == 0,
            values <= cutoffs[0],
            values <= cutoffs[1],
        ),
        ("unseen", "zero", "low", "mid"),
        default="high",
    )


def _exposure_bins(
    values: np.ndarray,
    seen: np.ndarray,
    cutoffs: tuple[float, float],
) -> np.ndarray:
    return np.select(
        (~seen, values <= cutoffs[0], values <= cutoffs[1]),
        ("unseen", "low", "mid"),
        default="high",
    )


def degree_feature_matrix(
    examples: pd.DataFrame,
    profile: TrainingDegreeProfile,
    *,
    leave_one_positive_edge_out: bool,
) -> np.ndarray:
    """Return fixed-order symmetric positive-degree features."""
    required_columns = ["protein_a", "protein_b"]
    if leave_one_positive_edge_out:
        required_columns.append("label")
    _require_columns(examples, required_columns, "PPI examples")
    degree_a = _mapped_int_values(
        examples["protein_a"], profile.positive_degree
    )
    degree_b = _mapped_int_values(
        examples["protein_b"], profile.positive_degree
    )
    if leave_one_positive_edge_out:
        protein_a = examples["protein_a"].astype(str).to_numpy()
        protein_b = examples["protein_b"].astype(str).to_numpy()
        canonical_a = np.where(protein_a <= protein_b, protein_a, protein_b)
        canonical_b = np.where(protein_a <= protein_b, protein_b, protein_a)
        row_edges = pd.MultiIndex.from_arrays((canonical_a, canonical_b))
        held_in = np.zeros(len(examples), dtype=bool)
        if profile.positive_edges:
            positive_edges = pd.MultiIndex.from_tuples(profile.positive_edges)
            held_in = (
                examples["label"].to_numpy(dtype=int) == 1
            ) & row_edges.isin(positive_edges)
        degree_a[held_in] -= 1
        degree_b[held_in] -= 1
        if (degree_a < 0).any() or (degree_b < 0).any():
            raise RuntimeError("Leave-one-edge-out degree became negative.")

    log_degree_a = np.log1p(degree_a.astype(np.float64, copy=False))
    log_degree_b = np.log1p(degree_b.astype(np.float64, copy=False))
    return canonicalize_matrix(np.column_stack((
        np.minimum(log_degree_a, log_degree_b),
        np.maximum(log_degree_a, log_degree_b),
        log_degree_a * log_degree_b,
    )).astype(np.float64, copy=False))


def degree_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    profile: TrainingDegreeProfile,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build LOO training and complete-graph held-out feature matrices."""
    return (
        degree_feature_matrix(
            train_df,
            profile,
            leave_one_positive_edge_out=True,
        ),
        None if val_df is None else degree_feature_matrix(
            val_df,
            profile,
            leave_one_positive_edge_out=False,
        ),
        None if test_df is None else degree_feature_matrix(
            test_df,
            profile,
            leave_one_positive_edge_out=False,
        ),
    )


def preferential_attachment_scores(
    examples: pd.DataFrame,
    profile: TrainingDegreeProfile,
) -> np.ndarray:
    """Return log1p of the training-positive degree product."""
    degree_a = _mapped_int_values(
        examples["protein_a"], profile.positive_degree
    )
    degree_b = _mapped_int_values(
        examples["protein_b"], profile.positive_degree
    )
    return np.log1p(degree_a * degree_b)


def degree_identity_context(
    profile: TrainingDegreeProfile,
    *,
    dataset_sha256: str,
    negative_construction: Any,
    grouping_identity: Any,
    protocol_instance: Any,
    control_selection_context: Any,
) -> dict[str, Any]:
    """Return hash gates shared by model and control degree artifacts."""
    hashes = profile.metadata["hashes"]
    return {
        "dataset_sha256": str(dataset_sha256),
        "positive_graph_sha256": hashes["positive_graph_sha256"],
        "training_positive_edges_sha256": hashes[
            "training_positive_edges_sha256"
        ],
        "training_examples_sha256": hashes["training_examples_sha256"],
        "split_assignments_sha256": hashes["split_assignments_sha256"],
        "negative_construction_sha256": stable_json_sha256(
            negative_construction
            if negative_construction is not None
            else {"negative_construction": "unspecified"}
        ),
        "grouping_sha256": stable_json_sha256(grouping_identity),
        "protocol_instance_sha256": stable_json_sha256(protocol_instance),
        "degree_provenance_sha256": profile.metadata["provenance_sha256"],
        "control_selection_context_sha256": stable_json_sha256(
            control_selection_context
        ),
    }


def _endpoint_annotations(
    examples: pd.DataFrame,
    profile: TrainingDegreeProfile,
) -> pd.DataFrame:
    _require_columns(
        examples,
        ("protein_a", "protein_b", "label"),
        "evaluation examples",
    )
    protein_a = examples["protein_a"].astype(str)
    protein_b = examples["protein_b"].astype(str)
    seen_a = protein_a.isin(profile.training_exposure).to_numpy(dtype=bool)
    seen_b = protein_b.isin(profile.training_exposure).to_numpy(dtype=bool)
    degree_a = _mapped_int_values(protein_a, profile.positive_degree)
    degree_b = _mapped_int_values(protein_b, profile.positive_degree)
    exposure_a = _mapped_int_values(protein_a, profile.training_exposure)
    exposure_b = _mapped_int_values(protein_b, profile.training_exposure)
    degree_bin_a = _positive_degree_bins(
        degree_a, seen_a, profile.positive_degree_cutoffs
    )
    degree_bin_b = _positive_degree_bins(
        degree_b, seen_b, profile.positive_degree_cutoffs
    )
    exposure_bin_a = _exposure_bins(
        exposure_a, seen_a, profile.exposure_cutoffs
    )
    exposure_bin_b = _exposure_bins(
        exposure_b, seen_b, profile.exposure_cutoffs
    )

    def ordered_bins(
        left: np.ndarray,
        right: np.ndarray,
        order: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_rank = pd.Categorical(left, categories=order, ordered=True).codes
        right_rank = pd.Categorical(right, categories=order, ordered=True).codes
        left_is_least = left_rank <= right_rank
        least = np.where(left_is_least, left, right)
        greatest = np.where(left_is_least, right, left)
        pair = np.char.add(np.char.add(least, "|"), greatest)
        return least, greatest, pair

    degree_least, degree_greatest, degree_pair = ordered_bins(
        degree_bin_a, degree_bin_b, POSITIVE_DEGREE_BIN_ORDER
    )
    exposure_least, exposure_greatest, exposure_pair = ordered_bins(
        exposure_bin_a, exposure_bin_b, EXPOSURE_BIN_ORDER
    )
    one_seen = seen_a ^ seen_b
    novelty = np.select(
        (seen_a & seen_b, one_seen),
        ("both_seen", "one_seen"),
        default="neither_seen",
    )
    familiar_degree = np.where(
        seen_a & ~seen_b,
        degree_bin_a,
        np.where(seen_b & ~seen_a, degree_bin_b, "not_applicable"),
    )
    familiar_exposure = np.where(
        seen_a & ~seen_b,
        exposure_bin_a,
        np.where(seen_b & ~seen_a, exposure_bin_b, "not_applicable"),
    )
    return pd.DataFrame({
        "positive_degree_least_bin": degree_least,
        "positive_degree_greatest_bin": degree_greatest,
        "positive_degree_pair_bin": degree_pair,
        "training_exposure_least_bin": exposure_least,
        "training_exposure_greatest_bin": exposure_greatest,
        "training_exposure_pair_bin": exposure_pair,
        "positive_degree_familiar_bin": familiar_degree,
        "training_exposure_familiar_bin": familiar_exposure,
        "endpoint_novelty": novelty,
    }, index=examples.index)


def _observed_strata(
    annotations: pd.DataFrame,
    column: str,
    axis: str,
    measure: str,
    selector: str,
    note: str = "",
) -> list[DegreeStratum]:
    values = annotations[column].astype(str).to_numpy()
    strata = []
    for value in sorted(np.unique(values)):
        mask = values == value
        mask.setflags(write=False)
        strata.append(DegreeStratum(
            stratification_axis=axis,
            degree_measure=measure,
            endpoint_selector=selector,
            stratum=value,
            mask=mask,
            stratification_note=note,
        ))
    return strata


def _degree_strata(
    examples: pd.DataFrame,
    profile: TrainingDegreeProfile,
    split_strategy: str,
) -> tuple[DegreeStratum, ...]:
    annotations = _endpoint_annotations(examples, profile)
    global_note = ""
    global_applicable = True
    if split_strategy == C3_SPLIT_STRATEGY:
        if not (annotations["endpoint_novelty"] == "neither_seen").all():
            raise RuntimeError(
                "C3 degree sentinel failed: every held-out endpoint must be "
                "unseen in training."
            )
        global_note = (
            "C3: degree-stratification not applicable "
            "(training degree = 0 by construction)"
        )
        global_applicable = False
    global_mask = np.ones(len(examples), dtype=bool)
    global_mask.setflags(write=False)
    rows = [DegreeStratum(
        stratification_axis="global",
        degree_measure="global",
        endpoint_selector="global",
        stratum="all",
        mask=global_mask,
        stratification_applicable=global_applicable,
        stratification_note=global_note,
    )]
    if split_strategy == C3_SPLIT_STRATEGY:
        return tuple(rows)
    if split_strategy == C2_SPLIT_STRATEGY:
        if not (annotations["endpoint_novelty"] == "one_seen").all():
            raise RuntimeError(
                "C2 degree sentinel failed: every held-out pair must contain "
                "exactly one training-seen endpoint."
            )
        rows.extend(_observed_strata(
            annotations,
            "positive_degree_familiar_bin",
            "positive_degree_familiar",
            "positive_degree",
            "familiar_max_endpoint",
            "C2 minimum endpoint degree is structurally zero",
        ))
        rows.extend(_observed_strata(
            annotations,
            "training_exposure_familiar_bin",
            "training_exposure_familiar",
            "training_exposure",
            "familiar_max_endpoint",
            "C2 minimum endpoint degree is structurally zero",
        ))
        return tuple(rows)

    for measure, prefix in (
        ("positive_degree", "positive_degree"),
        ("training_exposure", "training_exposure"),
    ):
        for selector in ("least", "greatest", "pair"):
            rows.extend(_observed_strata(
                annotations,
                f"{prefix}_{selector}_bin",
                f"{measure}_{selector}",
                measure,
                selector,
            ))
    rows.extend(_observed_strata(
        annotations,
        "endpoint_novelty",
        "endpoint_novelty",
        "endpoint_novelty",
        "novelty_pattern",
    ))
    return tuple(rows)


def build_degree_evaluation_plan(
    examples: pd.DataFrame,
    profile: TrainingDegreeProfile,
    split_strategy: str,
) -> DegreeEvaluationPlan:
    """Prepare cohort degree annotations once for reuse by every model."""
    targets = examples["label"].to_numpy(dtype=int)
    scores = preferential_attachment_scores(examples, profile)
    targets.setflags(write=False)
    scores.setflags(write=False)
    return DegreeEvaluationPlan(
        split_strategy=split_strategy,
        targets=targets,
        preferential_attachment_scores=scores,
        strata=_degree_strata(examples, profile, split_strategy),
        cohort_sha256=evaluation_cohort_sha256(examples),
    )


def _endpoint_novelty_descriptor(
    split_strategy: str,
    stratum: DegreeStratum,
) -> str:
    if stratum.stratification_axis == "endpoint_novelty":
        return stratum.stratum
    if split_strategy == "c1":
        return "both_seen_by_protocol"
    if split_strategy == C2_SPLIT_STRATEGY:
        return "one_seen_by_protocol"
    if split_strategy == C3_SPLIT_STRATEGY:
        return "neither_seen_by_protocol"
    return "not_stratified"


def _threshold_free_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    metrics = binary_classification_metrics(
        targets,
        scores,
        np.zeros(len(targets), dtype=int),
    )
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        metrics[metric_name] = np.nan
    return metrics


def assert_c2_preferential_attachment_invariants(
    targets: np.ndarray,
    scores: np.ndarray,
    metrics: Mapping[str, float],
) -> None:
    """Fail if a C2 PA score does not collapse to its algebraic constant."""
    if len(scores) and not np.allclose(scores, scores[0]):
        raise RuntimeError(
            "C2 preferential-attachment scores must be constant because one "
            "endpoint has training degree zero."
        )
    prevalence = float(np.mean(targets))
    if np.any(targets == 1) and not np.isclose(metrics["auprc"], prevalence):
        raise RuntimeError(
            "C2 preferential-attachment AUPRC must equal prevalence."
        )
    if len(np.unique(targets)) == 2 and not np.isclose(metrics["auroc"], 0.5):
        raise RuntimeError("C2 preferential-attachment AUROC must equal 0.5.")


def assert_c3_degree_control_invariants(
    targets: np.ndarray,
    scores: np.ndarray,
    metrics: Mapping[str, float],
) -> None:
    """Fail unless a C3 training-degree control is constant at chance rank."""
    if len(scores) and not np.allclose(scores, scores[0]):
        raise RuntimeError(
            "C3 degree-control scores must be constant because both endpoints "
            "have training degree zero."
        )
    prevalence = float(np.mean(targets))
    if np.any(targets == 1) and not np.isclose(metrics["auprc"], prevalence):
        raise RuntimeError("C3 degree-control AUPRC must equal prevalence.")
    if len(np.unique(targets)) == 2 and not np.isclose(metrics["auroc"], 0.5):
        raise RuntimeError("C3 degree-control AUROC must equal 0.5.")


def degree_metric_rows(
    plan: DegreeEvaluationPlan,
    *,
    scores: Sequence[float],
    predictions: Sequence[int] | None,
    global_threshold: float | None,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Evaluate one model on global and protocol-appropriate degree strata."""
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    if len(score_array) != len(plan.targets):
        raise ValueError("Degree diagnostic scores do not match evaluation rows.")
    prediction_array = None
    if predictions is not None:
        prediction_array = np.asarray(predictions, dtype=int).reshape(-1)
        if len(prediction_array) != len(plan.targets):
            raise ValueError(
                "Degree diagnostic predictions do not match evaluation rows."
            )
    output_rows = []
    for stratum in plan.strata:
        targets = plan.targets[stratum.mask]
        stratum_scores = score_array[stratum.mask]
        if prediction_array is None:
            metrics = _threshold_free_metrics(targets, stratum_scores)
        else:
            metrics = binary_classification_metrics(
                targets,
                stratum_scores,
                prediction_array[stratum.mask],
            )
        if (
            plan.split_strategy == C2_SPLIT_STRATEGY
            and metadata.get("model_name")
            == PREFERENTIAL_ATTACHMENT_CLASSIFIER
        ):
            assert_c2_preferential_attachment_invariants(
                targets,
                stratum_scores,
                metrics,
            )
        if (
            plan.split_strategy == C3_SPLIT_STRATEGY
            and metadata.get("model_name") in {
                PREFERENTIAL_ATTACHMENT_CLASSIFIER,
                "degree_logistic",
                "degree_hgb",
            }
        ):
            assert_c3_degree_control_invariants(
                targets,
                stratum_scores,
                metrics,
            )
        positives = int(targets.sum())
        count = len(targets)
        output_rows.append({
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "degree_diagnostic_schema_version": (
                DEGREE_DIAGNOSTIC_SCHEMA_VERSION
            ),
            **metadata,
            **stratum.metadata(),
            "endpoint_novelty": _endpoint_novelty_descriptor(
                plan.split_strategy,
                stratum,
            ),
            "count": count,
            "positives": positives,
            "negatives": count - positives,
            "prevalence": float(positives / count),
            "global_threshold": (
                np.nan if global_threshold is None else float(global_threshold)
            ),
            **metrics,
        })
    return pd.DataFrame(output_rows)


def deduplicate_preferential_attachment_rows(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse execution-level copies of the deterministic PA reference."""
    if metrics.empty or "model_name" not in metrics.columns:
        return metrics
    pa_mask = metrics["model_name"] == PREFERENTIAL_ATTACHMENT_CLASSIFIER
    if not pa_mask.any():
        return metrics
    ignored = {"execution_id", "run_dir"}
    pa = metrics.loc[pa_mask].copy()
    subset = [column for column in pa.columns if column not in ignored]
    pa = pa.drop_duplicates(subset=subset)
    return pd.concat((metrics.loc[~pa_mask], pa), ignore_index=True)


def summarize_degree_metrics(metrics_path: Path) -> pd.DataFrame:
    """Summarize model reruns without pooling different artifact identities."""
    metrics = deduplicate_preferential_attachment_rows(pd.read_csv(metrics_path))
    value_columns = [
        column
        for column in (*DEGREE_COUNT_COLUMNS, *DEGREE_METRIC_COLUMNS)
        if column in metrics.columns
    ]
    group_columns = [
        column
        for column in DEGREE_SUMMARY_IDENTITY_COLUMNS
        if column in metrics.columns
    ]
    grouped = metrics.groupby(group_columns, dropna=False)
    counts = grouped.size().rename("n_runs").reset_index()
    means = grouped[value_columns].mean().add_suffix("_mean").reset_index()
    errors = (
        grouped[value_columns]
        .sem(ddof=1)
        .add_suffix("_standard_error")
        .reset_index()
    )
    return (
        counts.merge(means, on=group_columns)
        .merge(errors, on=group_columns)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
