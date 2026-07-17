"""Benchmark-level aggregation for PPI degree diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..tasks.ppi_degree import (
    DEGREE_COUNT_COLUMNS,
    DEGREE_METRIC_COLUMNS,
    PREFERENTIAL_ATTACHMENT_CLASSIFIER,
    TEST_DEGREE_METRICS_FILENAME,
    VAL_DEGREE_METRICS_FILENAME,
    deduplicate_preferential_attachment_rows,
    stable_json_sha256,
)
from ..splitting.artifacts import write_metadata_json


BENCHMARK_DEGREE_SUMMARY_FILENAME = "benchmark_degree_summary.csv"
BENCHMARK_DEGREE_LIFT_FILENAME = "benchmark_degree_lift.csv"
BENCHMARK_DEGREE_CONTROL_SELECTION_FILENAME = (
    "benchmark_degree_control_selection.json"
)
DEGREE_CONTROL_SELECTION_SCHEMA_VERSION = 1
DEFAULT_PRIMARY_DEGREE_CONTROL = "degree_logistic"
SENSITIVITY_DEGREE_CONTROL = "degree_hgb"
CONTROL_SELECTION_CONTEXT_COLUMN = "control_selection_context_sha256"
CONTROL_SELECTION_SPLIT_SEEDS = frozenset(range(5))
CONTROL_SELECTION_MIN_MEDIAN_DELTA = 0.02
CONTROL_SELECTION_MIN_WINS = 4
LIFT_GATE_COLUMNS = (
    "task",
    "split",
    "split_strategy",
    "protocol_id",
    "protocol_version",
    "split_seed",
    "dataset_sha256",
    "training_positive_edges_sha256",
    "training_examples_sha256",
    "split_assignments_sha256",
    "evaluation_cohort_sha256",
    "negative_construction_sha256",
    "grouping_sha256",
    "protocol_instance_sha256",
    "degree_provenance_sha256",
    CONTROL_SELECTION_CONTEXT_COLUMN,
)
NOMINAL_JOIN_COLUMNS = (
    "task",
    "split",
    "split_strategy",
    "protocol_id",
    "protocol_version",
    "split_seed",
    "dataset_sha256",
)
SUMMARY_IDENTITY_COLUMNS = (
    "task",
    "split",
    "model_name",
    "classifier",
    "model_role",
    "selected_fitted_control",
    "features",
    "split_strategy",
    "protocol_id",
    "protocol_version",
    "dataset_sha256",
    "negative_construction_sha256",
    "grouping_sha256",
    CONTROL_SELECTION_CONTEXT_COLUMN,
    "stratification_axis",
    "degree_measure",
    "endpoint_selector",
    "stratum",
    "stratification_applicable",
    "stratification_note",
    "endpoint_novelty",
)
EMPTY_LIFT_COLUMNS = (
    "model_name",
    "selected_fitted_control",
    "model_auprc",
    "degree_control_auprc",
    "pa_auprc",
    "residual_over_degree_control",
    "absolute_lift",
    "metric_ratio",
    "absolute_lift_vs_pa",
    "metric_ratio_vs_pa",
    "fitting_gain",
)


def read_degree_metrics(
    run_dirs: Iterable[Path],
    *,
    splits: Iterable[str] = ("val", "test"),
) -> pd.DataFrame:
    """Read additive degree artifacts, tolerating legacy runs without them."""
    requested_splits = frozenset(splits)
    unknown = requested_splits - {"val", "test"}
    if unknown:
        raise ValueError(f"Unknown degree metric splits: {sorted(unknown)}")
    frames = []
    for run_dir in run_dirs:
        for expected_split, filename in (
            ("val", VAL_DEGREE_METRICS_FILENAME),
            ("test", TEST_DEGREE_METRICS_FILENAME),
        ):
            if expected_split not in requested_splits:
                continue
            path = run_dir / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            if "split" not in frame.columns:
                raise ValueError(f"Degree metrics lack split identity: {path}")
            if not (frame["split"] == expected_split).all():
                raise ValueError(
                    f"Degree metrics contain the wrong split for {path}."
                )
            frame.insert(0, "run_dir", str(run_dir))
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    required = {
        "degree_diagnostic_schema_version",
        "classifier",
        "model_name",
        "model_role",
        "auprc",
        *LIFT_GATE_COLUMNS,
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(
            "Degree metric artifacts are missing required identity columns: "
            f"{sorted(missing)}"
        )
    return metrics


def _plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _control_selection_evidence(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    evidence = metrics[
        (metrics["split"] == "val")
        & (metrics["stratification_axis"] == "global")
        & metrics["classifier"].isin((
            DEFAULT_PRIMARY_DEGREE_CONTROL,
            SENSITIVITY_DEGREE_CONTROL,
        ))
        & metrics["split_strategy"].isin(("random", "c1", "c2"))
    ].copy()
    if evidence.empty:
        return evidence
    if CONTROL_SELECTION_CONTEXT_COLUMN not in evidence.columns:
        evidence[CONTROL_SELECTION_CONTEXT_COLUMN] = "single_context"
    group_columns = [
        CONTROL_SELECTION_CONTEXT_COLUMN,
        "split_strategy",
        *[
            column
            for column in (
                "protocol_id",
                "protocol_version",
                "grouping_sha256",
            )
            if column in evidence.columns
        ],
        "split_seed",
        "classifier",
    ]
    return (
        evidence.groupby(group_columns, dropna=False)["auprc"]
        .mean()
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def _select_context_control(
    context_sha256: str,
    evidence: pd.DataFrame,
) -> dict[str, Any]:
    protocol_evidence = []
    promote = False
    protocol_context_columns = [
        column
        for column in ("protocol_id", "protocol_version", "grouping_sha256")
        if column in evidence.columns
    ]
    for protocol in ("random", "c1", "c2"):
        protocol_rows = evidence[evidence["split_strategy"] == protocol]
        grouped_contexts = (
            [((), protocol_rows)]
            if not protocol_context_columns or protocol_rows.empty
            else protocol_rows.groupby(protocol_context_columns, dropna=False)
        )
        for context_key, context_rows in grouped_contexts:
            if protocol_context_columns and not isinstance(context_key, tuple):
                context_key = (context_key,)
            context = {
                column: _plain_value(value)
                for column, value in zip(protocol_context_columns, context_key)
            }
            by_seed = (
                context_rows.groupby(
                    ["split_seed", "classifier"], dropna=False
                )["auprc"]
                .mean()
                .unstack("classifier")
            )
            required_seeds = sorted(CONTROL_SELECTION_SPLIT_SEEDS)
            complete = (
                set(by_seed.index) >= CONTROL_SELECTION_SPLIT_SEEDS
                and DEFAULT_PRIMARY_DEGREE_CONTROL in by_seed.columns
                and SENSITIVITY_DEGREE_CONTROL in by_seed.columns
                and by_seed.loc[
                    required_seeds,
                    [
                        DEFAULT_PRIMARY_DEGREE_CONTROL,
                        SENSITIVITY_DEGREE_CONTROL,
                    ],
                ].notna().all().all()
            )
            if complete:
                selected = by_seed.loc[required_seeds]
                deltas = (
                    selected[SENSITIVITY_DEGREE_CONTROL]
                    - selected[DEFAULT_PRIMARY_DEGREE_CONTROL]
                )
                median_delta = float(deltas.median())
                wins = int((deltas > 0.0).sum())
                protocol_promotes = (
                    median_delta > CONTROL_SELECTION_MIN_MEDIAN_DELTA
                    and not np.isclose(
                        median_delta,
                        CONTROL_SELECTION_MIN_MEDIAN_DELTA,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    and wins >= CONTROL_SELECTION_MIN_WINS
                )
                promote = promote or protocol_promotes
            else:
                median_delta = None
                wins = 0
            protocol_evidence.append({
                "split_strategy": protocol,
                **context,
                "complete": bool(complete),
                "median_delta": median_delta,
                "hgb_wins": wins,
            })
    return {
        CONTROL_SELECTION_CONTEXT_COLUMN: context_sha256,
        "selected_control": (
            SENSITIVITY_DEGREE_CONTROL
            if promote
            else DEFAULT_PRIMARY_DEGREE_CONTROL
        ),
        "protocol_evidence": protocol_evidence,
    }


def select_primary_degree_control(metrics: pd.DataFrame) -> dict[str, Any]:
    """Apply and record the predeclared validation-only control decision."""
    evidence = _control_selection_evidence(metrics)
    selections = []
    if not evidence.empty:
        for context_sha256, context_rows in evidence.groupby(
            CONTROL_SELECTION_CONTEXT_COLUMN,
            dropna=False,
        ):
            selections.append(_select_context_control(
                str(context_sha256),
                context_rows,
            ))
    selected_controls = {
        selection["selected_control"] for selection in selections
    }
    selected_control = (
        next(iter(selected_controls))
        if len(selected_controls) == 1
        else (
            "context_specific"
            if selected_controls
            else DEFAULT_PRIMARY_DEGREE_CONTROL
        )
    )
    evidence_records = (
        evidence.astype(object)
        .where(pd.notna(evidence), None)
        .to_dict(orient="records")
    )
    return {
        "degree_control_selection_schema_version": (
            DEGREE_CONTROL_SELECTION_SCHEMA_VERSION
        ),
        "selected_control": selected_control,
        "default_on_tie_or_missing": DEFAULT_PRIMARY_DEGREE_CONTROL,
        "required_split_seeds": sorted(CONTROL_SELECTION_SPLIT_SEEDS),
        "median_delta_threshold": CONTROL_SELECTION_MIN_MEDIAN_DELTA,
        "minimum_hgb_wins": CONTROL_SELECTION_MIN_WINS,
        "selection_evidence_sha256": stable_json_sha256(evidence_records),
        "selections": selections,
    }


def load_or_create_control_selection(
    candidate: dict[str, Any],
    selection_path: Path,
) -> dict[str, Any]:
    """Create an immutable selection lock or validate its exact evidence."""
    if selection_path.exists():
        with selection_path.open("r", encoding="utf-8") as input_file:
            locked = json.load(input_file)
        if locked != candidate:
            raise ValueError(
                "Refusing to change the locked fitted degree control at "
                f"{selection_path}. Validation evidence or selection context "
                "changed after the control was locked."
            )
        return locked
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    write_metadata_json(candidate, selection_path)
    return candidate


def selected_control_for_context(
    selection: Mapping[str, Any],
    context_sha256: Any,
) -> str:
    """Resolve the immutable selected control for one benchmark context."""
    context_key = str(context_sha256)
    for context_selection in selection.get("selections", []):
        if str(context_selection[CONTROL_SELECTION_CONTEXT_COLUMN]) == context_key:
            return str(context_selection["selected_control"])
    return str(selection["default_on_tie_or_missing"])


def summarize_benchmark_degree_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Report means and standard errors across split seeds."""
    if metrics.empty:
        return pd.DataFrame(columns=(
            *SUMMARY_IDENTITY_COLUMNS,
            "n_split_seeds",
            "n_model_runs",
            "auprc_mean",
            "auprc_standard_error",
        ))
    metrics = deduplicate_preferential_attachment_rows(metrics)
    identity = [
        column for column in SUMMARY_IDENTITY_COLUMNS
        if column in metrics.columns
    ]
    values = [
        column
        for column in (*DEGREE_COUNT_COLUMNS, *DEGREE_METRIC_COLUMNS)
        if column in metrics.columns
    ]
    per_seed_groups = [*identity, "split_seed"]
    per_seed = (
        metrics.groupby(per_seed_groups, dropna=False)[values]
        .mean()
        .reset_index()
    )
    grouped = per_seed.groupby(identity, dropna=False)
    seed_counts = grouped["split_seed"].nunique().rename(
        "n_split_seeds"
    ).reset_index()
    means = grouped[values].mean().add_suffix("_mean").reset_index()
    errors = (
        grouped[values]
        .sem(ddof=1)
        .add_suffix("_standard_error")
        .reset_index()
    )
    run_counts = (
        metrics.groupby(identity, dropna=False)
        .size()
        .rename("n_model_runs")
        .reset_index()
    )
    return (
        seed_counts.merge(run_counts, on=identity)
        .merge(means, on=identity)
        .merge(errors, on=identity)
        .sort_values(identity)
        .reset_index(drop=True)
    )


def _same_gate(row: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    mask = np.ones(len(candidates), dtype=bool)
    for column in LIFT_GATE_COLUMNS:
        value = row[column]
        if pd.isna(value):
            mask &= candidates[column].isna().to_numpy()
        else:
            mask &= (candidates[column] == value).to_numpy()
    return pd.Series(mask, index=candidates.index)


def _same_nominal_context(row: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    mask = np.ones(len(candidates), dtype=bool)
    for column in NOMINAL_JOIN_COLUMNS:
        value = row[column]
        if pd.isna(value):
            mask &= candidates[column].isna().to_numpy()
        else:
            mask &= (candidates[column] == value).to_numpy()
    return pd.Series(mask, index=candidates.index)


def _matching_reference(
    row: pd.Series,
    references: pd.DataFrame,
    reference_name: str,
) -> pd.DataFrame:
    nominal = references.loc[_same_nominal_context(row, references)]
    if nominal.empty:
        raise ValueError(
            f"Refusing degree lift join for model {row['model_name']!r}: "
            f"required {reference_name} rows are missing for the model's "
            "evaluation context."
        )
    matched = nominal.loc[_same_gate(row, nominal)]
    if matched.empty:
        mismatched = [
            column
            for column in LIFT_GATE_COLUMNS
            if not nominal[column].eq(row[column]).all()
        ]
        raise ValueError(
            f"Refusing degree lift join for model {row['model_name']!r}: "
            f"{reference_name} has mismatched identity gates "
            f"{mismatched}."
        )
    return matched


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return float(numerator / denominator)


def build_degree_lift(
    metrics: pd.DataFrame,
    selection: Mapping[str, Any] | str,
) -> pd.DataFrame:
    """Join real models to PA and the fitted control under strict hash gates."""
    if metrics.empty:
        return pd.DataFrame(columns=EMPTY_LIFT_COLUMNS)
    global_rows = metrics[
        metrics["stratification_axis"] == "global"
    ].copy()
    models = global_rows[global_rows["model_role"] == "predictive_model"]
    controls = global_rows[global_rows["classifier"].isin((
        DEFAULT_PRIMARY_DEGREE_CONTROL,
        SENSITIVITY_DEGREE_CONTROL,
    ))]
    pa_rows = global_rows[
        global_rows["classifier"] == PREFERENTIAL_ATTACHMENT_CLASSIFIER
    ]
    output = []
    for _, model_row in models.iterrows():
        if isinstance(selection, str):
            selected_control = selection
        else:
            selected_control = selected_control_for_context(
                selection,
                model_row.get(CONTROL_SELECTION_CONTEXT_COLUMN, "single_context"),
            )
        matched_control = _matching_reference(
            model_row,
            controls[controls["classifier"] == selected_control],
            selected_control,
        )
        matched_pa = _matching_reference(
            model_row,
            pa_rows,
            PREFERENTIAL_ATTACHMENT_CLASSIFIER,
        )
        control_auprc = float(matched_control["auprc"].mean())
        pa_values = matched_pa["auprc"].dropna().to_numpy(dtype=float)
        if len(pa_values) > 1 and not np.allclose(
            pa_values,
            pa_values[0],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Refusing degree lift join: preferential-attachment AUPRC "
                "differs across otherwise identical hash gates."
            )
        pa_auprc = float(matched_pa["auprc"].mean())
        model_auprc = float(model_row["auprc"])
        absolute_lift = model_auprc - control_auprc
        row = {
            key: model_row[key]
            for key in (
                "run_dir",
                "execution_id",
                "task",
                "split",
                "model_name",
                "classifier",
                "features",
                "run_number",
                "model_seed",
                "split_strategy",
                "protocol_id",
                "protocol_version",
                "split_seed",
                *LIFT_GATE_COLUMNS[6:],
            )
            if key in model_row.index
        }
        row.update({
            "selected_fitted_control": selected_control,
            "model_auprc": model_auprc,
            "degree_control_auprc": control_auprc,
            "pa_auprc": pa_auprc,
            "residual_over_degree_control": absolute_lift,
            "absolute_lift": absolute_lift,
            "metric_ratio": _safe_ratio(model_auprc, control_auprc),
            "absolute_lift_vs_pa": model_auprc - pa_auprc,
            "metric_ratio_vs_pa": _safe_ratio(model_auprc, pa_auprc),
            "fitting_gain": control_auprc - pa_auprc,
        })
        output.append(row)
    if not output:
        return pd.DataFrame(columns=EMPTY_LIFT_COLUMNS)
    return pd.DataFrame(output)


def aggregate_degree_diagnostics(
    run_dirs: Iterable[Path],
    summary_out: Path,
    lift_out: Path,
    selection_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Write benchmark degree summary/lift artifacts and selection evidence."""
    run_dirs = tuple(run_dirs)
    validation_metrics = read_degree_metrics(run_dirs, splits=("val",))
    candidate = select_primary_degree_control(validation_metrics)
    selection = load_or_create_control_selection(candidate, selection_out)
    metrics = read_degree_metrics(run_dirs)
    if not metrics.empty:
        metrics["selected_fitted_control"] = [
            selected_control_for_context(selection, context_sha256)
            for context_sha256 in metrics[CONTROL_SELECTION_CONTEXT_COLUMN]
        ]
    summary = summarize_benchmark_degree_metrics(metrics)
    lift = build_degree_lift(metrics, selection)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    lift_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    lift.to_csv(lift_out, index=False)
    return summary, lift, selection


def protocol_lift_table(lift: pd.DataFrame) -> pd.DataFrame:
    """Return a compact protocol-level table for CLI logging."""
    if lift.empty:
        return pd.DataFrame()
    group_columns = [
        "split",
        "split_strategy",
        "model_name",
        "selected_fitted_control",
    ]
    values = [
        "pa_auprc",
        "degree_control_auprc",
        "model_auprc",
        "residual_over_degree_control",
        "metric_ratio",
        "fitting_gain",
    ]
    return (
        lift.groupby(group_columns, dropna=False)[values]
        .mean()
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
