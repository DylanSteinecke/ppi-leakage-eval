"""Validation-only decision-threshold selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


FIXED_THRESHOLD = "fixed"
VALIDATION_F1_THRESHOLD = "validation_f1"
THRESHOLD_SELECTION_CHOICES = (
    VALIDATION_F1_THRESHOLD,
    FIXED_THRESHOLD,
)


@dataclass(frozen=True)
class ThresholdSelection:
    """Auditable result of selecting one binary decision threshold."""

    threshold: float
    default_threshold: float
    strategy: str
    metric_name: str | None
    metric_value: float | None
    n_candidates: int


def predictions_at_threshold(
        scores: np.ndarray, threshold: float,
    ) -> np.ndarray:
    """Convert model scores to hard 0/1 predictions."""
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    return (score_array >= threshold).astype(int)


def metric_at_threshold(
        labels: np.ndarray, scores: np.ndarray, threshold: float,
    ) -> float:
    """Return validation F1 at one decision threshold."""
    label_array = np.asarray(labels).reshape(-1)
    predictions = predictions_at_threshold(scores, threshold)
    return float(f1_score(label_array, predictions, zero_division=0))


def select_validation_f1_threshold(
        labels: np.ndarray, scores: np.ndarray, default_threshold: float,
    ) -> ThresholdSelection:
    """Select the validation-F1 maximum with deterministic tie-breaking.

    Ties prefer the threshold closest to the backend default, then the higher
    (more conservative) threshold. Candidate evaluation is O(n log n), not
    O(n squared), even when every validation score is unique.
    """
    label_array = np.asarray(labels).reshape(-1)
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    if len(label_array) != len(score_array):
        raise ValueError("Validation labels and scores must have equal length.")
    if not len(label_array):
        raise ValueError("Validation labels and scores must not be empty.")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("Validation threshold selection requires 0/1 labels.")
    if not np.isfinite(score_array).all():
        raise ValueError("Validation threshold scores must be finite.")
    if not np.isfinite(default_threshold):
        raise ValueError("The default decision threshold must be finite.")

    candidates = np.unique(np.concatenate((
        score_array,
        np.asarray([default_threshold]),
    )))

    descending_order = np.argsort(-score_array, kind="stable")
    sorted_scores = score_array[descending_order]
    sorted_labels = label_array[descending_order].astype(np.int64)
    cumulative_true_positives = np.cumsum(sorted_labels)
    selected_counts = np.searchsorted(
        -sorted_scores,
        -candidates,
        side="right",
    )
    true_positives = np.zeros(len(candidates), dtype=float)
    has_selected = selected_counts > 0
    true_positives[has_selected] = cumulative_true_positives[
        selected_counts[has_selected] - 1
    ]
    false_positives = selected_counts - true_positives
    false_negatives = sorted_labels.sum() - true_positives
    denominators = (
        2.0 * true_positives + false_positives + false_negatives
    )
    f1_values = np.divide(
        2.0 * true_positives,
        denominators,
        out=np.zeros_like(true_positives),
        where=denominators > 0.0,
    )
    best_f1 = float(f1_values.max())
    tied = np.flatnonzero(np.isclose(
        f1_values,
        best_f1,
        rtol=1e-12,
        atol=1e-12,
    ))
    distances = np.abs(candidates[tied] - default_threshold)
    closest = tied[np.isclose(
        distances,
        distances.min(),
        rtol=1e-12,
        atol=1e-12,
    )]
    selected_index = int(closest[np.argmax(candidates[closest])])
    return ThresholdSelection(
        threshold=float(candidates[selected_index]),
        default_threshold=float(default_threshold),
        strategy=VALIDATION_F1_THRESHOLD,
        metric_name="f1",
        metric_value=best_f1,
        n_candidates=len(candidates),
    )


def fixed_threshold_selection(
        default_threshold: float, labels: np.ndarray | None = None,
        scores: np.ndarray | None = None, strategy: str = FIXED_THRESHOLD,
    ) -> ThresholdSelection:
    """Return a fixed backend-default threshold with optional validation F1."""
    metric_value = None
    if labels is not None and scores is not None:
        metric_value = metric_at_threshold(labels, scores, default_threshold)
    return ThresholdSelection(
        threshold=float(default_threshold),
        default_threshold=float(default_threshold),
        strategy=strategy,
        metric_name="f1" if metric_value is not None else None,
        metric_value=metric_value,
        n_candidates=1,
    )
