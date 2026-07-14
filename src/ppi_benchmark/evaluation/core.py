"""Task-neutral evaluation policies and standard binary metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .thresholds import (
    FIXED_THRESHOLD,
    VALIDATION_F1_THRESHOLD,
    ThresholdSelection,
    fixed_threshold_selection,
    predictions_at_threshold,
    select_validation_f1_threshold,
)

if TYPE_CHECKING:
    from ..backends.base import BackendPrediction


def binary_classification_metrics(
        targets: Any, scores: Any, predictions: Any,
    ) -> dict[str, float]:
    """Return robust binary metrics without failing on one-class splits."""
    target_array = np.asarray(targets).reshape(-1)
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    prediction_array = np.asarray(predictions).reshape(-1)
    if not (
        len(target_array) == len(score_array) == len(prediction_array)
    ):
        raise ValueError(
            "targets, scores, and predictions must have the same length."
        )
    if not len(target_array):
        raise ValueError("Evaluation arrays must not be empty.")
    if not np.isin(target_array, (0, 1)).all():
        raise ValueError("Binary evaluation requires 0/1 targets.")
    if not np.isin(prediction_array, (0, 1)).all():
        raise ValueError("Binary evaluation requires 0/1 predictions.")
    if not np.isfinite(score_array).all():
        raise ValueError("Evaluation scores must be finite.")

    has_positive_target = bool(np.any(target_array == 1))
    has_both_target_classes = len(np.unique(target_array)) == 2
    has_predicted_positive = bool(np.any(prediction_array == 1))
    return {
        "accuracy": float(accuracy_score(target_array, prediction_array)),
        "precision": (
            float(precision_score(
                target_array,
                prediction_array,
                zero_division=0,
            ))
            if has_predicted_positive
            else np.nan
        ),
        "recall": float(recall_score(
            target_array,
            prediction_array,
            zero_division=0,
        )),
        "f1": float(f1_score(
            target_array,
            prediction_array,
            zero_division=0,
        )),
        "auprc": (
            float(average_precision_score(target_array, score_array))
            if has_positive_target
            else np.nan
        ),
        "auroc": (
            float(roc_auc_score(target_array, score_array))
            if has_both_target_classes
            else np.nan
        ),
    }


@runtime_checkable
class EvaluationPolicy(Protocol):
    """Task-owned operating-point selection and metric computation."""

    def select(
            self, targets: Any, prediction: BackendPrediction,
            has_validation: bool,
        ) -> Any:
        """Select an operating point using train or validation predictions."""
        ...

    def evaluate(
            self, targets: Any, prediction: BackendPrediction,
            operating_point: Any,
        ) -> tuple[np.ndarray, Mapping[str, float]]:
        """Return final predictions and standardized metrics."""
        ...


@dataclass(frozen=True)
class BinaryClassificationPolicy:
    """Binary evaluation with optional validation-selected F1 threshold."""

    threshold_strategy: str = VALIDATION_F1_THRESHOLD
    force_fixed: bool = False
    fixed_strategy: str = "fixed_baseline"

    def __post_init__(self) -> None:
        if self.threshold_strategy not in {
            VALIDATION_F1_THRESHOLD,
            FIXED_THRESHOLD,
        }:
            raise ValueError(
                f"Unknown threshold strategy: {self.threshold_strategy}"
            )

    def select(
            self, targets: Any, prediction: BackendPrediction,
            has_validation: bool,
        ) -> ThresholdSelection:
        """Select one threshold without consulting held-out test targets."""
        if prediction.default_threshold is None:
            raise ValueError(
                "Binary evaluation requires a backend default threshold."
            )
        targets_for_fixed = targets if has_validation else None
        scores_for_fixed = prediction.scores if has_validation else None
        if self.force_fixed:
            return fixed_threshold_selection(
                default_threshold=prediction.default_threshold,
                labels=targets_for_fixed,
                scores=scores_for_fixed,
                strategy=self.fixed_strategy,
            )
        if (
            self.threshold_strategy == VALIDATION_F1_THRESHOLD
            and has_validation
        ):
            return select_validation_f1_threshold(
                labels=np.asarray(targets),
                scores=prediction.scores,
                default_threshold=prediction.default_threshold,
            )
        fixed_strategy = (
            FIXED_THRESHOLD
            if self.threshold_strategy == FIXED_THRESHOLD
            else "fixed_no_validation"
        )
        return fixed_threshold_selection(
            default_threshold=prediction.default_threshold,
            labels=targets_for_fixed,
            scores=scores_for_fixed,
            strategy=fixed_strategy,
        )

    def evaluate(
            self, targets: Any, prediction: BackendPrediction,
            operating_point: ThresholdSelection,
        ) -> tuple[np.ndarray, Mapping[str, float]]:
        """Apply the selected threshold and calculate binary metrics."""
        if prediction.default_threshold is None:
            raise ValueError(
                "Binary evaluation requires a backend default threshold."
            )
        if not np.isclose(
            prediction.default_threshold,
            operating_point.default_threshold,
        ):
            raise RuntimeError(
                "Model backend returned inconsistent default thresholds."
            )
        predictions = predictions_at_threshold(
            prediction.scores,
            operating_point.threshold,
        )
        metrics = binary_classification_metrics(
            targets=targets,
            scores=prediction.scores,
            predictions=predictions,
        )
        return predictions, metrics
