"""Public interfaces for task-neutral evaluation and threshold selection."""

from .core import (
    BinaryClassificationPolicy,
    EvaluationPolicy,
    binary_classification_metrics,
)
from .thresholds import (
    FIXED_THRESHOLD,
    THRESHOLD_SELECTION_CHOICES,
    VALIDATION_F1_THRESHOLD,
    ThresholdSelection,
    fixed_threshold_selection,
    metric_at_threshold,
    predictions_at_threshold,
    select_validation_f1_threshold,
)

__all__ = [
    "FIXED_THRESHOLD",
    "THRESHOLD_SELECTION_CHOICES",
    "VALIDATION_F1_THRESHOLD",
    "BinaryClassificationPolicy",
    "EvaluationPolicy",
    "ThresholdSelection",
    "binary_classification_metrics",
    "fixed_threshold_selection",
    "metric_at_threshold",
    "predictions_at_threshold",
    "select_validation_f1_threshold",
]
