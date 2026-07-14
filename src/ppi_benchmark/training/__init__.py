"""Public interfaces for task-neutral training and evaluation."""

from .runner import (
    SplitEvaluation,
    SupervisedRunResult,
    TaskSplitData,
    fit_and_evaluate_backend,
    fit_and_evaluate_task,
)

__all__ = [
    "SplitEvaluation",
    "SupervisedRunResult",
    "TaskSplitData",
    "fit_and_evaluate_backend",
    "fit_and_evaluate_task",
]
