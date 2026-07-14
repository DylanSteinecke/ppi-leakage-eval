"""Task-neutral fitting and evaluation over model backends."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ..backends.base import BackendFitResult, ModelBackend, SupervisedSplit
from ..evaluation import EvaluationPolicy
from ..tasks.base import TaskAdapter


@dataclass(frozen=True)
class SplitEvaluation:
    """Predictions, metrics, and timing for one supervised split."""

    name: str
    targets: np.ndarray
    scores: np.ndarray
    predictions: np.ndarray
    metrics: Mapping[str, float]
    evaluation_seconds: float


@dataclass(frozen=True)
class SupervisedRunResult:
    """Framework- and task-neutral output from one fitted backend."""

    fit_result: BackendFitResult
    operating_point: Any
    split_evaluations: Mapping[str, SplitEvaluation]
    operating_point_selection_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class TaskSplitData:
    """Task examples paired with backend inputs for one named split."""

    name: str
    examples: Any
    inputs: Any


def fit_and_evaluate_backend(
        backend: ModelBackend,
        train: SupervisedSplit,
        validation: SupervisedSplit | None,
        test: SupervisedSplit | None,
        evaluation_policy: EvaluationPolicy,
    ) -> SupervisedRunResult:
    """Fit once, select on validation, and evaluate requested splits."""
    requested_splits = tuple(
        split
        for split in (train, validation, test)
        if split is not None
    )
    split_names = [split.name for split in requested_splits]
    if len(set(split_names)) != len(split_names):
        raise ValueError("Supervised split names must be unique.")

    run_started_at = perf_counter()
    fit_result = backend.fit(train=train, validation=validation)

    selection_split = validation if validation is not None else train
    selection_started_at = perf_counter()
    selection_prediction = backend.predict(selection_split.inputs)
    operating_point = evaluation_policy.select(
        targets=selection_split.targets,
        prediction=selection_prediction,
        has_validation=validation is not None,
    )
    selection_seconds = perf_counter() - selection_started_at
    prediction_cache = {selection_split.name: selection_prediction}

    split_evaluations: dict[str, SplitEvaluation] = {}
    for split in requested_splits:
        evaluation_started_at = perf_counter()
        prediction = prediction_cache.get(split.name)
        if prediction is None:
            prediction = backend.predict(split.inputs)
        final_predictions, metrics = evaluation_policy.evaluate(
            targets=split.targets,
            prediction=prediction,
            operating_point=operating_point,
        )
        split_evaluations[split.name] = SplitEvaluation(
            name=split.name,
            targets=np.asarray(split.targets).reshape(-1),
            scores=np.asarray(prediction.scores).reshape(-1),
            predictions=np.asarray(final_predictions).reshape(-1),
            metrics=dict(metrics),
            evaluation_seconds=float(
                perf_counter() - evaluation_started_at
            ),
        )

    return SupervisedRunResult(
        fit_result=fit_result,
        operating_point=operating_point,
        split_evaluations=split_evaluations,
        operating_point_selection_seconds=float(selection_seconds),
        total_seconds=float(perf_counter() - run_started_at),
    )


def fit_and_evaluate_task(
        backend: ModelBackend, task: TaskAdapter,
        train: TaskSplitData,
        validation: TaskSplitData | None,
        test: TaskSplitData | None,
        evaluation_policy: EvaluationPolicy,
    ) -> SupervisedRunResult:
    """Connect task-owned examples to the framework-neutral run engine."""
    return fit_and_evaluate_backend(
        backend=backend,
        train=task.make_split(train.name, train.examples, train.inputs),
        validation=(
            None
            if validation is None
            else task.make_split(
                validation.name,
                validation.examples,
                validation.inputs,
            )
        ),
        test=(
            None
            if test is None
            else task.make_split(test.name, test.examples, test.inputs)
        ),
        evaluation_policy=evaluation_policy,
    )
