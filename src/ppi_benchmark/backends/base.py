"""Backend-neutral model training and prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SupervisedSplit:
    """Inputs and targets for one named supervised-learning split."""

    name: str
    inputs: Any
    targets: Any


@dataclass(frozen=True)
class BackendFitResult:
    """Framework-neutral observations produced by model fitting."""

    fit_seconds: float
    iteration_report: dict[str, Any] | None = None
    training_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendPrediction:
    """Task scores plus optional conventional backend decisions."""

    scores: Any
    predictions: Any | None = None
    default_threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    """Contract shared by sklearn and future deep-learning backends."""

    backend_name: str

    def fit(
            self, train: SupervisedSplit,
            validation: SupervisedSplit | None = None,
        ) -> BackendFitResult:
        """Fit from training data, optionally using validation data."""
        ...

    def predict(self, inputs: Any) -> BackendPrediction:
        """Return task scores; task evaluation owns final decisions."""
        ...
