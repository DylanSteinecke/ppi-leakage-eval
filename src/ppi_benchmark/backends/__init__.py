"""Model backend construction and public contracts."""

from pathlib import Path
from typing import Any, Mapping

from ..models import TORCH_CLASSIFIER_CHOICES
from .base import (
    BackendFitResult,
    BackendPrediction,
    ModelBackend,
    SupervisedSplit,
)
from .sklearn import SklearnBackend


DEFAULT_BACKEND = "sklearn"
TORCH_BACKEND = "torch"
BACKEND_CHOICES = (DEFAULT_BACKEND, TORCH_BACKEND)


def backend_name_for_classifier(classifier_name: str) -> str:
    """Return the framework backend that owns a classifier."""
    if classifier_name in TORCH_CLASSIFIER_CHOICES:
        return TORCH_BACKEND
    return DEFAULT_BACKEND


def make_model_backend(
        classifier_name: str, max_iter: int, random_state: int,
        backend_name: str | None = None,
        best_checkpoint_path: str | Path | None = None,
        last_checkpoint_path: str | Path | None = None,
        resume_from: str | Path | None = None,
        backend_options: Mapping[str, Any] | None = None,
        task_name: str = "binary_classification",
        task_schema_version: int = 1,
    ) -> ModelBackend:
    """Construct a model backend without coupling the runner to a framework."""
    resolved_backend = (
        backend_name or backend_name_for_classifier(classifier_name)
    )
    options = dict(backend_options or {})
    if resolved_backend == DEFAULT_BACKEND:
        if classifier_name in TORCH_CLASSIFIER_CHOICES:
            raise ValueError(
                f"Classifier {classifier_name!r} requires the torch backend."
            )
        if options:
            raise ValueError("sklearn backend does not accept backend options.")
        return SklearnBackend.from_classifier(
            classifier_name=classifier_name,
            max_iter=max_iter,
            random_state=random_state,
        )
    if resolved_backend == TORCH_BACKEND:
        if classifier_name != "torch_mlp":
            raise ValueError(
                "torch backend currently supports only classifier "
                "'torch_mlp'."
            )
        try:
            from .torch_mlp import TorchMLPBackend, TorchMLPConfig
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "torch_mlp requires PyTorch. Install the project with "
                    "`python -m pip install -e '.[torch]'`."
                ) from exc
            raise
        config = TorchMLPConfig(max_epochs=max_iter, **options)
        return TorchMLPBackend(
            config=config,
            random_state=random_state,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            resume_from=resume_from,
            task_name=task_name,
            task_schema_version=task_schema_version,
        )
    raise ValueError(f"Unknown model backend: {resolved_backend}")


__all__ = [
    "BACKEND_CHOICES",
    "DEFAULT_BACKEND",
    "TORCH_BACKEND",
    "BackendFitResult",
    "BackendPrediction",
    "ModelBackend",
    "SklearnBackend",
    "SupervisedSplit",
    "backend_name_for_classifier",
    "make_model_backend",
]
