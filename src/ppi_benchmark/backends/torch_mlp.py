"""Mini-batched Torch MLP using the shared task-neutral trainer."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from scipy import sparse
import torch
from torch import nn

from ..evaluation import binary_classification_metrics
from ..training.checkpoints import (
    TORCH_CHECKPOINT_FORMAT_VERSION,
    TorchCheckpointManager,
)
from ..training.torch_trainer import (
    TorchStepOutput,
    TorchTaskRuntime,
    TorchTrainer,
    TorchTrainerConfig,
)
from ..torch_utils import TORCH_DEVICE_CHOICES, resolve_torch_device
from .base import BackendFitResult, BackendPrediction, SupervisedSplit


TORCH_VALIDATION_MONITORS = ("auprc", "loss")


@dataclass(frozen=True)
class TorchMLPConfig:
    """Training and architecture configuration for the small Torch MLP."""

    max_epochs: int
    batch_size: int = 256
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 5
    min_delta: float = 1e-4
    device: str = "auto"
    precision: str = "float32"
    validation_monitor: str = "auprc"

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be at least 0 and less than 1.")
        if self.device not in TORCH_DEVICE_CHOICES:
            raise ValueError(
                "device must be one of: "
                f"{', '.join(TORCH_DEVICE_CHOICES)}."
            )
        if self.validation_monitor not in TORCH_VALIDATION_MONITORS:
            raise ValueError(
                "validation_monitor must be one of: "
                f"{', '.join(TORCH_VALIDATION_MONITORS)}."
            )
        self.trainer_config()

    def trainer_config(self) -> TorchTrainerConfig:
        """Return only framework-owned trainer settings."""
        return TorchTrainerConfig(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            min_delta=self.min_delta,
            precision=self.precision,
            validation_monitor=self.validation_monitor,
            validation_monitor_mode=(
                "min" if self.validation_monitor == "loss" else "max"
            ),
        )


class _BinaryMLP(nn.Module):
    """One-hidden-layer head retained for the current matrix benchmark."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(1)


@dataclass(frozen=True)
class _PreparedMatrixSplit:
    name: str
    inputs: Any
    targets: np.ndarray | None


class _MatrixBinaryTask(TorchTaskRuntime):
    """Task connector for sparse/dense binary feature matrices."""

    def __init__(
            self, *, input_dim: int, hidden_dim: int, dropout: float,
            train_targets: np.ndarray, device: torch.device,
            task_name: str, task_schema_version: int,
        ):
        self.name = task_name
        self.schema_version = task_schema_version
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.module = _BinaryMLP(input_dim, hidden_dim, dropout).to(device)
        n_positive = float(train_targets.sum())
        n_negative = float(len(train_targets) - n_positive)
        positive_weight = (
            n_negative / n_positive
            if n_positive > 0.0 and n_negative > 0.0
            else 1.0
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, device=device),
        )
        self.max_dense_batch_rows = 0

    def n_examples(self, prepared_split: _PreparedMatrixSplit) -> int:
        return int(prepared_split.inputs.shape[0])

    def iter_batches(
            self, prepared_split: _PreparedMatrixSplit, batch_size: int,
            row_order: np.ndarray,
        ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        for start in range(0, len(row_order), batch_size):
            row_indices = row_order[start:start + batch_size]
            matrix_batch = prepared_split.inputs[row_indices]
            if sparse.issparse(matrix_batch):
                dense_batch = matrix_batch.toarray()
            else:
                dense_batch = np.asarray(matrix_batch)
            dense_batch = np.asarray(
                dense_batch,
                dtype=np.float32,
                order="C",
            )
            self.max_dense_batch_rows = max(
                self.max_dense_batch_rows,
                dense_batch.shape[0],
            )
            input_tensor = torch.from_numpy(dense_batch).to(self.device)
            target_tensor = (
                None
                if prepared_split.targets is None
                else torch.from_numpy(
                    prepared_split.targets[row_indices]
                ).to(self.device)
            )
            yield input_tensor, target_tensor

    def step(
            self, batch: tuple[torch.Tensor, torch.Tensor | None],
            training: bool,
        ) -> TorchStepOutput:
        del training
        inputs, targets = batch
        logits = self.module(inputs)
        loss = None if targets is None else self.criterion(logits, targets)
        return TorchStepOutput(
            scores=torch.sigmoid(logits),
            targets=targets,
            loss=loss,
            n_observations=inputs.shape[0],
        )

    def metric_values(
            self, targets: np.ndarray, scores: np.ndarray,
        ) -> Mapping[str, float]:
        predictions = (np.asarray(scores).reshape(-1) >= 0.5).astype(int)
        return binary_classification_metrics(
            targets=np.asarray(targets).reshape(-1),
            scores=np.asarray(scores).reshape(-1),
            predictions=predictions,
        )

    @staticmethod
    def _update_digest_with_array(
            digest: Any, array: np.ndarray,
        ) -> None:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(memoryview(contiguous).cast("B"))

    @classmethod
    def _matrix_signature(cls, inputs: Any) -> dict[str, Any]:
        digest = hashlib.sha256()
        if sparse.issparse(inputs):
            matrix = inputs.tocsr(copy=False)
            matrix_format = "csr"
            digest.update(matrix_format.encode("utf-8"))
            cls._update_digest_with_array(digest, matrix.indptr)
            cls._update_digest_with_array(digest, matrix.indices)
            cls._update_digest_with_array(digest, matrix.data)
            dtype = matrix.dtype.str
        else:
            matrix = np.asarray(inputs)
            matrix_format = "dense"
            digest.update(matrix_format.encode("utf-8"))
            cls._update_digest_with_array(digest, matrix)
            dtype = matrix.dtype.str
        return {
            "format": matrix_format,
            "shape": list(matrix.shape),
            "dtype": dtype,
            "sha256": digest.hexdigest(),
        }

    @classmethod
    def _split_signature(
            cls, split: _PreparedMatrixSplit | None,
        ) -> dict[str, Any] | None:
        if split is None:
            return None
        target_sha256 = None
        if split.targets is not None:
            digest = hashlib.sha256()
            cls._update_digest_with_array(digest, split.targets)
            target_sha256 = digest.hexdigest()
        return {
            "inputs": cls._matrix_signature(split.inputs),
            "targets_sha256": target_sha256,
        }

    def data_signature(
            self, train: _PreparedMatrixSplit,
            validation: _PreparedMatrixSplit | None,
        ) -> Mapping[str, Any]:
        train_signature = self._split_signature(train)
        validation_signature = self._split_signature(validation)
        if train_signature is None:
            raise RuntimeError("Training signature is missing.")
        return {
            "train_inputs": train_signature["inputs"],
            "train_targets_sha256": train_signature["targets_sha256"],
            "validation_inputs": (
                None
                if validation_signature is None
                else validation_signature["inputs"]
            ),
            "validation_targets_sha256": (
                None
                if validation_signature is None
                else validation_signature["targets_sha256"]
            ),
        }

    def checkpoint_identity(self) -> Mapping[str, Any]:
        return {
            "head": "binary_mlp",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    def checkpoint_components(self) -> Mapping[str, nn.Module]:
        return {"task_head": self.module}

    def runtime_state(self) -> Mapping[str, Any]:
        return {"max_dense_batch_rows": self.max_dense_batch_rows}

    def load_runtime_state(self, state: Mapping[str, Any]) -> None:
        self.max_dense_batch_rows = int(
            state.get("max_dense_batch_rows", 0)
        )

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {
            "max_dense_batch_rows": self.max_dense_batch_rows,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }


class TorchMLPBackend:
    """Matrix compatibility adapter over the shared Torch trainer."""

    backend_name = "torch"
    CHECKPOINT_FORMAT_VERSION = TORCH_CHECKPOINT_FORMAT_VERSION
    _RESUME_CONFIG_FIELDS = (
        "batch_size",
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "dropout",
        "patience",
        "min_delta",
        "precision",
        "validation_monitor",
    )
    _RESUME_CONFIG_DEFAULTS = {
        "precision": "float32",
        "validation_monitor": "auprc",
    }

    def __init__(
            self, config: TorchMLPConfig, random_state: int,
            best_checkpoint_path: str | Path | None = None,
            last_checkpoint_path: str | Path | None = None,
            resume_from: str | Path | None = None,
            task_name: str = "binary_classification",
            task_schema_version: int = 1,
        ):
        self.config = config
        self.random_state = random_state
        self.task_name = task_name
        self.task_schema_version = task_schema_version
        self.device = resolve_torch_device(config.device)
        self.checkpoints = TorchCheckpointManager(
            best_path=best_checkpoint_path,
            last_path=last_checkpoint_path,
            resume_from=resume_from,
        )
        self.input_dim: int | None = None
        self._task: _MatrixBinaryTask | None = None
        self._trainer: TorchTrainer | None = None

    @staticmethod
    def _prepare_inputs(inputs: Any, split_name: str) -> Any:
        if not hasattr(inputs, "shape") or len(inputs.shape) != 2:
            raise ValueError(
                f"{split_name} inputs must be a two-dimensional matrix."
            )
        if inputs.shape[0] < 1 or inputs.shape[1] < 1:
            raise ValueError(f"{split_name} inputs must not be empty.")
        if sparse.issparse(inputs):
            return inputs.tocsr(copy=False)
        return np.asarray(inputs)

    @staticmethod
    def _prepare_targets(
            targets: Any, expected_rows: int, split_name: str,
        ) -> np.ndarray:
        target_array = np.asarray(targets, dtype=np.float32).reshape(-1)
        if len(target_array) != expected_rows:
            raise ValueError(
                f"{split_name} inputs and targets have different lengths."
            )
        if not np.isin(target_array, (0.0, 1.0)).all():
            raise ValueError("torch_mlp requires binary 0/1 targets.")
        return target_array

    def _prepare_split(self, split: SupervisedSplit) -> _PreparedMatrixSplit:
        inputs = self._prepare_inputs(split.inputs, split.name)
        targets = self._prepare_targets(
            split.targets,
            inputs.shape[0],
            split.name,
        )
        return _PreparedMatrixSplit(split.name, inputs, targets)

    def fit(
            self, train: SupervisedSplit,
            validation: SupervisedSplit | None = None,
        ) -> BackendFitResult:
        """Train through the reusable engine and restore its best epoch."""
        train_data = self._prepare_split(train)
        validation_data = (
            None if validation is None else self._prepare_split(validation)
        )
        if (
            validation_data is not None
            and validation_data.inputs.shape[1] != train_data.inputs.shape[1]
        ):
            raise ValueError(
                "Training and validation feature dimensions differ."
            )
        self.input_dim = int(train_data.inputs.shape[1])
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._task = _MatrixBinaryTask(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            train_targets=train_data.targets,
            device=self.device,
            task_name=self.task_name,
            task_schema_version=self.task_schema_version,
        )
        self._trainer = TorchTrainer(
            task=self._task,
            config=self.config.trainer_config(),
            device=self.device,
            random_state=self.random_state,
            checkpoint_manager=self.checkpoints,
            backend_name=self.backend_name,
            model_name="torch_mlp",
            checkpoint_config=asdict(self.config),
            compatible_config_fields=self._RESUME_CONFIG_FIELDS,
            config_defaults=self._RESUME_CONFIG_DEFAULTS,
        )
        result = self._trainer.fit(train_data, validation_data)
        return BackendFitResult(
            fit_seconds=result.fit_seconds,
            training_history=result.history,
            metadata=result.metadata,
        )

    def predict(self, inputs: Any) -> BackendPrediction:
        """Predict in batches while densifying only the active matrix slice."""
        if self._task is None or self._trainer is None or self.input_dim is None:
            raise RuntimeError("torch_mlp must be fit before prediction.")
        prediction_inputs = self._prepare_inputs(inputs, "prediction")
        if prediction_inputs.shape[1] != self.input_dim:
            raise ValueError(
                "Prediction feature dimension differs from training."
            )
        prepared = _PreparedMatrixSplit(
            name="prediction",
            inputs=prediction_inputs,
            targets=None,
        )
        scores = self._trainer.predict(prepared).reshape(-1)
        return BackendPrediction(
            scores=scores,
            default_threshold=0.5,
        )
