"""Mini-batched PyTorch MLP over precomputed pair-feature matrices."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score
import torch
from torch import nn

from ..datasets.common import file_sha256
from ..torch_utils import resolve_torch_device
from .base import BackendFitResult, BackendPrediction, SupervisedSplit


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

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and greater than 0.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be at least 0 and less than 1.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta must be finite and nonnegative.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(
                "device must be one of: auto, cpu, cuda, mps."
            )


class _BinaryMLP(nn.Module):
    """One-hidden-layer binary classifier used by ``TorchMLPBackend``."""

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


class TorchMLPBackend:
    """Train and evaluate a small MLP without densifying a full sparse split."""

    backend_name = "torch"
    CHECKPOINT_FORMAT_VERSION = 3
    _RESUME_CONFIG_FIELDS = (
        "batch_size",
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "dropout",
        "patience",
        "min_delta",
    )

    def __init__(
            self, config: TorchMLPConfig, random_state: int,
            best_checkpoint_path: str | Path | None = None,
            last_checkpoint_path: str | Path | None = None,
            resume_from: str | Path | None = None,
        ):
        self.config = config
        self.random_state = random_state
        self.best_checkpoint_path = (
            None
            if best_checkpoint_path is None
            else Path(best_checkpoint_path)
        )
        self.last_checkpoint_path = (
            None
            if last_checkpoint_path is None
            else Path(last_checkpoint_path)
        )
        self.resume_from = (
            None if resume_from is None else Path(resume_from)
        )
        self.device = resolve_torch_device(config.device)
        self.model: _BinaryMLP | None = None
        self.input_dim: int | None = None
        self.data_signature: dict[str, Any] | None = None
        self.max_dense_batch_rows = 0
        self._memory_best_checkpoint: dict[str, Any] | None = None
        self._memory_last_checkpoint: dict[str, Any] | None = None

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

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

    def _batch_tensors(
            self, inputs: Any, targets: np.ndarray | None,
            row_indices: np.ndarray,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        matrix_batch = inputs[row_indices]
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
        if targets is None:
            target_tensor = None
        else:
            target_tensor = torch.from_numpy(
                targets[row_indices],
            ).to(self.device)
        return input_tensor, target_tensor

    def _iter_batches(
            self, inputs: Any, targets: np.ndarray | None,
            row_order: np.ndarray,
        ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        for start in range(0, len(row_order), self.config.batch_size):
            row_indices = row_order[start:start + self.config.batch_size]
            yield self._batch_tensors(inputs, targets, row_indices)

    def _validation_summary(
            self, inputs: Any, targets: np.ndarray,
            criterion: nn.Module,
        ) -> tuple[float, float]:
        if self.model is None:
            raise RuntimeError("torch_mlp has not been initialized.")
        total_loss = 0.0
        n_observations = 0
        score_batches = []
        self.model.eval()
        row_order = np.arange(inputs.shape[0])
        with torch.inference_mode():
            for input_batch, target_batch in self._iter_batches(
                    inputs, targets, row_order):
                if target_batch is None:
                    raise RuntimeError("Validation targets are missing.")
                logits = self.model(input_batch)
                loss = criterion(logits, target_batch)
                batch_rows = input_batch.shape[0]
                total_loss += float(loss.item()) * batch_rows
                n_observations += batch_rows
                score_batches.append(torch.sigmoid(logits).cpu().numpy())
        scores = np.concatenate(score_batches)
        validation_auprc = float(average_precision_score(targets, scores))
        return total_loss / n_observations, validation_auprc

    def _model_state_dict(self) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("torch_mlp has not been initialized.")
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

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
        """Hash a matrix without densifying sparse feature inputs."""
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
    def _data_signature(
            cls, train_inputs: Any, train_targets: np.ndarray,
            validation_inputs: Any | None,
            validation_targets: np.ndarray | None,
        ) -> dict[str, Any]:
        target_digest = hashlib.sha256()
        cls._update_digest_with_array(target_digest, train_targets)
        signature = {
            "train_inputs": cls._matrix_signature(train_inputs),
            "train_targets_sha256": target_digest.hexdigest(),
            "validation_inputs": None,
            "validation_targets_sha256": None,
        }
        if validation_inputs is not None and validation_targets is not None:
            validation_target_digest = hashlib.sha256()
            cls._update_digest_with_array(
                validation_target_digest,
                validation_targets,
            )
            signature.update({
                "validation_inputs": cls._matrix_signature(
                    validation_inputs),
                "validation_targets_sha256": (
                    validation_target_digest.hexdigest()),
            })
        return signature

    @staticmethod
    def _save_payload(
            payload: dict[str, Any], path: Path | None,
        ) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(
            f".{path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _load_payload(path: Path, device: torch.device) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError(f"Resume checkpoint does not exist: {path}")
        payload = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid Torch checkpoint payload: {path}")
        return payload

    def _base_checkpoint_payload(self) -> dict[str, Any]:
        if self.input_dim is None or self.data_signature is None:
            raise RuntimeError("torch_mlp has not been initialized.")
        return {
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "backend_name": self.backend_name,
            "classifier_name": "torch_mlp",
            "input_dim": self.input_dim,
            "config": asdict(self.config),
            "random_state": self.random_state,
            "data_signature": self.data_signature,
        }

    def _save_best_checkpoint(
            self, best_state_dict: dict[str, torch.Tensor], epoch: int,
            monitor_metric: str, monitor_value: float,
        ) -> None:
        payload = {
            **self._base_checkpoint_payload(),
            "checkpoint_kind": "best",
            "model_state_dict": best_state_dict,
            "epoch": epoch,
            "monitor_metric": monitor_metric,
            "monitor_value": monitor_value,
        }
        self._memory_best_checkpoint = payload
        self._save_payload(payload, self.best_checkpoint_path)

    def _rng_state(self) -> dict[str, Any]:
        state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
            "torch_mps": None,
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        if self.device.type == "mps":
            state["torch_mps"] = torch.mps.get_rng_state()
        return state

    def _restore_rng_state(
            self, checkpoint: dict[str, Any],
            random_generator: np.random.Generator,
        ) -> None:
        random_generator.bit_generator.state = checkpoint[
            "numpy_rng_state"
        ]
        rng_state = checkpoint["torch_rng_state"]
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if self.device.type == "cuda" and rng_state["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        if self.device.type == "mps" and rng_state["torch_mps"] is not None:
            torch.mps.set_rng_state(rng_state["torch_mps"])

    def _save_last_checkpoint(
            self, optimizer: torch.optim.Optimizer,
            random_generator: np.random.Generator,
            history: list[dict[str, Any]], best_epoch: int,
            best_monitor_value: float,
            best_validation_loss: float | None,
            best_state_dict: dict[str, torch.Tensor],
            epochs_without_improvement: int, optimizer_steps: int,
            stopped_early: bool, monitor_metric: str,
        ) -> None:
        payload = {
            **self._base_checkpoint_payload(),
            "checkpoint_kind": "last",
            "model_state_dict": self._model_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(history[-1]["epoch"]),
            "history": history,
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor_value,
            "best_validation_loss": best_validation_loss,
            "best_model_state_dict": best_state_dict,
            "epochs_without_improvement": epochs_without_improvement,
            "optimizer_steps": optimizer_steps,
            "stopped_early": stopped_early,
            "monitor_metric": monitor_metric,
            "numpy_rng_state": random_generator.bit_generator.state,
            "torch_rng_state": self._rng_state(),
            "max_dense_batch_rows": self.max_dense_batch_rows,
        }
        self._memory_last_checkpoint = payload
        self._save_payload(payload, self.last_checkpoint_path)

    def _validate_resume_checkpoint(
            self, checkpoint: dict[str, Any], input_dim: int,
        ) -> None:
        if checkpoint.get("checkpoint_kind") != "last":
            raise ValueError(
                "--torch-resume-from must point to a last checkpoint."
            )
        if checkpoint.get("format_version") != self.CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported Torch checkpoint format version: "
                f"{checkpoint.get('format_version')!r}; expected "
                f"{self.CHECKPOINT_FORMAT_VERSION}."
            )
        if checkpoint.get("classifier_name") != "torch_mlp":
            raise ValueError("Resume checkpoint is not for torch_mlp.")
        if int(checkpoint.get("input_dim", -1)) != input_dim:
            raise ValueError(
                "Resume checkpoint input dimension does not match features."
            )
        if int(checkpoint.get("random_state", -1)) != self.random_state:
            raise ValueError(
                "Resume checkpoint random_state does not match model seed."
            )
        if checkpoint.get("data_signature") != self.data_signature:
            raise ValueError(
                "Resume checkpoint training/validation data do not match "
                "the current split and feature matrices."
            )
        old_config = checkpoint.get("config", {})
        new_config = asdict(self.config)
        changed_fields = [
            field_name
            for field_name in self._RESUME_CONFIG_FIELDS
            if old_config.get(field_name) != new_config[field_name]
        ]
        if changed_fields:
            raise ValueError(
                "Resume checkpoint training configuration differs for: "
                f"{changed_fields}"
            )
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        if checkpoint_epoch > self.config.max_epochs:
            raise ValueError(
                "torch_max_epochs must be at least the checkpoint epoch."
            )

    def fit(
            self, train: SupervisedSplit,
            validation: SupervisedSplit | None = None,
        ) -> BackendFitResult:
        """Train in sparse-sliced batches and restore the best-AUPRC epoch."""
        self._synchronize_device()
        fit_started_at = perf_counter()
        train_inputs = self._prepare_inputs(train.inputs, train.name)
        train_targets = self._prepare_targets(
            train.targets,
            train_inputs.shape[0],
            train.name,
        )
        validation_inputs = None
        validation_targets = None
        if validation is not None:
            validation_inputs = self._prepare_inputs(
                validation.inputs,
                validation.name,
            )
            validation_targets = self._prepare_targets(
                validation.targets,
                validation_inputs.shape[0],
                validation.name,
            )
            if validation_inputs.shape[1] != train_inputs.shape[1]:
                raise ValueError(
                    "Training and validation feature dimensions differ."
                )

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        random_generator = np.random.default_rng(self.random_state)
        self.input_dim = int(train_inputs.shape[1])
        self.data_signature = self._data_signature(
            train_inputs=train_inputs,
            train_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
        )
        self.model = _BinaryMLP(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(self.device)

        n_positive = float(train_targets.sum())
        n_negative = float(len(train_targets) - n_positive)
        positive_weight = (
            n_negative / n_positive
            if n_positive > 0.0 and n_negative > 0.0
            else 1.0
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, device=self.device),
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        has_validation = validation_inputs is not None
        monitor_metric = "validation_auprc" if has_validation else "train_loss"
        history: list[dict[str, Any]] = []
        best_epoch = 0
        best_monitor_value = -math.inf if has_validation else math.inf
        best_validation_loss = None
        best_state_dict = None
        epochs_without_improvement = 0
        optimizer_steps = 0
        stopped_early = False
        resumed_from_epoch = 0
        self.max_dense_batch_rows = 0

        if self.resume_from is not None:
            resume_checkpoint = self._load_payload(
                self.resume_from,
                self.device,
            )
            self._validate_resume_checkpoint(
                resume_checkpoint,
                self.input_dim,
            )
            if resume_checkpoint.get("monitor_metric") != monitor_metric:
                raise ValueError(
                    "Resume checkpoint validation configuration differs."
                )
            self.model.load_state_dict(
                resume_checkpoint["model_state_dict"])
            optimizer.load_state_dict(
                resume_checkpoint["optimizer_state_dict"])
            history = [dict(row) for row in resume_checkpoint["history"]]
            resumed_from_epoch = int(resume_checkpoint["epoch"])
            best_epoch = int(resume_checkpoint["best_epoch"])
            best_monitor_value = float(
                resume_checkpoint["best_monitor_value"])
            best_validation_loss = resume_checkpoint[
                "best_validation_loss"
            ]
            best_state_dict = resume_checkpoint["best_model_state_dict"]
            epochs_without_improvement = int(
                resume_checkpoint["epochs_without_improvement"])
            optimizer_steps = int(resume_checkpoint["optimizer_steps"])
            stopped_early = bool(resume_checkpoint["stopped_early"])
            self.max_dense_batch_rows = int(
                resume_checkpoint.get("max_dense_batch_rows", 0))
            self._restore_rng_state(
                resume_checkpoint,
                random_generator,
            )
            self._save_best_checkpoint(
                best_state_dict=best_state_dict,
                epoch=best_epoch,
                monitor_metric=monitor_metric,
                monitor_value=best_monitor_value,
            )

        start_epoch = resumed_from_epoch + 1
        trained_epoch = False
        for epoch in range(start_epoch, self.config.max_epochs + 1):
            trained_epoch = True
            stopped_early = False
            epoch_started_at = perf_counter()
            self.model.train()
            train_loss_total = 0.0
            train_observations = 0
            row_order = random_generator.permutation(train_inputs.shape[0])
            for input_batch, target_batch in self._iter_batches(
                    train_inputs, train_targets, row_order):
                if target_batch is None:
                    raise RuntimeError("Training targets are missing.")
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(input_batch)
                loss = criterion(logits, target_batch)
                loss.backward()
                optimizer.step()
                batch_rows = input_batch.shape[0]
                train_loss_total += float(loss.item()) * batch_rows
                train_observations += batch_rows
                optimizer_steps += 1

            train_loss = train_loss_total / train_observations
            validation_loss = None
            validation_auprc = None
            if validation_inputs is not None:
                if validation_targets is None:
                    raise RuntimeError("Validation targets are missing.")
                validation_loss, validation_auprc = self._validation_summary(
                    validation_inputs,
                    validation_targets,
                    criterion,
                )
                monitor_value = validation_auprc
                improved = (
                    best_epoch == 0
                    or monitor_value
                    > best_monitor_value + self.config.min_delta
                )
            else:
                monitor_value = train_loss
                improved = (
                    best_epoch == 0
                    or monitor_value
                    < best_monitor_value - self.config.min_delta
                )

            if improved:
                best_epoch = epoch
                best_monitor_value = monitor_value
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
                best_state_dict = self._model_state_dict()
                self._save_best_checkpoint(
                    best_state_dict=best_state_dict,
                    epoch=epoch,
                    monitor_metric=monitor_metric,
                    monitor_value=monitor_value,
                )
            else:
                epochs_without_improvement += 1

            self._synchronize_device()
            should_stop = bool(
                has_validation
                and epochs_without_improvement >= self.config.patience
            )
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_auprc": validation_auprc,
                "monitor_metric": monitor_metric,
                "monitor_value": monitor_value,
                "improved": improved,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": float(
                    perf_counter() - epoch_started_at
                ),
                "train_batches": math.ceil(
                    train_inputs.shape[0] / self.config.batch_size
                ),
                "validation_batches": (
                    0
                    if validation_inputs is None
                    else math.ceil(
                        validation_inputs.shape[0] / self.config.batch_size
                    )
                ),
            })
            if best_state_dict is None:
                raise RuntimeError("No best torch_mlp state was selected.")
            self._save_last_checkpoint(
                optimizer=optimizer,
                random_generator=random_generator,
                history=history,
                best_epoch=best_epoch,
                best_monitor_value=best_monitor_value,
                best_validation_loss=best_validation_loss,
                best_state_dict=best_state_dict,
                epochs_without_improvement=epochs_without_improvement,
                optimizer_steps=optimizer_steps,
                stopped_early=should_stop,
                monitor_metric=monitor_metric,
            )
            if should_stop:
                stopped_early = True
                break

        if best_state_dict is None:
            raise RuntimeError("No best torch_mlp state was selected.")
        if not trained_epoch and self.resume_from is not None:
            self._save_last_checkpoint(
                optimizer=optimizer,
                random_generator=random_generator,
                history=history,
                best_epoch=best_epoch,
                best_monitor_value=best_monitor_value,
                best_validation_loss=best_validation_loss,
                best_state_dict=best_state_dict,
                epochs_without_improvement=epochs_without_improvement,
                optimizer_steps=optimizer_steps,
                stopped_early=stopped_early,
                monitor_metric=monitor_metric,
            )

        self.model.load_state_dict(best_state_dict)
        self._synchronize_device()
        fit_seconds = perf_counter() - fit_started_at
        metadata = {
            "epochs_completed": len(history),
            "max_epochs": self.config.max_epochs,
            "optimizer_steps": optimizer_steps,
            "best_epoch": best_epoch,
            "best_validation_auprc": (
                best_monitor_value if has_validation else None
            ),
            "best_validation_loss": best_validation_loss,
            "early_stopping_metric": monitor_metric,
            "stopped_early": stopped_early,
            "resumed_from": (
                None if self.resume_from is None else str(self.resume_from)
            ),
            "resumed_from_sha256": (
                None
                if self.resume_from is None
                else file_sha256(self.resume_from)
            ),
            "resumed_from_epoch": resumed_from_epoch,
            "best_checkpoint_path": (
                None
                if self.best_checkpoint_path is None
                else str(self.best_checkpoint_path)
            ),
            "last_checkpoint_path": (
                None
                if self.last_checkpoint_path is None
                else str(self.last_checkpoint_path)
            ),
            "device": str(self.device),
            "batch_size": self.config.batch_size,
            "max_dense_batch_rows": self.max_dense_batch_rows,
            "input_dim": self.input_dim,
            "hidden_dim": self.config.hidden_dim,
        }
        return BackendFitResult(
            fit_seconds=fit_seconds,
            training_history=tuple(history),
            metadata=metadata,
        )

    def predict(self, inputs: Any) -> BackendPrediction:
        """Predict in mini-batches, densifying only the active batch."""
        if self.model is None or self.input_dim is None:
            raise RuntimeError("torch_mlp must be fit before prediction.")
        prediction_inputs = self._prepare_inputs(inputs, "prediction")
        if prediction_inputs.shape[1] != self.input_dim:
            raise ValueError(
                "Prediction feature dimension differs from training."
            )

        score_batches = []
        self.model.eval()
        row_order = np.arange(prediction_inputs.shape[0])
        with torch.inference_mode():
            for input_batch, _ in self._iter_batches(
                    prediction_inputs, None, row_order):
                logits = self.model(input_batch)
                score_batches.append(
                    torch.sigmoid(logits).cpu().numpy()
                )
        scores = np.concatenate(score_batches).astype(float, copy=False)
        predictions = (scores >= 0.5).astype(int)
        return BackendPrediction(
            scores=scores,
            predictions=predictions,
            default_threshold=0.5,
        )
