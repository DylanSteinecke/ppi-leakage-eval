from dataclasses import dataclass

import pytest

from ppi_benchmark.evaluation import binary_classification_metrics
from ppi_benchmark.tasks.base import stable_task_data_signature


@dataclass(frozen=True)
class ToyResidue:
    example_id: str
    features: tuple[float, float]
    target: int | None


def test_grad_scaler_supports_torch_before_2_3(monkeypatch):
    torch = pytest.importorskip("torch")
    from ppi_benchmark import torch_training

    sentinel = object()
    calls = []
    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)

    def legacy_grad_scaler(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(torch.cuda.amp, "GradScaler", legacy_grad_scaler)

    assert torch_training._make_grad_scaler(enabled=False) is sentinel
    assert calls == [{"enabled": False}]


def test_shared_torch_trainer_accepts_non_matrix_ptm_connector(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    from ppi_benchmark.torch_checkpoints import TorchCheckpointManager
    from ppi_benchmark.torch_training import (
        TorchStepOutput,
        TorchTaskRuntime,
        TorchTrainer,
        TorchTrainerConfig,
    )

    class ToyPTMTask:
        name = "ptm"
        schema_version = 1

        def __init__(self):
            self.encoder = nn.Linear(2, 3)
            self.task_head = nn.Linear(3, 1)
            self.module = nn.ModuleDict({
                "encoder": self.encoder,
                "task_head": self.task_head,
            })
            self.criterion = nn.BCEWithLogitsLoss()
            self.batches_seen = 0

        def n_examples(self, prepared_split):
            return len(prepared_split)

        def iter_batches(self, prepared_split, batch_size, row_order):
            for start in range(0, len(row_order), batch_size):
                examples = [
                    prepared_split[index]
                    for index in row_order[start:start + batch_size]
                ]
                features = torch.tensor(
                    [example.features for example in examples],
                    dtype=torch.float32,
                )
                targets = (
                    None
                    if examples[0].target is None
                    else torch.tensor(
                        [example.target for example in examples],
                        dtype=torch.float32,
                    )
                )
                self.batches_seen += 1
                yield features, targets

        def step(self, batch, training):
            del training
            features, targets = batch
            logits = self.task_head(torch.relu(self.encoder(features))).squeeze(1)
            return TorchStepOutput(
                scores=torch.sigmoid(logits),
                targets=targets,
                loss=None if targets is None else self.criterion(logits, targets),
                n_observations=len(features),
            )

        def metric_values(self, targets, scores):
            predictions = (scores >= 0.5).astype(int)
            return binary_classification_metrics(targets, scores, predictions)

        def data_signature(self, train, validation):
            records = [
                {
                    "example_id": example.example_id,
                    "features": list(example.features),
                    "target": example.target,
                    "split": split_name,
                }
                for split_name, examples in (
                    ("train", train),
                    ("val", validation or []),
                )
                for example in examples
            ]
            return stable_task_data_signature("ptm", 1, records)

        def checkpoint_identity(self):
            return {"head": "toy_residue", "representation_dim": 3}

        def checkpoint_components(self):
            return {"encoder": self.encoder, "task_head": self.task_head}

        def runtime_state(self):
            return {"batches_seen": self.batches_seen}

        def load_runtime_state(self, state):
            self.batches_seen = int(state.get("batches_seen", 0))

        def runtime_metadata(self):
            return {"batches_seen": self.batches_seen}

    train = [
        ToyResidue(f"train-{index}", (float(index), float(index % 2)), index % 2)
        for index in range(8)
    ]
    validation = [
        ToyResidue(f"val-{index}", (float(index), float(index % 2)), index % 2)
        for index in range(4)
    ]
    prediction_rows = [
        ToyResidue("prediction", (1.0, 0.0), None),
    ]
    torch.manual_seed(7)
    task = ToyPTMTask()
    manager = TorchCheckpointManager(
        best_path=tmp_path / "ptm.best.pt",
        last_path=tmp_path / "ptm.last.pt",
    )
    config = TorchTrainerConfig(
        max_epochs=3,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        patience=2,
        min_delta=0.0,
        precision="float32",
        validation_monitor="loss",
        validation_monitor_mode="min",
    )
    checkpoint_config = {
        "max_epochs": 3,
        "batch_size": 2,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "patience": 2,
        "min_delta": 0.0,
        "precision": "float32",
        "validation_monitor": "loss",
    }
    trainer = TorchTrainer(
        task=task,
        config=config,
        device=torch.device("cpu"),
        random_state=7,
        checkpoint_manager=manager,
        backend_name="torch",
        model_name="toy_ptm",
        checkpoint_config=checkpoint_config,
        compatible_config_fields=tuple(checkpoint_config),
    )

    result = trainer.fit(train, validation)
    scores = trainer.predict(prediction_rows)
    last_checkpoint = torch.load(
        tmp_path / "ptm.last.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert isinstance(task, TorchTaskRuntime)
    assert len(result.history) >= 1
    assert set(row["monitor_metric"] for row in result.history) == {
        "validation_loss"
    }
    assert result.metadata["task"] == "ptm"
    assert result.metadata["early_stopping_mode"] == "min"
    assert scores.shape == (1,)
    assert set(last_checkpoint["component_state_dicts"]) == {
        "encoder",
        "task_head",
    }
    assert last_checkpoint["encoder_state_dict"] is not None
    assert last_checkpoint["task_head_state_dict"] is not None
    assert last_checkpoint["optimizer_state_dict"]
    assert "scheduler_state_dict" in last_checkpoint
    assert "scaler_state_dict" in last_checkpoint
    assert "python_rng_state" in last_checkpoint
    assert "numpy_rng_state" in last_checkpoint
    assert "torch_rng_state" in last_checkpoint
    assert last_checkpoint["task_name"] == "ptm"
    assert last_checkpoint["task_schema_version"] == 1
    assert last_checkpoint["data_signature"]["n_examples"] == 12
