import numpy as np
import pandas as pd
import pytest

from ppi_benchmark.backends import BackendFitResult, BackendPrediction
from ppi_benchmark.evaluation import BinaryClassificationPolicy
from ppi_benchmark.protein_encoders import (
    ResidueTokenAlignment,
    TokenizedProteinBatch,
)
from ppi_benchmark.schema import STANDARD_PREDICTION_COLUMNS
from ppi_benchmark.tasks import (
    PPI_TASK,
    PTM_TASK,
    PTMResidueDataset,
    PTMWindowCollator,
    TaskAdapter,
    split_ptm_examples_by_protein,
)
from ppi_benchmark.training import (
    SplitEvaluation,
    TaskSplitData,
    fit_and_evaluate_task,
)


class OffsetAlignmentProvider:
    """Map residues after two leading tokens to catch offset assumptions."""

    def tokenize_with_alignment(self, sequences):
        sequences = tuple(sequences)
        maximum_tokens = max(len(sequence) + 3 for sequence in sequences)
        attention = np.zeros((len(sequences), maximum_tokens), dtype=bool)
        residues = np.zeros_like(attention)
        input_ids = np.zeros_like(attention, dtype=np.int64)
        for row, sequence in enumerate(sequences):
            n_tokens = len(sequence) + 3
            attention[row, :n_tokens] = True
            residues[row, 2:2 + len(sequence)] = True
            input_ids[row, :n_tokens] = np.arange(n_tokens)
        return TokenizedProteinBatch(
            sequences=sequences,
            model_inputs={"input_ids": input_ids},
            alignment=ResidueTokenAlignment.from_token_masks(
                attention,
                residues,
            ),
        )


class ScoreBackend:
    backend_name = "score"

    def fit(self, train, validation=None):
        self.train = train
        self.validation = validation
        return BackendFitResult(fit_seconds=0.0)

    def predict(self, inputs):
        scores = np.asarray(inputs, dtype=float)
        return BackendPrediction(
            scores=scores,
            default_threshold=0.5,
        )


def ptm_frame():
    return pd.DataFrame({
        "example_id": ["a0", "a4", "b2"],
        "protein_id": ["A", "A", "B"],
        "residue_index": [0, 4, 2],
        "residue": ["A", "F", "K"],
        "ptm_type": ["phosphorylation"] * 3,
        "label": [0, 1, 1],
    })


def test_ptm_examples_are_split_by_protein_before_window_generation():
    dataset = PTMResidueDataset.from_dataframe(ptm_frame())

    splits = split_ptm_examples_by_protein(
        dataset,
        {"A": "train", "B": "val"},
    )

    assert set(splits) == {"train", "val"}
    assert {example.protein_id for example in splits["train"]} == {"A"}
    assert {example.protein_id for example in splits["val"]} == {"B"}
    assert [example.example_id for example in splits["train"]] == ["a0", "a4"]
    with pytest.raises(ValueError, match="without split assignments"):
        split_ptm_examples_by_protein(dataset, {"A": "train"})


def test_ptm_window_collator_preserves_edges_and_exact_token_alignment():
    dataset = PTMResidueDataset.from_dataframe(ptm_frame())
    collator = PTMWindowCollator(
        protein_sequences={"A": "ACDEF", "B": "HIKLM"},
        window_radius=2,
        alignment_provider=OffsetAlignmentProvider(),
    )

    batch = collator([dataset[0], dataset[1], dataset[2]])

    assert batch.window_sequences == ("ACD", "DEF", "HIKLM")
    np.testing.assert_array_equal(batch.window_starts, [0, 2, 0])
    np.testing.assert_array_equal(
        batch.target_window_residue_indices,
        [0, 2, 2],
    )
    # The provider uses two leading tokens, so this cannot pass through a
    # hard-coded ESM +1 offset.
    np.testing.assert_array_equal(batch.target_token_indices, [2, 4, 4])
    np.testing.assert_array_equal(batch.targets, [0, 1, 1])


def test_ptm_window_collator_rejects_residue_mismatch():
    frame = ptm_frame()
    frame.loc[0, "residue"] = "Z"
    example = PTMResidueDataset.from_dataframe(frame)[0]
    collator = PTMWindowCollator(
        protein_sequences={"A": "ACDEF"},
        window_radius=2,
        alignment_provider=OffsetAlignmentProvider(),
    )

    with pytest.raises(ValueError, match="residue mismatch"):
        collator([example])


def test_ptm_task_formats_standard_predictions_and_signatures():
    frame = ptm_frame()
    evaluation = SplitEvaluation(
        name="test",
        targets=np.asarray([0, 1, 1]),
        scores=np.asarray([0.1, 0.8, 0.7]),
        predictions=np.asarray([0, 1, 1]),
        metrics={"auprc": 1.0},
        evaluation_seconds=0.01,
    )

    predictions = PTM_TASK.prediction_frame(
        frame,
        evaluation,
        {"execution_id": "run", "model_name": "ptm_head"},
    )
    signature = PTM_TASK.data_signature(frame)
    changed = frame.copy()
    changed.loc[0, "residue_index"] = 1

    assert set(STANDARD_PREDICTION_COLUMNS) <= set(predictions.columns)
    assert set(predictions["task"]) == {"ptm"}
    assert predictions["example_id"].tolist() == ["a0", "a4", "b2"]
    assert isinstance(PTM_TASK, TaskAdapter)
    assert signature["schema_version"] == PTM_TASK.schema_version
    assert signature["n_examples"] == 3
    assert signature["sha256"] != PTM_TASK.data_signature(changed)["sha256"]


def test_high_level_task_interface_runs_ptm_without_ppi_columns():
    train_frame = ptm_frame().iloc[[0, 1]].reset_index(drop=True)
    validation_frame = ptm_frame().iloc[[1, 2]].reset_index(drop=True)
    backend = ScoreBackend()

    result = fit_and_evaluate_task(
        backend=backend,
        task=PTM_TASK,
        train=TaskSplitData("train", train_frame, np.asarray([0.1, 0.9])),
        validation=TaskSplitData(
            "val",
            validation_frame,
            np.asarray([0.8, 0.7]),
        ),
        test=None,
        evaluation_policy=BinaryClassificationPolicy(),
    )

    assert set(result.split_evaluations) == {"train", "val"}
    assert result.split_evaluations["val"].metrics["f1"] == 1.0
    assert backend.train.targets.tolist() == [0, 1]


def test_ppi_task_signature_covers_order_endpoints_and_targets():
    frame = pd.DataFrame({
        "pair_id": ["p0", "p1"],
        "protein_a": ["A", "B"],
        "protein_b": ["B", "C"],
        "label": [1, 0],
    })
    signature = PPI_TASK.data_signature(frame)

    assert signature == PPI_TASK.data_signature(frame.copy())
    assert signature["sha256"] != PPI_TASK.data_signature(
        frame.iloc[::-1].reset_index(drop=True)
    )["sha256"]


def test_ppi_and_ptm_torch_heads_use_task_specific_composition():
    torch = pytest.importorskip("torch")
    from ppi_benchmark.tasks import PPIPairHead, PTMResidueHead

    proteins = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    ppi_head = PPIPairHead(
        representation_dim=2,
        hidden_dim=3,
        dropout=0.0,
    )
    ppi_head.eval()
    forward = ppi_head(proteins, torch.tensor([0]), torch.tensor([1]))
    reverse = ppi_head(proteins, torch.tensor([1]), torch.tensor([0]))
    torch.testing.assert_close(forward, reverse)

    ptm_head = PTMResidueHead(representation_dim=2)
    with torch.no_grad():
        ptm_head.classifier.weight.fill_(1.0)
        ptm_head.classifier.bias.zero_()
    token_representations = torch.tensor([
        [[1.0, 1.0], [2.0, 3.0], [9.0, 9.0]],
        [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
    ])
    logits = ptm_head(token_representations, torch.tensor([1, 2]))
    torch.testing.assert_close(logits, torch.tensor([5.0, 17.0]))
