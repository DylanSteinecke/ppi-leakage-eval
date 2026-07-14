import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from ppi_benchmark.backends import BackendFitResult, BackendPrediction
from ppi_benchmark.evaluation import (
    BinaryClassificationPolicy,
    binary_classification_metrics,
)
from ppi_benchmark.schema import STANDARD_PREDICTION_COLUMNS
from ppi_benchmark.tasks import (
    PPI_TASK,
    PPIPairCollator,
    PPIPairDataset,
    SymmetricPairComposer,
)
from ppi_benchmark.training import (
    SplitEvaluation,
    fit_and_evaluate_backend,
)


class ScoreBackend:
    """Treat one-dimensional backend inputs as deterministic scores."""

    backend_name = "score_backend"

    def __init__(self):
        self.fit_train = None
        self.fit_validation = None
        self.predicted_inputs = []

    def fit(self, train, validation=None):
        self.fit_train = train
        self.fit_validation = validation
        return BackendFitResult(fit_seconds=0.25)

    def predict(self, inputs):
        self.predicted_inputs.append(inputs)
        scores = np.asarray(inputs, dtype=float).reshape(-1)
        return BackendPrediction(
            scores=scores,
            default_threshold=0.5,
        )


def ppi_frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "source_row_index",
            "pair_id",
            "protein_a",
            "protein_b",
            "label",
        ],
    )


def test_shared_engine_fits_once_and_selects_only_on_validation():
    backend = ScoreBackend()
    train_examples = ppi_frame([
        (0, "train_0", "A", "B", 0),
        (1, "train_1", "B", "C", 1),
    ])
    validation_examples = ppi_frame([
        (2, "val_0", "C", "D", 0),
        (3, "val_1", "D", "E", 0),
        (4, "val_2", "E", "F", 1),
        (5, "val_3", "F", "G", 1),
    ])
    test_examples = ppi_frame([
        (6, "test_0", "G", "H", 1),
        (7, "test_1", "H", "I", 0),
    ])
    train = PPI_TASK.make_split(
        "train", train_examples, np.asarray([0.2, 0.7]))
    validation = PPI_TASK.make_split(
        "val", validation_examples, np.asarray([0.1, 0.3, 0.4, 0.8]))
    test = PPI_TASK.make_split(
        "test", test_examples, np.asarray([0.2, 0.7]))

    result = fit_and_evaluate_backend(
        backend=backend,
        train=train,
        validation=validation,
        test=test,
        evaluation_policy=BinaryClassificationPolicy(),
    )

    assert backend.fit_train is train
    assert backend.fit_validation is validation
    assert len(backend.predicted_inputs) == 3
    assert result.fit_result.fit_seconds == 0.25
    assert result.operating_point.threshold == pytest.approx(0.4)
    assert result.operating_point.strategy == "validation_f1"
    assert set(result.split_evaluations) == {"train", "val", "test"}
    assert result.split_evaluations["val"].metrics["f1"] == 1.0


def test_binary_metrics_are_defined_safely_for_one_class_splits():
    metrics = binary_classification_metrics(
        targets=[0, 0],
        scores=[0.1, 0.2],
        predictions=[0, 0],
    )

    assert metrics["accuracy"] == 1.0
    assert np.isnan(metrics["precision"])
    assert np.isnan(metrics["auprc"])
    assert np.isnan(metrics["auroc"])


def test_ppi_pair_dataset_and_collator_encode_each_batch_protein_once():
    examples = ppi_frame([
        (10, "pair_a", "A", "B", 1),
        (11, "pair_b", "B", "C", 0),
    ])
    dataset = PPIPairDataset(examples)
    batch = PPIPairCollator({"A": "AAA", "B": "BBB", "C": "CCC"})(
        [dataset[0], dataset[1]]
    )

    assert len(dataset) == 2
    assert batch.example_ids == ("pair_a", "pair_b")
    assert batch.protein_ids == ("A", "B", "C")
    assert batch.sequences == ("AAA", "BBB", "CCC")
    np.testing.assert_array_equal(batch.protein_a_indices, [0, 1])
    np.testing.assert_array_equal(batch.protein_b_indices, [1, 2])
    np.testing.assert_array_equal(batch.targets, [1, 0])


def test_symmetric_pair_composer_matches_dense_and_sparse_inputs():
    examples = pd.DataFrame({
        "protein_a": ["A", "B"],
        "protein_b": ["B", "A"],
    })
    protein_ids = pd.Index(["A", "B"])
    protein_features = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    composer = SymmetricPairComposer(chunk_size=1)

    dense_pairs = composer.compose(
        examples, protein_ids, protein_features)
    sparse_pairs = composer.compose(
        examples, protein_ids, csr_matrix(protein_features))

    expected = np.asarray([
        [4.0, 6.0, 2.0, 2.0, 3.0, 8.0],
        [4.0, 6.0, 2.0, 2.0, 3.0, 8.0],
    ])
    np.testing.assert_array_equal(dense_pairs, expected)
    np.testing.assert_array_equal(sparse_pairs.toarray(), expected)


def test_ppi_predictions_include_legacy_and_common_result_fields():
    examples = ppi_frame([
        (10, "pair_a", "A", "B", 1),
        (11, "pair_b", "B", "C", 0),
    ])
    evaluation = SplitEvaluation(
        name="test",
        targets=np.asarray([1, 0]),
        scores=np.asarray([0.8, 0.2]),
        predictions=np.asarray([1, 0]),
        metrics={"f1": 1.0},
        evaluation_seconds=0.01,
    )

    predictions = PPI_TASK.prediction_frame(
        examples=examples,
        evaluation=evaluation,
        model_metadata={"execution_id": "run-1", "model_name": "model"},
    )

    assert set(STANDARD_PREDICTION_COLUMNS) <= set(predictions.columns)
    assert predictions["example_id"].tolist() == ["pair_a", "pair_b"]
    assert predictions["task"].tolist() == ["ppi", "ppi"]
    assert predictions["split"].tolist() == ["test", "test"]
    np.testing.assert_array_equal(predictions["target"], predictions["label"])
    np.testing.assert_array_equal(
        predictions["score"], predictions["pred_score"])
    np.testing.assert_array_equal(
        predictions["prediction"], predictions["pred_label"])


def test_ppi_task_rejects_misaligned_backend_inputs():
    examples = ppi_frame([
        (10, "pair_a", "A", "B", 1),
        (11, "pair_b", "B", "C", 0),
    ])

    with pytest.raises(ValueError, match="different lengths"):
        PPI_TASK.make_split("train", examples, np.ones((1, 2)))
