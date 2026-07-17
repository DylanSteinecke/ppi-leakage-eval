import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ppi_benchmark.backends import (
    BackendPrediction,
    ModelBackend,
    SklearnBackend,
    SupervisedSplit,
    make_model_backend,
)
from ppi_benchmark.backends.models import (
    ESTIMATOR_IDS,
    make_estimator,
    score_estimator,
)
from ppi_benchmark.reporting.performance import solver_iteration_report
from ppi_benchmark.tasks.ppi_models import (
    CONFIGURED_FEATURE_MATRIX,
    CONTROL_GROUP,
    NO_MATRIX,
    PPI_MODEL_CHOICES,
    PPI_MODEL_SPECS,
    PPIModelSpec,
    TRAINING_DEGREE_MATRIX,
    ppi_model_spec,
)


def test_sklearn_backend_matches_existing_estimator_path():
    x_train = np.asarray([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    y_train = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    x_eval = np.asarray([[-1.5, -1.5], [0.5, 1.0], [2.0, 2.0]])

    parameters = {
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "liblinear",
    }
    estimator = make_estimator(
        estimator_id="logistic",
        estimator_params=parameters,
        max_iter=100,
        random_state=7,
    )
    estimator.fit(x_train, y_train)
    expected_scores, expected_threshold = score_estimator(
        estimator,
        x_eval,
    )

    backend = make_model_backend(
        estimator_id="logistic",
        estimator_params=parameters,
        max_iter=100,
        random_state=7,
    )
    fit_result = backend.fit(
        SupervisedSplit("train", x_train, y_train),
        validation=SupervisedSplit("val", x_eval, np.asarray([0, 1, 1])),
    )
    prediction = backend.predict(x_eval)

    assert isinstance(backend, ModelBackend)
    assert isinstance(prediction, BackendPrediction)
    np.testing.assert_allclose(prediction.scores, expected_scores)
    assert prediction.default_threshold == expected_threshold == 0.5
    assert fit_result.fit_seconds >= 0.0
    assert fit_result.iteration_report == solver_iteration_report(estimator)
    assert fit_result.training_history == ()


class RecordingEstimator:
    def __init__(self):
        self.fit_inputs = None
        self.fit_targets = None

    def fit(self, inputs, targets):
        self.fit_inputs = inputs
        self.fit_targets = targets
        return self

    def predict_proba(self, inputs):
        positive_scores = np.full(len(inputs), 0.75)
        return np.column_stack((1.0 - positive_scores, positive_scores))


def test_sklearn_backend_fits_only_the_training_split():
    estimator = RecordingEstimator()
    backend = SklearnBackend(estimator)
    x_train = np.asarray([[1.0], [2.0]])
    y_train = np.asarray([0, 1])
    validation = SupervisedSplit(
        "val",
        np.asarray([[100.0]]),
        np.asarray([0]),
    )

    backend.fit(
        SupervisedSplit("train", x_train, y_train),
        validation=validation,
    )

    assert estimator.fit_inputs is x_train
    assert estimator.fit_targets is y_train


def test_unsupported_and_unknown_backends_fail_clearly():
    with pytest.raises(ValueError, match="requires backend 'sklearn'"):
        make_model_backend(
            estimator_id="logistic",
            estimator_params={},
            max_iter=100,
            random_state=7,
            backend_name="torch",
        )
    with pytest.raises(ValueError, match="Unknown model backend: jax"):
        make_model_backend(
            estimator_id="logistic",
            estimator_params={},
            max_iter=100,
            random_state=7,
            backend_name="jax",
        )


def test_ppi_model_registry_owns_public_routing_and_execution_policies():
    assert PPI_MODEL_CHOICES == tuple(
        spec.model_name for spec in PPI_MODEL_SPECS
    )
    assert set(ESTIMATOR_IDS) == {
        "constant",
        "logistic",
        "linear_svm",
        "sgd_logistic",
        "hist_gradient_boosting",
        "torch_mlp",
    }
    positive = ppi_model_spec("always_positive")
    degree_logistic = ppi_model_spec("degree_logistic")
    degree_hgb = ppi_model_spec("degree_hgb")
    logistic = ppi_model_spec("logistic")
    assert positive.matrix_source == NO_MATRIX
    assert positive.reporting_group == CONTROL_GROUP
    assert positive.execution_policy.run_per_model_seed is False
    assert positive.execution_policy.force_fixed_threshold is True
    assert degree_logistic.matrix_source == TRAINING_DEGREE_MATRIX
    assert degree_logistic.estimator_id == "logistic"
    assert degree_logistic.execution_policy.resolve_iteration_budget(
        max_iter=17,
        torch_max_epochs=5,
    ) == 17
    assert degree_hgb.execution_policy.resolve_iteration_budget(
        max_iter=17,
        torch_max_epochs=5,
    ) == 100
    assert logistic.matrix_source == CONFIGURED_FEATURE_MATRIX
    assert logistic.reporting_role == "predictive_model"
    with pytest.raises(TypeError):
        positive.estimator_params["positive_probability"] = 0.5
    with pytest.raises(ValueError, match="Unknown PPI model: missing"):
        ppi_model_spec("missing")
    with pytest.raises(TypeError, match="JSON-compatible"):
        PPIModelSpec(
            "invalid",
            "logistic",
            CONFIGURED_FEATURE_MATRIX,
            "predictor",
            logistic.execution_policy,
            {"invalid": {1, 2}},
        )


def test_degree_controls_use_the_fixed_predeclared_estimators():
    logistic_spec = ppi_model_spec("degree_logistic")
    hgb_spec = ppi_model_spec("degree_hgb")
    logistic = make_estimator(
        logistic_spec.estimator_id,
        estimator_params=logistic_spec.estimator_params,
        max_iter=3,
        random_state=17,
    )
    hgb = make_estimator(
        hgb_spec.estimator_id,
        estimator_params=hgb_spec.estimator_params,
        max_iter=100,
        random_state=17,
    )

    assert logistic.solver == "liblinear"
    assert logistic.class_weight == "balanced"
    assert logistic.C == 1.0
    assert logistic.max_iter == 3
    assert logistic.random_state == 17
    assert hgb.max_depth == 3
    assert hgb.max_iter == 100
    assert hgb.learning_rate == 0.05
    assert hgb.l2_regularization == 1.0
    assert hgb.early_stopping is False
    assert hgb.class_weight == "balanced"
    assert hgb.random_state == 17


def test_torch_mlp_batches_sparse_densification_and_early_stops(
        tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    dense_train = np.asarray([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    y_train = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    x_train = csr_matrix(dense_train)
    x_val = csr_matrix(dense_train[:6])
    y_val = np.asarray([0, 0, 0, 1, 1, 1])
    best_checkpoint_path = tmp_path / "model.best.pt"
    last_checkpoint_path = tmp_path / "model.last.pt"

    dense_batch_shapes = []
    original_toarray = csr_matrix.toarray

    def tracked_toarray(matrix, *args, **kwargs):
        dense_batch_shapes.append(matrix.shape)
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", tracked_toarray)
    backend = make_model_backend(
        estimator_id="torch_mlp",
        estimator_params={},
        max_iter=10,
        random_state=7,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        backend_options={
            "batch_size": 3,
            "hidden_dim": 4,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "patience": 2,
            "min_delta": 10.0,
            "device": "cpu",
        },
    )

    fit_result = backend.fit(
        SupervisedSplit("train", x_train, y_train),
        validation=SupervisedSplit("val", x_val, y_val),
    )
    prediction = backend.predict(x_val)

    assert backend.backend_name == "torch"
    assert len(fit_result.training_history) == 3
    assert fit_result.metadata["stopped_early"] is True
    assert fit_result.metadata["best_epoch"] == 1
    assert fit_result.metadata["epochs_completed"] == 3
    assert fit_result.metadata["optimizer_steps"] == 12
    assert fit_result.metadata["max_dense_batch_rows"] == 3
    assert fit_result.metadata["early_stopping_metric"] == (
        "validation_auprc")
    assert best_checkpoint_path.exists()
    assert last_checkpoint_path.exists()
    assert dense_batch_shapes
    assert max(n_rows for n_rows, _ in dense_batch_shapes) <= 3
    assert x_train.shape not in dense_batch_shapes
    assert prediction.scores.shape == (x_val.shape[0],)
    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    last_checkpoint = torch.load(
        last_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert best_checkpoint["checkpoint_kind"] == "best"
    assert best_checkpoint["format_version"] == 3
    assert best_checkpoint["monitor_metric"] == "validation_auprc"
    assert best_checkpoint["epoch"] == 1
    assert best_checkpoint["input_dim"] == x_train.shape[1]
    assert best_checkpoint["task_schema_version"] == 1
    assert set(best_checkpoint["component_state_dicts"]) == {"task_head"}
    assert best_checkpoint["task_head_state_dict"] is not None
    assert last_checkpoint["checkpoint_kind"] == "last"
    assert last_checkpoint["data_signature"]["train_inputs"]["format"] == (
        "csr")
    assert last_checkpoint["epoch"] == 3
    assert len(last_checkpoint["history"]) == 3
    assert "scheduler_state_dict" in last_checkpoint
    assert "scaler_state_dict" in last_checkpoint
    assert "python_rng_state" in last_checkpoint
    assert "torch_rng_state" in last_checkpoint


def test_torch_mlp_resumes_legacy_last_checkpoint_optimizer_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    x_train = csr_matrix(np.asarray([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
    ], dtype=np.float32))
    y_train = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    x_val = x_train[:4]
    y_val = np.asarray([0, 0, 1, 1])
    backend_options = {
        "batch_size": 2,
        "hidden_dim": 5,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "dropout": 0.2,
        "patience": 10,
        "min_delta": 0.0,
        "device": "cpu",
    }
    source_best = tmp_path / "source.best.pt"
    source_last = tmp_path / "source.last.pt"
    first_backend = make_model_backend(
        estimator_id="torch_mlp",
        estimator_params={},
        max_iter=2,
        random_state=19,
        best_checkpoint_path=source_best,
        last_checkpoint_path=source_last,
        backend_options=backend_options,
    )
    first_result = first_backend.fit(
        SupervisedSplit("train", x_train, y_train),
        validation=SupervisedSplit("val", x_val, y_val),
    )

    # Simulate the pre-refactor format-3 payload. The shared checkpoint
    # manager intentionally adapts its single model state and default config.
    legacy_checkpoint = torch.load(
        source_last,
        map_location="cpu",
        weights_only=True,
    )
    for field_name in (
        "task_name",
        "task_schema_version",
        "task_identity",
        "component_state_dicts",
        "best_component_state_dicts",
        "encoder_state_dict",
        "task_head_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "task_runtime_state",
        "best_validation_metrics",
        "monitor_mode",
        "python_rng_state",
    ):
        legacy_checkpoint.pop(field_name, None)
    legacy_checkpoint["config"].pop("precision", None)
    legacy_checkpoint["config"].pop("validation_monitor", None)
    torch.save(legacy_checkpoint, source_last)

    best_only_backend = make_model_backend(
        estimator_id="torch_mlp",
        estimator_params={},
        max_iter=4,
        random_state=19,
        resume_from=source_best,
        backend_options=backend_options,
    )
    with pytest.raises(ValueError, match="must point to a last checkpoint"):
        best_only_backend.fit(
            SupervisedSplit("train", x_train, y_train),
            validation=SupervisedSplit("val", x_val, y_val),
        )

    resumed_backend = make_model_backend(
        estimator_id="torch_mlp",
        estimator_params={},
        max_iter=4,
        random_state=19,
        best_checkpoint_path=tmp_path / "resumed.best.pt",
        last_checkpoint_path=tmp_path / "resumed.last.pt",
        resume_from=source_last,
        backend_options=backend_options,
    )
    resumed_result = resumed_backend.fit(
        SupervisedSplit("train", x_train, y_train),
        validation=SupervisedSplit("val", x_val, y_val),
    )

    uninterrupted_backend = make_model_backend(
        estimator_id="torch_mlp",
        estimator_params={},
        max_iter=4,
        random_state=19,
        best_checkpoint_path=tmp_path / "uninterrupted.best.pt",
        last_checkpoint_path=tmp_path / "uninterrupted.last.pt",
        backend_options=backend_options,
    )
    uninterrupted_result = uninterrupted_backend.fit(
        SupervisedSplit("train", x_train, y_train),
        validation=SupervisedSplit("val", x_val, y_val),
    )

    assert len(first_result.training_history) == 2
    assert [row["epoch"] for row in resumed_result.training_history] == [
        1,
        2,
        3,
        4,
    ]
    assert resumed_result.metadata["resumed_from_epoch"] == 2
    assert resumed_result.metadata["optimizer_steps"] == 16
    assert uninterrupted_result.metadata["optimizer_steps"] == 16
    np.testing.assert_allclose(
        resumed_backend.predict(x_val).scores,
        uninterrupted_backend.predict(x_val).scores,
        rtol=0.0,
        atol=1e-7,
    )

    changed_train = x_train.copy()
    changed_train[0, 0] = 0.0
    mismatched_backend = make_model_backend(
        estimator_id="torch_mlp",
        estimator_params={},
        max_iter=4,
        random_state=19,
        resume_from=source_last,
        backend_options=backend_options,
    )
    with pytest.raises(ValueError, match="data do not match"):
        mismatched_backend.fit(
            SupervisedSplit("train", changed_train, y_train),
            validation=SupervisedSplit("val", x_val, y_val),
        )
