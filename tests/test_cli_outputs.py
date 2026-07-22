import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ppi_benchmark.cli.train import (
    argument_parser,
    configure_logging,
    make_configuration_id,
)


REMOVED_OUTPUT_FLAGS = [
    "--pred-out",
    "--train-metrics-out",
    "--test-metrics-out",
    "--val-metrics-out",
    "--train-metrics-summary-out",
    "--test-metrics-summary-out",
    "--val-metrics-summary-out",
    "--train-metrics-plot-out",
    "--test-metrics-plot-out",
    "--val-metrics-plot-out",
    "--train-test-metrics-plot-out",
    "--train-test-metrics-png-out",
    "--train-test-f1-heatmap-out",
]


def assert_exists(*paths):
    """
    Assert that every path exists.
    """
    for path in paths:
        assert path.exists(), f"Missing expected output: {path}"


def assert_not_written(*paths):
    """
    Assert that optional output paths were not written.
    """
    for path in paths:
        assert not path.exists(), f"Unexpected output was written: {path}"


def base_cli_args(
        pairs_path, fasta_path, run_dir, classifier="always_positive"):
    """
    Return fast CLI args shared by workflow regression tests.
    """
    args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", classifier,
        "--max-iter", "100",
        "--train-size", "0.50",
        "--val-size", "0.0",
        "--split-seed", "11",
        "--model-seeds", "11",
    ]

    return args


def parse_cli_args(*args):
    """Parse CLI arguments without starting a Python subprocess."""
    return argument_parser([*map(str, args)])


def test_cli_without_run_dir_fails_clearly(ppi_test_data, capsys):
    pairs_path, fasta_path = ppi_test_data

    with pytest.raises(SystemExit) as error:
        parse_cli_args(
            "--pairs", pairs_path,
            "--fasta", fasta_path,
            "--classifier", "always_positive",
        )

    assert error.value.code == 2
    assert "--run-dir" in capsys.readouterr().err


def test_cli_help_shows_canonical_seed_flags_only(capsys):
    with pytest.raises(SystemExit) as error:
        parse_cli_args("--help")
    stdout = capsys.readouterr().out

    assert error.value.code == 0
    assert "--split-seed" in stdout
    assert "--model-seeds" in stdout
    assert "--num-reruns" not in stdout
    assert "--model-seed MODEL_SEED" not in stdout
    assert "--append-results" not in stdout
    assert "--execution-id" not in stdout
    assert "--classifiers" not in stdout


def test_cli_defaults_to_explicit_esm2_adapter(tmp_path, ppi_test_data):
    pairs_path, fasta_path = ppi_test_data

    args = parse_cli_args(
        *base_cli_args(pairs_path, fasta_path, tmp_path / "adapter"),
        "--no-metrics-plots",
    )

    assert args.plm_adapter == "esm2"


def test_classifier_interface_rejects_repetition_and_duplicates(
        tmp_path, ppi_test_data, capsys):
    pairs_path, fasta_path = ppi_test_data
    common_args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", tmp_path / "classifiers",
    ]

    with pytest.raises(SystemExit) as repeated_error:
        parse_cli_args(
            *common_args,
            "--classifier", "logistic",
            "--classifier", "linear_svm",
        )
    assert repeated_error.value.code == 2
    assert "exactly once" in capsys.readouterr().err

    with pytest.raises(SystemExit) as duplicate_error:
        parse_cli_args(
            *common_args,
            "--classifier", "logistic", "logistic",
        )
    assert duplicate_error.value.code == 2
    assert "duplicates" in capsys.readouterr().err

    hidden_alias = parse_cli_args(
        *common_args,
        "--classifiers", "logistic", "linear_svm",
    )
    assert hidden_alias.classifiers == ["logistic", "linear_svm"]


def test_configuration_id_is_exact_deterministic_and_path_safe():
    assert make_configuration_id(
        "tfidf+k3 / hash", "model/name"
    ) == "tfidf_k3_hash__model_name"


def test_control_only_selection_ignores_only_implicit_features(
        tmp_path, ppi_test_data, capsys):
    pairs_path, fasta_path = ppi_test_data
    common_args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", tmp_path / "controls",
        "--classifier", "degree_logistic", "always_positive",
    ]

    implicit = parse_cli_args(*common_args)
    assert implicit.features == ("tfidf",)
    assert implicit.features_explicit is False

    with pytest.raises(SystemExit) as feature_error:
        parse_cli_args(*common_args, "--features", "tfidf")
    assert feature_error.value.code == 2
    assert "unused" in capsys.readouterr().err

    with pytest.raises(SystemExit) as plm_error:
        parse_cli_args(*common_args, "--plm-device", "cpu")
    assert plm_error.value.code == 2
    assert "unused" in capsys.readouterr().err


def test_cli_preset_resolves_identity_and_safe_defaults(
        tmp_path, ppi_test_data, capsys):
    pairs_path, fasta_path = ppi_test_data
    base_args = base_cli_args(
        pairs_path, fasta_path, tmp_path / "preset")
    base_args[base_args.index("always_positive")] = "sgd_logistic"

    args = parse_cli_args(
        *base_args,
        "--features", "plm",
        "--plm-preset", "protbert",
    )

    assert args.plm_adapter == "protbert"
    assert args.plm_model == "Rostlab/prot_bert"
    assert args.plm_device == "cpu"
    assert args.plm_max_batch_sequences == 1
    assert args.plm_preset_metadata["pretraining_scope"] == "self_supervised"

    with pytest.raises(SystemExit) as error:
        parse_cli_args(
            *base_args,
            "--features", "plm",
            "--plm-preset", "protbert",
            "--plm-model", "another/model",
        )
    assert error.value.code == 2
    assert "conflicts" in capsys.readouterr().err


def test_explicit_model_seeds_control_every_fit(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "explicit_model_seeds"

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "sgd_logistic",
        "--features", "count",
        "--train-size", "0.50",
        "--val-size", "0.25",
        "--split-seed", "3",
        "--model-seeds", "11", "19",
        "--no-metrics-plots",
    )

    metrics = pd.read_csv(run_dir / "train_metrics.csv")
    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["model_seed"].tolist() == [11, 19]
    assert metadata["model_seeds"] == [11, 19]


def test_legacy_seed_flags_remain_accepted_with_warning(
        tmp_path, ppi_test_data, caplog):
    pairs_path, fasta_path = ppi_test_data

    args = parse_cli_args(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", tmp_path / "legacy_seeds",
        "--classifier", "always_positive",
        "--seed", "7",
        "--model-seed", "13",
        "--num-reruns", "1",
        "--no-metrics-plots",
    )
    with caplog.at_level("WARNING"):
        configure_logging(args)

    assert "Deprecated CLI: --seed is deprecated" in caplog.text
    assert "--model-seed/--num-reruns are deprecated" in (
        caplog.text
    )


@pytest.mark.parametrize("removed_flag", REMOVED_OUTPUT_FLAGS)
def test_removed_output_flags_are_not_accepted(
        tmp_path, ppi_test_data, capsys, removed_flag):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "run"

    with pytest.raises(SystemExit) as error:
        parse_cli_args(
            *base_cli_args(pairs_path, fasta_path, run_dir),
            removed_flag,
            tmp_path / "old_output.csv",
            "--no-metrics-plots",
        )

    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_cli_with_run_dir_writes_no_validation_outputs(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "run"

    run_train(*base_cli_args(pairs_path, fasta_path, run_dir))

    assert_exists(
        run_dir / "train_metrics.csv",
        run_dir / "test_metrics.csv",
        run_dir / "predictions.csv",
        run_dir / "train_metrics_summary.csv",
        run_dir / "test_metrics_summary.csv",
        run_dir / "plots" / "train_metrics_summary.svg",
        run_dir / "plots" / "test_metrics_summary.svg",
        run_dir / "plots" / "train_test_metrics_summary.svg",
        run_dir / "splits" / "split_assignments.csv",
        run_dir / "splits" / "dropped_pairs.csv",
        run_dir / "splits" / "split_metadata.json",
        run_dir / "invocations.jsonl",
        run_dir / "performance.jsonl",
    )
    assert_not_written(
        run_dir / "val_metrics.csv",
        run_dir / "val_metrics_summary.csv",
        run_dir / "plots" / "val_metrics_summary.svg",
        run_dir / "plots" / "train_val_metrics_summary.svg",
        run_dir / "splits" / "split_diagnostics.json",
    )
    predictions = pd.read_csv(run_dir / "predictions.csv", nrows=0)
    assert "source_row_index" in predictions.columns[:6]
    assert {
        "evaluation_schema_version",
        "task",
        "split",
        "example_id",
        "target",
        "score",
        "prediction",
    } <= set(predictions.columns)
    performance = json.loads(
        (run_dir / "performance.jsonl").read_text(encoding="utf-8"))
    train_metrics = pd.read_csv(run_dir / "train_metrics.csv")
    assert performance["evaluation_schema_version"] == 2
    assert performance["task"] == "ppi"
    assert performance["total_seconds"] > 0.0
    assert performance["peak_memory_bytes"] > 0
    assert performance["matrices"] == {}
    assert performance["model_runs"][0]["solver_iterations"] is None
    assert performance["runtime_provenance"]["packages"]["numpy"]
    assert "classifier" not in train_metrics.columns
    assert train_metrics.loc[0, "model_name"] == "always_positive"
    assert train_metrics.loc[0, "estimator_id"] == "constant"
    assert train_metrics.loc[0, "configuration_id"] == (
        "none__always_positive"
    )
    assert train_metrics.loc[0, "matrix_source"] == "none"
    assert train_metrics.loc[0, "reporting_group"] == "control"
    assert pd.isna(train_metrics.loc[0, "matrix_sha256"])
    assert json.loads(train_metrics.loc[0, "estimator_params"]) == {
        "positive_probability": 1.0,
    }
    assert {
        "load_pairs",
        "load_sequences",
        "validate_inputs",
        "sample_cohort",
        "split_cohort",
        "prepare_artifacts",
        "baseline_models",
        "summarize_results",
        "plot_results",
    } <= set(performance["stages_seconds"])


def test_default_generated_split_is_true_train_val_test(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "generated_val_run"

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive",
        "--max-iter", "100",
        "--split-seed", "11",
        "--model-seeds", "11",
        "--no-metrics-plots",
    )

    assert_exists(
        run_dir / "train_metrics.csv",
        run_dir / "val_metrics.csv",
        run_dir / "train_metrics_summary.csv",
        run_dir / "val_metrics_summary.csv",
    )
    assert_not_written(
        run_dir / "test_metrics.csv",
        run_dir / "test_metrics_summary.csv",
        run_dir / "predictions.csv",
        run_dir / "plots" / "test_metrics_summary.svg",
        run_dir / "plots" / "train_test_metrics_summary.svg",
    )
    assignments = pd.read_csv(
        run_dir / "splits" / "split_assignments.csv")
    assert set(assignments["split"]) == {"train", "val", "test"}

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))
    train_metrics = pd.read_csv(run_dir / "train_metrics.csv")
    val_metrics = pd.read_csv(run_dir / "val_metrics.csv")
    assert set(train_metrics["task"]) == {"ppi"}
    assert set(val_metrics["task"]) == {"ppi"}
    assert set(train_metrics["evaluation_schema_version"]) == {2}
    assert metadata["task"] == "ppi"
    assert metadata["evaluation_schema_version"] == 2
    assert metadata["target_train_size"] == pytest.approx(0.8)
    assert metadata["target_val_size"] == pytest.approx(0.1)
    assert metadata["target_test_size"] == pytest.approx(0.1)
    assert metadata["n_val"] > 0
    assert metadata["n_test"] > 0
    assert metadata["actual_test_size"] > 0.0


def test_learned_model_reports_matrix_and_solver_performance(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "learned_performance"

    run_train(
        *base_cli_args(
            pairs_path, fasta_path, run_dir, classifier="sgd_logistic"),
        "--features", "count",
        "--no-metrics-plots",
    )

    performance = json.loads(
        (run_dir / "performance.jsonl").read_text(encoding="utf-8"))
    train_matrix = performance["matrices"]["train"]
    model_run = performance["model_runs"][0]
    train_metrics = pd.read_csv(run_dir / "train_metrics.csv")

    assert performance["stages_seconds"]["feature_extraction"] > 0.0
    assert train_matrix["shape"][0] == 12
    assert train_matrix["n_columns"] > 0
    assert 0.0 < train_matrix["density"] <= 1.0
    assert train_matrix["storage_bytes"] > 0
    assert train_matrix["matrix_source"] == "configured_features"
    assert train_matrix["split"] == "train"
    assert train_matrix["matrix_schema_id"] == (
        "ppi.configured_features.v1"
    )
    assert train_matrix["pair_composition_schema_id"] == (
        "ppi.sum_absdiff_product.v1"
    )
    assert train_matrix["matrix_persisted"] is False
    assert len(train_matrix["matrix_contract_sha256"]) == 64
    assert len(train_matrix["row_identity_sha256"]) == 64
    assert len(train_matrix["matrix_sha256"]) == 64
    assert train_matrix["matrix_contract"][
        "fitted_extractor_sha256"
    ] == model_run["fitted_extractor_sha256"]
    assert model_run["matrices"]["train"]["matrix_sha256"] == (
        train_matrix["matrix_sha256"]
    )
    assert "matrix_contract" not in model_run["matrices"]["train"]
    assert model_run["fit_seconds"] > 0.0
    assert model_run["solver_iterations"]["maximum"] >= 1
    assert model_run["evaluation_seconds"]["train"] > 0.0
    assert train_metrics.loc[0, "fit_seconds"] > 0.0
    assert train_metrics.loc[0, "solver_iterations"] >= 1
    assert train_metrics.loc[0, "evaluation_seconds"] > 0.0
    assert train_metrics.loc[0, "threshold_selection"] == (
        "fixed_no_validation"
    )
    assert train_metrics.loc[0, "decision_threshold"] == pytest.approx(0.5)
    assert train_metrics.loc[0, "default_decision_threshold"] == (
        pytest.approx(0.5)
    )
    assert train_metrics.loc[0, "matrix_contract_sha256"] == (
        train_matrix["matrix_contract_sha256"]
    )
    assert train_metrics.loc[0, "row_identity_sha256"] == (
        train_matrix["row_identity_sha256"]
    )
    assert train_metrics.loc[0, "matrix_sha256"] == (
        train_matrix["matrix_sha256"]
    )
    assert not bool(train_metrics.loc[0, "matrix_persisted"])
    assert not any(
        path.suffix in {".npy", ".npz", ".mtx"}
        for path in run_dir.rglob("*")
    )


def test_mixed_matrix_sources_route_models_and_seed_policies(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "mixed_sources"

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive", "degree_logistic", "logistic",
        "--features", "count",
        "--train-size", "0.50",
        "--val-size", "0.0",
        "--split-seed", "11",
        "--model-seeds", "3", "7",
        "--max-iter", "20",
        "--no-metrics-plots",
    )

    metrics = pd.read_csv(run_dir / "train_metrics.csv")
    assert metrics.groupby("model_name").size().to_dict() == {
        "always_positive": 1,
        "degree_logistic": 2,
        "logistic": 2,
    }
    assert metrics.groupby("model_name")["matrix_source"].first().to_dict() == {
        "always_positive": "none",
        "degree_logistic": "training_degree",
        "logistic": "configured_features",
    }
    assert set(metrics["configuration_id"]) == {
        "none__always_positive",
        "training_degree_v1__degree_logistic",
        next(
            value
            for value in metrics["configuration_id"]
            if value.endswith("__logistic")
        ),
    }


def test_torch_mlp_writes_history_checkpoint_and_all_benchmark_outputs(
        tmp_path, ppi_test_data, run_train):
    pytest.importorskip("torch")
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "torch_mlp"

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "sgd_logistic", "torch_mlp",
        "--features", "count",
        "--k", "2",
        "--train-size", "0.50",
        "--val-size", "0.25",
        "--split-seed", "11",
        "--model-seeds", "11",
        "--eval-test-set",
        "--torch-max-epochs", "5",
        "--torch-batch-size", "4",
        "--torch-hidden-dim", "8",
        "--torch-dropout", "0",
        "--torch-patience", "1",
        "--torch-min-delta", "100",
        "--torch-device", "cpu",
        "--no-metrics-plots",
    )

    assert_exists(
        run_dir / "train_metrics.csv",
        run_dir / "val_metrics.csv",
        run_dir / "test_metrics.csv",
        run_dir / "train_metrics_summary.csv",
        run_dir / "val_metrics_summary.csv",
        run_dir / "test_metrics_summary.csv",
        run_dir / "predictions.csv",
        run_dir / "training_history.csv",
        run_dir / "performance.jsonl",
    )
    history = pd.read_csv(run_dir / "training_history.csv")
    train_metrics = pd.read_csv(run_dir / "train_metrics.csv")
    val_metrics = pd.read_csv(run_dir / "val_metrics.csv")
    test_metrics = pd.read_csv(run_dir / "test_metrics.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    performance = json.loads(
        (run_dir / "performance.jsonl").read_text(encoding="utf-8"))
    torch_metrics = train_metrics[
        train_metrics["model_name"] == "torch_mlp"
    ].iloc[0]
    model_run = next(
        record
        for record in performance["model_runs"]
        if record["model_name"] == "torch_mlp"
    )
    training = model_run["training"]
    best_checkpoint_path = Path(training["best_checkpoint_path"])
    last_checkpoint_path = Path(training["last_checkpoint_path"])

    assert history["epoch"].tolist() == [1, 2]
    assert set(history["task"]) == {"ppi"}
    assert set(history["evaluation_schema_version"]) == {2}
    assert set(history["backend"]) == {"torch"}
    assert set(history["feature_spec_sha256"]) == {
        torch_metrics["feature_spec_sha256"]
    }
    assert set(history["fitted_extractor_sha256"]) == {
        torch_metrics["fitted_extractor_sha256"]
    }
    assert set(history["matrix_schema_id"]) == {
        "ppi.configured_features.v1"
    }
    assert set(history["pair_composition_schema_id"]) == {
        "ppi.sum_absdiff_product.v1"
    }
    assert history["validation_loss"].notna().all()
    assert history["validation_auprc"].notna().all()
    assert set(history["monitor_metric"]) == {"validation_auprc"}
    assert history["train_batches"].tolist() == [3, 3]
    assert set(train_metrics["model_name"]) == {
        "sgd_logistic",
        "torch_mlp",
    }
    assert torch_metrics["max_iter"] == 5
    assert set(predictions["model_name"]) == {
        "sgd_logistic",
        "torch_mlp",
    }
    assert predictions["pred_score"].between(0.0, 1.0).all()
    assert predictions["decision_threshold"].notna().all()
    assert set(predictions["task"]) == {"ppi"}
    assert set(predictions["split"]) == {"test"}
    assert predictions["target"].equals(predictions["label"])
    assert predictions["score"].equals(predictions["pred_score"])
    assert predictions["prediction"].equals(predictions["pred_label"])
    assert set(predictions["threshold_selection"]) == {"validation_f1"}
    assert model_run["backend"] == "torch"
    assert training["stopped_early"] is True
    assert training["best_epoch"] == 1
    assert training["early_stopping_metric"] == "validation_auprc"
    assert training["max_dense_batch_rows"] <= 4
    assert best_checkpoint_path.exists()
    assert last_checkpoint_path.exists()
    learned_metrics = pd.concat(
        [train_metrics, val_metrics, test_metrics],
        ignore_index=True,
    )
    assert set(learned_metrics["threshold_selection"]) == {
        "validation_f1",
    }
    assert set(learned_metrics["threshold_metric"]) == {"f1"}
    for _, model_metrics in learned_metrics.groupby("model_name"):
        assert model_metrics["decision_threshold"].nunique() == 1
        validation_f1 = model_metrics.loc[
            model_metrics["split"] == "val",
            "f1",
        ].iloc[0]
        assert model_metrics["threshold_metric_value"].tolist() == (
            pytest.approx([validation_f1] * len(model_metrics))
        )
    assert model_run["decision_threshold"]["strategy"] == "validation_f1"
    assert model_run["decision_threshold"]["metric"] == "f1"


@pytest.mark.integration
@pytest.mark.slow
def test_torch_mlp_cli_resumes_from_last_checkpoint(
        tmp_path, ppi_test_data, run_cli, capsys):
    pytest.importorskip("torch")
    pairs_path, fasta_path = ppi_test_data
    first_run_dir = tmp_path / "torch_first"
    resumed_run_dir = tmp_path / "torch_resumed"
    shared_args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--classifier", "torch_mlp",
        "--features", "count",
        "--k", "2",
        "--train-size", "0.50",
        "--val-size", "0.25",
        "--split-seed", "13",
        "--model-seeds", "17",
        "--torch-batch-size", "4",
        "--torch-hidden-dim", "8",
        "--torch-dropout", "0",
        "--torch-patience", "10",
        "--torch-device", "cpu",
        "--no-metrics-plots",
    ]
    run_cli(
        *shared_args,
        "--run-dir", first_run_dir,
        "--torch-max-epochs", "1",
    )
    first_performance = json.loads(
        (first_run_dir / "performance.jsonl").read_text(encoding="utf-8"))
    first_training = first_performance["model_runs"][0]["training"]
    last_checkpoint = Path(first_training["last_checkpoint_path"])
    source_checkpoint_sha256 = hashlib.sha256(
        last_checkpoint.read_bytes()
    ).hexdigest()

    run_cli(
        *shared_args,
        "--run-dir", resumed_run_dir,
        "--torch-max-epochs", "2",
        "--torch-resume-from", last_checkpoint,
    )

    history = pd.read_csv(resumed_run_dir / "training_history.csv")
    resumed_performance = json.loads(
        (resumed_run_dir / "performance.jsonl").read_text(encoding="utf-8"))
    resumed_training = resumed_performance["model_runs"][0]["training"]
    assert history["epoch"].tolist() == [1, 2]
    assert resumed_training["resumed_from"] == str(last_checkpoint)
    assert resumed_training["resumed_from_sha256"] == (
        source_checkpoint_sha256)
    assert resumed_training["resumed_from_epoch"] == 1
    assert resumed_training["epochs_completed"] == 2
    assert hashlib.sha256(last_checkpoint.read_bytes()).hexdigest() == (
        source_checkpoint_sha256)

    with pytest.raises(SystemExit) as error:
        parse_cli_args(
            *shared_args,
            "--run-dir", first_run_dir,
            "--torch-max-epochs", "2",
            "--torch-resume-from", last_checkpoint,
        )
    assert error.value.code == 2
    assert "must be outside --run-dir" in capsys.readouterr().err
    assert hashlib.sha256(last_checkpoint.read_bytes()).hexdigest() == (
        source_checkpoint_sha256)


@pytest.mark.integration
@pytest.mark.slow
def test_frozen_plm_cli_caches_the_cohort_and_reuses_it_across_splits(
        tmp_path, ppi_test_data, tiny_esm_model, run_cli, run_train):
    pytest.importorskip("torch")
    pairs_path, fasta_path = ppi_test_data
    cache_dir = tmp_path / "embedding_cache"
    common_args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--classifier", "sgd_logistic",
        "--features", "plm",
        "--plm-model", tiny_esm_model,
        "--plm-revision", "local-test-revision",
        "--plm-device", "cpu",
        "--plm-max-length", "32",
        "--plm-max-batch-tokens", "64",
        "--plm-max-batch-sequences", "4",
        "--embedding-cache-dir", cache_dir,
        "--max-iter", "20",
        "--train-size", "0.50",
        "--val-size", "0.25",
        "--model-seeds", "17",
        "--no-metrics-plots",
    ]
    first_run_dir = tmp_path / "plm_split_1"
    second_run_dir = tmp_path / "plm_split_2"

    run_cli(
        *common_args,
        "--run-dir", first_run_dir,
        "--split-seed", "3",
    )
    first_metadata = json.loads(
        (first_run_dir / "protein_encoder.json").read_text(
            encoding="utf-8"))
    first_performance = json.loads(
        (first_run_dir / "performance.jsonl").read_text(encoding="utf-8"))

    assert first_metadata["n_proteins"] == 48
    assert first_metadata["adapter"] == "esm2"
    assert first_metadata["n_unique_sequences"] == 10
    assert first_metadata["cache_hits"] == 0
    assert first_metadata["cache_misses"] == 10
    assert first_metadata["model_loaded_for_cache_misses"] is True
    assert first_metadata["model_released_after_encoding"] is True
    assert set(first_performance["matrices"]) == {"train", "val"}
    assert first_performance["observations"]["protein_encoder"] == (
        first_metadata)
    plm_train_matrix = first_performance["matrices"]["train"]
    assert plm_train_matrix["matrix_contract"]["encoder_fingerprint"] == (
        first_metadata["encoder_fingerprint"]
    )
    assert plm_train_matrix["matrix_contract"][
        "fitted_extractor_sha256"
    ] is None
    assert plm_train_matrix["matrix_persisted"] is False
    assert first_performance["runtime_provenance"][
        "configured_precisions"
    ]["frozen_plm"] == "float32"
    assert not (first_run_dir / "test_metrics.csv").exists()
    first_metrics = pd.read_csv(first_run_dir / "train_metrics.csv")
    assert first_metrics.loc[0, "encoder_fingerprint"] == (
        first_metadata["encoder_fingerprint"])
    assert first_metrics.loc[0, "encoder_adapter"] == "esm2"
    assert bool(first_metrics.loc[0, "encoder_label_independent"]) is True
    assert pd.isna(first_metrics.loc[0, "encoder_checkpoint_sha256"])
    assert pd.isna(first_metrics.loc[0, "encoder_training_split_sha256"])
    assert first_metadata["encoder_fingerprint"][:12] in (
        first_metrics.loc[0, "feature_identity"])

    run_cli(
        *common_args,
        "--run-dir", second_run_dir,
        "--split-seed", "11",
    )
    second_metadata = json.loads(
        (second_run_dir / "protein_encoder.json").read_text(
            encoding="utf-8"))

    assert second_metadata["encoder_fingerprint"] == (
        first_metadata["encoder_fingerprint"])
    assert second_metadata["n_proteins"] == 48
    assert second_metadata["cache_hits"] == 10
    assert second_metadata["cache_misses"] == 0
    assert second_metadata["encoded_batches"] == 0
    assert second_metadata["model_loaded_for_cache_misses"] is False


    # Freeze the first assignment, then change only held-out labels. Neither
    # cached embeddings nor train/validation model results may change.
    assignments = pd.read_csv(
        first_run_dir / "splits" / "split_assignments.csv"
    ).sort_values("source_row_index")
    original_pairs = pd.read_csv(pairs_path)
    original_pairs["heldout_split"] = assignments["split"].to_numpy()
    changed_pairs = original_pairs.copy()
    heldout_rows = changed_pairs["heldout_split"] == "test"
    changed_pairs.loc[heldout_rows, "label"] = (
        1 - changed_pairs.loc[heldout_rows, "label"]
    )
    provided_pairs_path = tmp_path / "provided_pairs.csv"
    changed_pairs_path = tmp_path / "changed_test_labels.csv"
    original_pairs.to_csv(provided_pairs_path, index=False)
    changed_pairs.to_csv(changed_pairs_path, index=False)

    invariant_metrics = []
    for name, input_pairs in (
            ("original", provided_pairs_path),
            ("changed", changed_pairs_path),
        ):
        run_dir = tmp_path / f"heldout_labels_{name}"
        provided_args = list(common_args)
        provided_args[1] = input_pairs
        run_train(
            *provided_args,
            "--run-dir", run_dir,
            "--split-col", "heldout_split",
            "--split-name", "heldout-invariance",
            "--split-seed", "3",
        )
        metadata = json.loads(
            (run_dir / "protein_encoder.json").read_text(encoding="utf-8"))
        assert metadata["cache_hits"] == 10
        assert metadata["cache_misses"] == 0
        split_metrics = pd.concat([
            pd.read_csv(run_dir / "train_metrics.csv"),
            pd.read_csv(run_dir / "val_metrics.csv"),
        ], ignore_index=True)
        invariant_metrics.append(split_metrics.drop(
            columns=[
                "execution_id",
                "fit_seconds",
                "evaluation_seconds",
                # The construction contract intentionally binds the full
                # input-dataset identity, including held-out labels.
                "matrix_contract_sha256",
            ],
        ))

    pd.testing.assert_frame_equal(
        invariant_metrics[0],
        invariant_metrics[1],
        check_exact=True,
    )


@pytest.mark.parametrize(
    ("adapter", "model_fixture", "preprocessing"),
    [
        (
            "protbert",
            "tiny_protbert_model",
            "prottrans_space_separated_map_uzob_to_x",
        ),
        (
            "prott5",
            "tiny_prott5_model",
            "prottrans_space_separated_map_uzob_to_x",
        ),
    ],
)
def test_prottrans_adapters_run_end_to_end_without_network(
        tmp_path, ppi_test_data, run_train, request,
        adapter, model_fixture, preprocessing):
    pytest.importorskip("torch")
    pairs_path, fasta_path = ppi_test_data
    model_path = request.getfixturevalue(model_fixture)
    run_dir = tmp_path / adapter

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "sgd_logistic",
        "--features", "plm",
        "--plm-adapter", adapter,
        "--plm-model", model_path,
        "--plm-revision", "local-test-revision",
        "--plm-device", "cpu",
        "--plm-max-length", "32",
        "--plm-max-batch-tokens", "64",
        "--plm-max-batch-sequences", "4",
        "--embedding-cache-dir", tmp_path / "cache",
        "--max-iter", "20",
        "--train-size", "0.50",
        "--val-size", "0.25",
        "--eval-test-set",
        "--model-seeds", "17",
        "--no-metrics-plots",
    )

    metadata = json.loads(
        (run_dir / "protein_encoder.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "test_metrics.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert metadata["adapter"] == adapter
    assert metadata["sequence_preprocessing"] == preprocessing
    assert metadata["model_loaded_for_cache_misses"] is True
    assert metadata["model_released_after_encoding"] is True
    assert metrics.loc[0, "encoder_adapter"] == adapter
    assert set(predictions["encoder_adapter"]) == {adapter}


def test_split_and_model_seeds_are_independent(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    assignments_by_seed = {}
    for split_seed in (3, 7):
        run_dir = tmp_path / f"split_seed_{split_seed}"
        run_train(
            "--pairs", pairs_path,
            "--fasta", fasta_path,
            "--run-dir", run_dir,
            "--classifier", "sgd_logistic",
            "--features", "count",
            "--train-size", "0.50",
            "--val-size", "0.25",
            "--split-seed", str(split_seed),
            "--model-seeds", "41",
            "--no-metrics-plots",
        )
        metrics = pd.read_csv(run_dir / "train_metrics.csv")
        metadata = json.loads(
            (run_dir / "splits" / "split_metadata.json").read_text(
                encoding="utf-8"))
        assignments_by_seed[split_seed] = pd.read_csv(
            run_dir / "splits" / "split_assignments.csv")
        assert metrics.loc[0, "split_seed"] == split_seed
        assert metrics.loc[0, "model_seed"] == 41
        assert metadata["split_seed"] == split_seed
        assert metadata["model_seed"] == 41

    assert not assignments_by_seed[3].equals(assignments_by_seed[7])


def test_cli_samples_the_whole_cohort_before_splitting(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "sampled_run"

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--max-pairs", "12",
        "--sampling-seed", "17",
        "--no-metrics-plots",
    )

    selection_path = run_dir / "sampling" / "selected_examples.csv"
    assert selection_path.exists()
    selection = pd.read_csv(selection_path)
    assignments = pd.read_csv(run_dir / "splits" / "split_assignments.csv")
    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))

    assert len(selection) == 12
    assert len(assignments) == 12
    assert selection.columns.tolist() == [
        "source_row_index",
        "sampling_rank",
        "pair_id",
        "label",
    ]
    assert selection["label"].value_counts().to_dict() == {0: 6, 1: 6}
    assert metadata["n_pairs_after_filtering"] == 24
    assert metadata["n_pairs_in_sampled_cohort"] == 12
    assert metadata["sampling"]["applied"] is True
    assert metadata["sampling"]["seed"] == 17
    assert metadata["sampling"]["n_excluded"] == 12
    assert metadata["sampling"]["selected_examples_path"] == str(
        selection_path)


def test_cli_sampling_append_requires_the_same_selection(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "sample_append"
    common_args = [
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--max-pairs", "12",
        "--sampling-seed", "17",
        "--no-metrics-plots",
    ]

    run_train(*common_args)
    run_train(*common_args, "--append-results")
    changed_selection = common_args.copy()
    changed_selection[changed_selection.index("--sampling-seed") + 1] = "18"
    with pytest.raises(
            SystemExit,
            match=r"selected_examples\.csv does not match",
        ):
        run_train(*changed_selection, "--append-results")


def test_cli_rejects_two_sampling_size_options(
        tmp_path, ppi_test_data, capsys):
    pairs_path, fasta_path = ppi_test_data
    with pytest.raises(SystemExit) as error:
        parse_cli_args(
            *base_cli_args(pairs_path, fasta_path, tmp_path / "run"),
            "--max-pairs", "12",
            "--sample-fraction", "0.5",
        )

    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_provided_split_sampling_preserves_every_split_and_label(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    pairs = pd.read_csv(pairs_path)
    pairs["provided_split"] = ["train"] * 8 + ["val"] * 8 + ["test"] * 8
    provided_pairs_path = tmp_path / "provided_pairs.csv"
    pairs.to_csv(provided_pairs_path, index=False)
    run_dir = tmp_path / "provided_sample"

    run_train(
        *base_cli_args(provided_pairs_path, fasta_path, run_dir),
        "--split-col", "provided_split",
        "--max-pairs", "12",
        "--sampling-seed", "4",
        "--no-metrics-plots",
    )

    selection = pd.read_csv(
        run_dir / "sampling" / "selected_examples.csv")
    assignments = pd.read_csv(
        run_dir / "splits" / "split_assignments.csv")
    joint_counts = selection.groupby(["split", "label"]).size()

    assert set(joint_counts.index) == {
        (split_name, label)
        for split_name in ("train", "val", "test")
        for label in (0, 1)
    }
    assert assignments.groupby("split").size().to_dict() == {
        "test": 4,
        "train": 4,
        "val": 4,
    }


def test_validation_without_eval_test_set_writes_only_train_val_outputs(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "val_run"

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--val-size", "0.25",
        "--no-metrics-plots",
    )

    assert_exists(
        run_dir / "train_metrics.csv",
        run_dir / "val_metrics.csv",
        run_dir / "train_metrics_summary.csv",
        run_dir / "val_metrics_summary.csv",
    )
    assert_not_written(
        run_dir / "test_metrics.csv",
        run_dir / "predictions.csv",
        run_dir / "test_metrics_summary.csv",
        run_dir / "plots" / "test_metrics_summary.svg",
        run_dir / "plots" / "train_test_metrics_summary.svg",
    )

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))
    assert metadata["n_val"] > 0
    assert metadata["actual_val_size"] > 0.0
    assert metadata["diagnostics"]["n_val"] > 0
    assert metadata["diagnostics"]["actual_val_size"] > 0.0
    assert "val_metrics_path" in metadata
    assert "test_metrics_path" not in metadata
    assert "predictions_path" not in metadata


def test_validation_with_eval_test_set_writes_train_val_test_outputs(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "val_test_run"

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--val-size", "0.25",
        "--eval-test-set",
    )

    assert_exists(
        run_dir / "train_metrics.csv",
        run_dir / "val_metrics.csv",
        run_dir / "test_metrics.csv",
        run_dir / "predictions.csv",
        run_dir / "plots" / "train_val_metrics_summary.svg",
        run_dir / "plots" / "train_test_metrics_summary.svg",
        run_dir / "plots" / "train_val_metrics_summary.png",
        run_dir / "plots" / "train_test_metrics_summary.png",
    )

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))
    assert metadata["n_val"] > 0
    assert metadata["diagnostics"]["n_val"] > 0
    assert "val_metrics_path" in metadata
    assert "test_metrics_path" in metadata
    assert "predictions_path" in metadata


def test_append_results_requires_identical_split(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "append_run"
    common_args = [
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    ]

    run_train(*common_args)
    run_train(*common_args, "--append-results")
    summary_path = run_dir / "train_metrics_summary.csv"
    summary_before_failure = summary_path.read_bytes()

    changed_split = common_args.copy()
    changed_split[changed_split.index("--split-seed") + 1] = "12"
    with pytest.raises(SystemExit, match="different split"):
        run_train(*changed_split, "--append-results")

    assert summary_path.read_bytes() == summary_before_failure


def test_matrix_append_preflight_does_not_partially_mutate_outputs(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "matrix_preflight"
    common_args = [
        *base_cli_args(
            pairs_path,
            fasta_path,
            run_dir,
            classifier="sgd_logistic",
        ),
        "--features", "count",
        "--no-metrics-plots",
    ]
    run_train(*common_args)

    guarded_paths = [
        run_dir / "train_metrics.csv",
        run_dir / "test_metrics.csv",
        run_dir / "predictions.csv",
        run_dir / "train_metrics_summary.csv",
        run_dir / "invocations.jsonl",
    ]
    before = {path: path.read_bytes() for path in guarded_paths}
    performance_path = run_dir / "performance.jsonl"
    report = json.loads(performance_path.read_text(encoding="utf-8"))
    report["matrices"]["train"]["matrix_sha256"] = "0" * 64
    performance_path.write_text(
        json.dumps(report, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="different matrix_sha256"):
        run_train(*common_args, "--append-results")

    assert {path: path.read_bytes() for path in guarded_paths} == before
    assert len(performance_path.read_text(encoding="utf-8").splitlines()) == 1


def test_append_results_rejects_changed_fasta_contents(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "append_changed_fasta"
    common_args = [
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    ]
    run_train(*common_args)
    summary_path = run_dir / "train_metrics_summary.csv"
    summary_before_failure = summary_path.read_bytes()
    changed_fasta_path = tmp_path / "changed_proteins.fasta"
    changed_fasta_path.write_text(
        fasta_path.read_text(encoding="utf-8").replace(
            "ACDEFGHIKL",
            "ACDEYGHIKL",
        ),
        encoding="utf-8",
    )
    changed_args = common_args.copy()
    changed_args[changed_args.index("--fasta") + 1] = changed_fasta_path

    with pytest.raises(SystemExit, match="input file contents changed"):
        run_train(*changed_args, "--append-results")

    assert summary_path.read_bytes() == summary_before_failure


def test_append_results_rejects_changed_protein_metadata(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    protein_ids = sorted(
        set(pd.read_csv(pairs_path)["protein_a"])
        | set(pd.read_csv(pairs_path)["protein_b"])
    )
    protein_metadata_path = tmp_path / "taxa.csv"
    pd.DataFrame({
        "protein_id": protein_ids,
        "taxon_id": ["1"] * len(protein_ids),
    }).to_csv(protein_metadata_path, index=False)
    run_dir = tmp_path / "append_changed_taxa"
    common_args = [
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--protein-metadata", protein_metadata_path,
        "--no-metrics-plots",
    ]

    run_train(*common_args)
    protein_metadata = pd.read_csv(protein_metadata_path, dtype="string")
    protein_metadata.loc[0, "taxon_id"] = "2"
    protein_metadata.to_csv(protein_metadata_path, index=False)
    with pytest.raises(SystemExit, match="protein_metadata_file_sha256"):
        run_train(*common_args, "--append-results")


def test_append_results_adds_model_rows_and_regenerates_summary(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "append_success"
    common_args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--max-iter", "100",
        "--train-size", "0.50",
        "--split-seed", "11",
        "--model-seeds", "11",
        "--no-metrics-plots",
    ]

    run_train(*common_args, "--classifier", "always_positive")
    run_train(
        *common_args,
        "--classifier", "always_negative",
        "--append-results",
    )

    train_metrics = pd.read_csv(run_dir / "train_metrics.csv")
    train_summary = pd.read_csv(run_dir / "train_metrics_summary.csv")
    degree_metrics = pd.read_csv(run_dir / "val_degree_metrics.csv")
    degree_summary = pd.read_csv(run_dir / "val_degree_metrics_summary.csv")
    invocations = [
        json.loads(line)
        for line in (run_dir / "invocations.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]

    assert len(train_metrics) == 2
    assert set(train_metrics["model_name"]) == {
        "always_positive",
        "always_negative",
    }
    assert set(train_summary["model_name"]) == {
        "always_positive",
        "always_negative",
    }
    assert len(invocations) == 2
    assert invocations[0]["append_results"] is False
    assert invocations[1]["append_results"] is True
    assert invocations[0]["resolved_args"]["classifiers"] == [
        "always_positive",
    ]
    assert invocations[1]["resolved_args"]["classifiers"] == [
        "always_negative",
    ]
    internal_degree_keys = {
        "degree_evaluation_cohort_sha256",
        "degree_identity_context",
        "degree_protocol_id",
        "degree_protocol_version",
    }
    assert internal_degree_keys.isdisjoint(
        invocations[1]["resolved_args"]
    )
    pa_metrics = degree_metrics[
        degree_metrics["model_name"] == "preferential_attachment"
    ]
    pa_summary = degree_summary[
        degree_summary["model_name"] == "preferential_attachment"
    ]
    assert len(pa_metrics) == len(pa_summary)
    assert set(pa_summary["n_runs"]) == {1}


def test_canonical_protein_metadata_is_discovered_and_audited(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    pairs = pd.read_csv(pairs_path)
    protein_ids = sorted(set(pairs["protein_a"]) | set(pairs["protein_b"]))
    protein_metadata_path = tmp_path / "protein_metadata.csv"
    pd.DataFrame({
        "protein_id": protein_ids,
        "taxon_id": ["1" if index < 24 else "2"
                     for index in range(len(protein_ids))],
    }).to_csv(protein_metadata_path, index=False)
    run_dir = tmp_path / "species_metadata"

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    )

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))
    species = metadata["species"]

    assert metadata["protein_metadata_path"] == str(protein_metadata_path)
    assert metadata["resolved_args"]["protein_metadata"] == str(
        protein_metadata_path)
    assert species["total"]["n_taxa"] == 2
    assert species["total"]["n_proteins_without_taxon"] == 0
    assert species["total"]["protein_counts_by_taxon"] == {
        "1": 24,
        "2": 24,
    }
    assert species["total"]["n_pairs_with_unknown_taxon"] == 0


def test_canonical_dataset_metadata_is_discovered_and_audited(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    dataset_metadata_path = tmp_path / "dataset_metadata.json"
    negative_construction = {
        "label_meaning": "sampled_unobserved_pair",
        "policy": "taxon_pair_matched",
        "negative_ratio_requested": 1.0,
        "negative_ratio_realized": 1.0,
        "seed": 17,
        "timing": "before_split",
        "partition_aware": False,
    }
    dataset_metadata_path.write_text(
        json.dumps({
            "output_pairs_file_sha256": hashlib.sha256(
                pairs_path.read_bytes()
            ).hexdigest(),
            "negative_construction": negative_construction,
        }),
        encoding="utf-8",
    )
    run_dir = tmp_path / "dataset_metadata_run"

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    )

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["dataset_metadata_path"] == str(dataset_metadata_path)
    assert metadata["resolved_args"]["dataset_metadata"] == str(
        dataset_metadata_path
    )
    assert len(metadata["dataset_metadata_file_sha256"]) == 64
    assert metadata["dataset_metadata_pairs_binding"] == "sha256"
    assert metadata["negative_construction"] == negative_construction


def test_dataset_metadata_rejects_mismatched_pairs_hash(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    dataset_metadata_path = tmp_path / "metadata.json"
    dataset_metadata_path.write_text(
        json.dumps({"output_pairs_file_sha256": "not-the-pairs-hash"}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="output_pairs_file_sha256"):
        run_train(
            *base_cli_args(
                pairs_path,
                fasta_path,
                tmp_path / "mismatched_metadata",
            ),
            "--dataset-metadata", dataset_metadata_path,
            "--no-metrics-plots",
        )


def test_dataset_metadata_rejects_unbound_legacy_manifest(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    dataset_metadata_path = tmp_path / "metadata.json"
    dataset_metadata_path.write_text(
        json.dumps({
            "sampled_negatives": True,
            "species_aware_sampling": True,
        }),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cannot be bound"):
        run_train(
            *base_cli_args(
                pairs_path,
                fasta_path,
                tmp_path / "unbound_metadata",
            ),
            "--dataset-metadata", dataset_metadata_path,
            "--no-metrics-plots",
        )


def test_dataset_metadata_rejects_malformed_negative_construction(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    dataset_metadata_path = tmp_path / "metadata.json"
    dataset_metadata_path.write_text(
        json.dumps({
            "output_pairs_file_sha256": hashlib.sha256(
                pairs_path.read_bytes()
            ).hexdigest(),
            "negative_construction": {"policy": "global"},
        }),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing required fields"):
        run_train(
            *base_cli_args(
                pairs_path,
                fasta_path,
                tmp_path / "malformed_metadata",
            ),
            "--dataset-metadata", dataset_metadata_path,
            "--no-metrics-plots",
        )


def test_append_results_rejects_changed_dataset_metadata(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    dataset_metadata_path = tmp_path / "dataset_metadata.json"
    base_metadata = {
        "output_pairs_path": str(pairs_path),
        "working_directory": str(tmp_path),
        "sampled_negatives": True,
        "species_aware_sampling": True,
        "negative_ratio": 1.0,
        "seed": 3,
    }
    dataset_metadata_path.write_text(
        json.dumps(base_metadata),
        encoding="utf-8",
    )
    run_dir = tmp_path / "append_changed_dataset_metadata"
    common_args = [
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    ]
    run_train(*common_args)
    split_metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        split_metadata["negative_construction"]["policy"]
        == "taxon_pair_matched"
    )
    assert split_metadata["negative_construction"]["seed"] == 3
    assert (
        split_metadata["dataset_metadata_pairs_binding"]
        == "legacy_path_only"
    )
    base_metadata["seed"] = 4
    dataset_metadata_path.write_text(
        json.dumps(base_metadata),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="dataset_metadata_file_sha256"):
        run_train(*common_args, "--append-results")


def test_fresh_rerun_removes_stale_test_outputs_when_test_is_held_out(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "fresh_cleanup"

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--val-size", "0.25",
        "--eval-test-set",
        "--no-metrics-plots",
    )
    assert_exists(run_dir / "test_metrics.csv", run_dir / "predictions.csv")

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--val-size", "0.25",
        "--no-metrics-plots",
    )

    assert_exists(run_dir / "train_metrics.csv", run_dir / "val_metrics.csv")
    assert_not_written(run_dir / "test_metrics.csv", run_dir / "predictions.csv")
    invocations = [
        json.loads(line)
        for line in (run_dir / "invocations.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    assert len(invocations) == 1
    assert invocations[0]["resolved_args"]["evaluate_test_metrics"] is False


def test_no_metrics_plots_writes_metrics_without_plot_artifacts(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "no_plots"

    run_train(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    )

    assert_exists(
        run_dir / "train_metrics.csv",
        run_dir / "test_metrics.csv",
        run_dir / "train_metrics_summary.csv",
        run_dir / "test_metrics_summary.csv",
    )
    plots_dir = run_dir / "plots"
    assert (not plots_dir.exists()) or not any(plots_dir.iterdir())


def test_dropped_pairs_and_metadata_are_written_for_missing_fasta_proteins(
        tmp_path, ppi_test_data, run_train):
    pairs_path, fasta_path = ppi_test_data
    pairs = pd.read_csv(pairs_path)
    missing_pairs = pd.DataFrame({
        "pair_id": ["missing_a", "missing_b", "missing_both"],
        "protein_a": ["MISSING_A", pairs.loc[0, "protein_a"], "MISSING_A"],
        "protein_b": [pairs.loc[0, "protein_b"], "MISSING_B", "MISSING_B"],
        "label": [1, 0, 1],
    })
    pairs_with_missing = pd.concat(
        [pairs, missing_pairs],
        ignore_index=True,
    )
    pairs_with_missing_path = tmp_path / "pairs_with_missing.csv"
    pairs_with_missing.to_csv(pairs_with_missing_path, index=False)
    run_dir = tmp_path / "dropped_pairs"

    run_train(
        *base_cli_args(pairs_with_missing_path, fasta_path, run_dir),
        "--no-metrics-plots",
    )

    dropped_pairs = pd.read_csv(run_dir / "splits" / "dropped_pairs.csv")
    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))

    assert len(dropped_pairs) == 3
    assert metadata["n_dropped_pairs"] == 3
    assert set(dropped_pairs["drop_reason"]) == {
        "missing_protein_a_sequence",
        "missing_protein_b_sequence",
        "missing_both_sequences",
    }


def test_c3_three_way_metadata_reports_pairwise_disjoint_proteins(
        tmp_path, run_train):
    run_dir = tmp_path / "c3_metadata"
    pair_rows = ["pair_id,protein_a,protein_b,label"]
    fasta_records = []
    for protein_index in range(12):
        fasta_records.append(
            f">P{protein_index}\nACDEFGHIKL{protein_index}A\n")
    for left_index in range(12):
        for right_index in range(left_index + 1, 12):
            label = (left_index + right_index) % 2
            pair_rows.append(
                f"pair_{left_index}_{right_index},P{left_index},"
                f"P{right_index},{label}")
    pairs_path = tmp_path / "c3_pairs.csv"
    fasta_path = tmp_path / "c3_proteins.fasta"
    pairs_path.write_text("\n".join(pair_rows) + "\n", encoding="utf-8")
    fasta_path.write_text("".join(fasta_records), encoding="utf-8")

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive",
        "--train-size", "0.7",
        "--val-size", "0.15",
        "--split-seed", "11",
        "--model-seeds", "11",
        "--split-strategy", "c3",
        "--n-split-trials", "50",
        "--no-metrics-plots",
    )

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))
    diagnostics = metadata["diagnostics"]

    assert diagnostics["n_shared_proteins_train_val"] == 0
    assert diagnostics["n_shared_proteins_train_test"] == 0
    assert diagnostics["n_shared_proteins_val_test"] == 0
    assert metadata["n_val"] > 0
    assert metadata["n_test"] > 0
    assert metadata["split_audit"]["mode"] == "c3"
    assert metadata["split_audit"]["all_invariants_passed"] is True
    assert metadata["n_discarded_edges"] > 0
    assert metadata["n_pairs_after_filtering"] == 66
    assert metadata["n_pairs_in_sampled_cohort"] == 66
