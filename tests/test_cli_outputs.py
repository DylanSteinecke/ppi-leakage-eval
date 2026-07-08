import json

import pandas as pd


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


def base_cli_args(pairs_path, fasta_path, run_dir):
    """
    Return fast CLI args shared by subprocess regression tests.
    """
    args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive",
        "--num-reruns", "1",
        "--max-iter", "100",
        "--train-size", "0.50",
        "--seed", "11",
    ]

    return args


def test_cli_without_run_dir_fails_clearly(ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data

    completed_process = run_cli(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--classifier", "always_positive",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "--run-dir" in completed_process.stderr


def test_cli_with_run_dir_writes_no_validation_outputs(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "run"

    run_cli(*base_cli_args(pairs_path, fasta_path, run_dir))

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
    )
    assert_not_written(
        run_dir / "val_metrics.csv",
        run_dir / "val_metrics_summary.csv",
        run_dir / "plots" / "val_metrics_summary.svg",
        run_dir / "plots" / "train_val_metrics_summary.svg",
    )

    predictions = pd.read_csv(run_dir / "predictions.csv", nrows=0)
    assert "source_row_index" in predictions.columns[:6]


def test_validation_without_eval_test_set_writes_only_train_val_outputs(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "val_run"

    run_cli(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--val-size", "0.25",
    )

    assert_exists(
        run_dir / "train_metrics.csv",
        run_dir / "val_metrics.csv",
        run_dir / "train_metrics_summary.csv",
        run_dir / "val_metrics_summary.csv",
        run_dir / "plots" / "train_metrics_summary.svg",
        run_dir / "plots" / "val_metrics_summary.svg",
        run_dir / "plots" / "train_val_metrics_summary.svg",
        run_dir / "plots" / "train_val_metrics_summary.png",
        run_dir / "plots" / "train_val_f1_heatmap.png",
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
    assert "val_metrics_path" in metadata
    assert "test_metrics_path" not in metadata
    assert "predictions_path" not in metadata


def test_validation_with_eval_test_set_writes_train_val_test_outputs(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "val_test_run"

    run_cli(
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
    assert "val_metrics_path" in metadata
    assert "test_metrics_path" in metadata
    assert "predictions_path" in metadata


def test_append_results_requires_identical_split(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "append_run"
    common_args = [
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    ]

    run_cli(*common_args)
    run_cli(*common_args, "--append-results")

    changed_split = common_args.copy()
    changed_split[changed_split.index("--seed") + 1] = "12"
    completed_process = run_cli(
        *changed_split,
        "--append-results",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "different split" in completed_process.stderr
