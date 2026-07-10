import json

import pandas as pd
import pytest


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


def test_c_splits_reject_validation_size(tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    completed_process = run_cli(
        *base_cli_args(pairs_path, fasta_path, tmp_path / "run"),
        "--split-strategy", "c3",
        "--val-size", "0.2",
        "--no-metrics-plots",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "require --val-size 0" in completed_process.stderr


@pytest.mark.parametrize("removed_flag", REMOVED_OUTPUT_FLAGS)
def test_removed_output_flags_are_not_accepted(
        tmp_path, ppi_test_data, run_cli, removed_flag):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "run"

    completed_process = run_cli(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        removed_flag,
        tmp_path / "old_output.csv",
        "--no-metrics-plots",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "unrecognized arguments" in completed_process.stderr


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
        run_dir / "invocations.jsonl",
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
    assert metadata["diagnostics"]["n_val"] > 0
    assert metadata["diagnostics"]["actual_val_size"] > 0.0
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
    assert metadata["diagnostics"]["n_val"] > 0
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
    summary_path = run_dir / "train_metrics_summary.csv"
    summary_before_failure = summary_path.read_bytes()

    changed_split = common_args.copy()
    changed_split[changed_split.index("--seed") + 1] = "12"
    completed_process = run_cli(
        *changed_split,
        "--append-results",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "different split" in completed_process.stderr
    assert summary_path.read_bytes() == summary_before_failure


def test_append_results_rejects_changed_fasta_contents(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "append_changed_fasta"
    common_args = [
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--no-metrics-plots",
    ]
    run_cli(*common_args)
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

    completed_process = run_cli(
        *changed_args,
        "--append-results",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "input file contents changed" in completed_process.stderr
    assert summary_path.read_bytes() == summary_before_failure


def test_append_results_rejects_changed_protein_metadata(
        tmp_path, ppi_test_data, run_cli):
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

    run_cli(*common_args)
    protein_metadata = pd.read_csv(protein_metadata_path, dtype="string")
    protein_metadata.loc[0, "taxon_id"] = "2"
    protein_metadata.to_csv(protein_metadata_path, index=False)
    completed_process = run_cli(
        *common_args,
        "--append-results",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "protein_metadata_file_sha256" in completed_process.stderr


def test_append_results_adds_model_rows_and_regenerates_summary(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "append_success"
    common_args = [
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--num-reruns", "1",
        "--max-iter", "100",
        "--train-size", "0.50",
        "--seed", "11",
        "--no-metrics-plots",
    ]

    run_cli(*common_args, "--classifier", "always_positive")
    run_cli(
        *common_args,
        "--classifier", "always_negative",
        "--append-results",
    )

    train_metrics = pd.read_csv(run_dir / "train_metrics.csv")
    train_summary = pd.read_csv(run_dir / "train_metrics_summary.csv")
    invocations = [
        json.loads(line)
        for line in (run_dir / "invocations.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]

    assert len(train_metrics) == 2
    assert set(train_metrics["classifier"]) == {
        "always_positive",
        "always_negative",
    }
    assert set(train_summary["classifier"]) == {
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


def test_canonical_protein_metadata_is_discovered_and_audited(
        tmp_path, ppi_test_data, run_cli):
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

    run_cli(
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


def test_fresh_rerun_removes_stale_test_outputs_when_test_is_held_out(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "fresh_cleanup"

    run_cli(
        *base_cli_args(pairs_path, fasta_path, run_dir),
        "--val-size", "0.25",
        "--eval-test-set",
        "--no-metrics-plots",
    )
    assert_exists(run_dir / "test_metrics.csv", run_dir / "predictions.csv")

    run_cli(
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
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "no_plots"

    run_cli(
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
        tmp_path, ppi_test_data, run_cli):
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

    run_cli(
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


def test_c3_metadata_reports_zero_shared_proteins_and_split_audit(
        tmp_path, run_cli):
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

    run_cli(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive",
        "--num-reruns", "1",
        "--train-size", "0.7",
        "--seed", "11",
        "--split-strategy", "c3",
        "--n-split-trials", "20",
        "--no-metrics-plots",
    )

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))
    diagnostics = metadata["diagnostics"]

    assert diagnostics["n_shared_proteins_train_test"] == 0
    assert metadata["split_audit"]["mode"] == "c3"
    assert metadata["split_audit"]["all_invariants_passed"] is True
    assert metadata["n_discarded_edges"] > 0
