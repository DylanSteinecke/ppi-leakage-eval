import json
import subprocess
from pathlib import Path

import pandas as pd

from aggregate_benchmark_results import aggregate_benchmark_results


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    """
    Write one JSON object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_invocations(path, records):
    """
    Write JSONL invocation records.
    """
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def metadata(run_dir, split_strategy="random"):
    """
    Return minimal split metadata for aggregation tests.
    """
    return {
        "execution_id": f"{split_strategy}_execution",
        "timestamp_utc": "2026-07-09T00:00:00+00:00",
        "run_dir": str(run_dir),
        "pairs_path": "processed/toy_pairs.csv",
        "fasta_path": "processed/toy_sequences.fasta",
        "pairs_file_sha256": "pairs-sha",
        "fasta_file_sha256": "fasta-sha",
        "git_commit": "abc123",
        "git_is_dirty": False,
        "split_strategy": split_strategy,
        "split_name": None,
        "split_col": None,
        "split_seed": 11,
        "target_train_size": 0.7,
        "target_val_size": 0.1,
        "target_test_size": 0.2,
        "actual_train_size": 0.7,
        "actual_val_size": 0.1,
        "actual_test_size": 0.2,
        "n_train": 7,
        "n_val": 1,
        "n_test": 2,
        "n_input_pairs_before_filtering": 10,
        "n_pairs_after_filtering": 10,
        "n_dropped_pairs": 0,
        "eval_test_set": False,
        "n_connected_components": None,
        "n_pruned_pairs": 0,
        "pruned_pair_fraction": 0.0,
        "resolved_args": {
            "features": ["tfidf"],
            "classifiers": ["always_positive"],
            "has_validation_split": True,
            "num_reruns": 3,
        },
        "diagnostics": {
            "n_shared_proteins_train_val": 0,
            "n_shared_proteins_train_test": 2,
            "n_shared_proteins_val_test": 0,
            "n_exact_ordered_pair_overlaps_train_val": 1,
            "n_exact_ordered_pair_overlaps_train_test": 0,
            "n_exact_ordered_pair_overlaps_val_test": 0,
            "n_unordered_pair_overlaps_train_val": 1,
            "n_unordered_pair_overlaps_train_test": 0,
            "n_unordered_pair_overlaps_val_test": 0,
            "n_duplicate_ordered_pairs_train": 0,
            "n_duplicate_ordered_pairs_val": 0,
            "n_duplicate_ordered_pairs_test": 0,
            "n_duplicate_unordered_pairs_train": 0,
            "n_duplicate_unordered_pairs_val": 0,
            "n_duplicate_unordered_pairs_test": 0,
        },
    }


def summary_row(split_name, features="tfidf", classifier="logistic"):
    """
    Return one metrics summary row.
    """
    return {
        "split": split_name,
        "model_name": f"{features}__{classifier}",
        "features": features,
        "classifier": classifier,
        "f1_mean": 0.75,
        "f1_standard_error": 0.01,
    }


def test_aggregate_benchmark_results_writes_manifest_and_summary(tmp_path):
    benchmark_dir = tmp_path / "benchmark"
    run_dir = benchmark_dir / "random_20260709T000000Z"
    second_run_dir = benchmark_dir / (
        "protein_disjoint_components_20260709T000000Z")

    write_json(
        run_dir / "splits" / "split_metadata.json",
        metadata(run_dir, split_strategy="random"),
    )
    write_invocations(
        run_dir / "invocations.jsonl",
        [
            {
                "append_results": False,
                "resolved_args": {
                    "features": ["tfidf"],
                    "classifiers": ["always_positive"],
                },
            },
            {
                "append_results": True,
                "resolved_args": {
                    "features": ["bm25", "count"],
                    "classifiers": ["logistic", "linear_svm"],
                },
            },
        ],
    )
    pd.DataFrame([summary_row("train")]).to_csv(
        run_dir / "train_metrics_summary.csv",
        index=False,
    )
    pd.DataFrame([summary_row("val")]).to_csv(
        run_dir / "val_metrics_summary.csv",
        index=False,
    )

    write_json(
        second_run_dir / "splits" / "split_metadata.json",
        metadata(
            second_run_dir,
            split_strategy="protein_disjoint_components",
        ),
    )
    pd.DataFrame([summary_row("train", classifier="always_positive")]).to_csv(
        second_run_dir / "train_metrics_summary.csv",
        index=False,
    )

    manifest_df, summary_df = aggregate_benchmark_results(benchmark_dir)

    assert (benchmark_dir / "benchmark_manifest.csv").exists()
    assert (benchmark_dir / "benchmark_summary.csv").exists()
    assert len(manifest_df) == 2
    assert len(summary_df) == 3

    first_manifest_row = manifest_df[
        manifest_df["split_strategy"] == "random"
    ].iloc[0]
    assert json.loads(first_manifest_row["features_run"]) == [
        "tfidf",
        "bm25",
        "count",
    ]
    assert json.loads(first_manifest_row["classifiers_run"]) == [
        "always_positive",
        "logistic",
        "linear_svm",
    ]
    assert first_manifest_row["diagnostics_n_shared_proteins_train_test"] == 2
    assert (
        first_manifest_row[
            "diagnostics_n_exact_ordered_pair_overlaps_train_val"
        ]
        == 1
    )

    assert set(summary_df["summary_split"]) == {"train", "val"}
    assert "diagnostics_n_unordered_pair_overlaps_train_val" in summary_df
    assert set(summary_df["execution_id"]) == {
        "random_execution",
        "protein_disjoint_components_execution",
    }


def test_aggregate_benchmark_results_reads_real_cli_run(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    benchmark_dir = tmp_path / "benchmark"
    run_dir = benchmark_dir / "random_real_run"

    run_cli(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive",
        "--num-reruns", "1",
        "--max-iter", "100",
        "--train-size", "0.50",
        "--seed", "11",
        "--no-metrics-plots",
    )

    manifest_df, summary_df = aggregate_benchmark_results(benchmark_dir)

    assert len(manifest_df) == 1
    assert set(summary_df["summary_split"]) == {"train", "test"}
    assert manifest_df.loc[0, "split_strategy"] == "random"
    assert manifest_df.loc[0, "train_size"] == 0.5
    assert json.loads(manifest_df.loc[0, "classifiers_run"]) == [
        "always_positive",
    ]
    assert "diagnostics_n_shared_proteins_train_test" in summary_df


def test_runner_script_has_valid_syntax_and_benchmark_knobs():
    runner_path = REPO_ROOT / "scripts" / "run_all_train_test_ppi_pred.sh"
    subprocess.run(["bash", "-n", str(runner_path)], check=True)

    runner_text = runner_path.read_text(encoding="utf-8")
    assert 'VAL_SIZE="${VAL_SIZE:-0.0}"' in runner_text
    assert "date -u +%Y%m%dT%H%M%SZ" in runner_text
    assert '--val-size "$VAL_SIZE"' in runner_text
    assert "scripts/aggregate_benchmark_results.py" in runner_text
