import json
import subprocess
from pathlib import Path

import pandas as pd

from ppi_benchmark.cli.aggregate import (
    aggregate_benchmark_results,
    main as aggregate_main,
)


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
        "model_seed": 23,
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
        "n_discarded_edges": 0,
        "discarded_edge_fraction": 0.0,
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
    second_run_dir = benchmark_dir / "c3_20260709T000000Z"

    first_metadata = metadata(run_dir, split_strategy="random")
    first_metadata.update({
        "sequence_clusters_path": "processed/sequence_clusters.csv",
        "sequence_clusters_file_sha256": "clusters-sha",
        "sequence_clusters": {
            "applied_to_split": False,
            "n_sequence_clusters": 17,
        },
    })
    write_json(
        run_dir / "splits" / "split_metadata.json",
        first_metadata,
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
            split_strategy="c3",
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
    assert first_manifest_row["model_seed"] == 23
    assert first_manifest_row["sequence_clusters_file_sha256"] == (
        "clusters-sha")
    assert not bool(
        first_manifest_row["sequence_cluster_grouping_applied"])
    assert first_manifest_row["n_sequence_clusters"] == 17
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
        "c3_execution",
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
        "--val-size", "0.0",
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


def test_aggregate_cli_plots_current_train_val_grid(
        tmp_path, monkeypatch):
    benchmark_dir = tmp_path / "benchmark"
    for strategy_index, strategy in enumerate(("random", "c1", "c2", "c3")):
        run_dir = benchmark_dir / f"{strategy}_current"
        run_metadata = metadata(run_dir, split_strategy=strategy)
        run_metadata["execution_id"] = f"current_grid__{strategy}"
        write_json(
            run_dir / "splits" / "split_metadata.json",
            run_metadata,
        )
        train_row = summary_row("train")
        val_row = summary_row("val")
        train_row["f1_mean"] = 0.80 - strategy_index * 0.02
        val_row["f1_mean"] = 0.74 - strategy_index * 0.04
        pd.DataFrame([train_row]).to_csv(
            run_dir / "train_metrics_summary.csv",
            index=False,
        )
        pd.DataFrame([val_row]).to_csv(
            run_dir / "val_metrics_summary.csv",
            index=False,
        )

    old_run_dir = benchmark_dir / "random_old"
    old_metadata = metadata(old_run_dir, split_strategy="random")
    old_metadata["execution_id"] = "old_grid__random"
    write_json(
        old_run_dir / "splits" / "split_metadata.json",
        old_metadata,
    )
    pd.DataFrame([summary_row("train")]).to_csv(
        old_run_dir / "train_metrics_summary.csv",
        index=False,
    )
    pd.DataFrame([summary_row("val")]).to_csv(
        old_run_dir / "val_metrics_summary.csv",
        index=False,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "ppi-aggregate",
            "--benchmark-dir", str(benchmark_dir),
            "--plot-execution-id-prefix", "current_grid__",
        ],
    )
    aggregate_main()

    plot_path = benchmark_dir / "benchmark_train_val_f1.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    from matplotlib import image as matplotlib_image

    plot_pixels = matplotlib_image.imread(plot_path)
    plot_height, plot_width = plot_pixels.shape[:2]
    assert plot_height > plot_width


def test_example_runners_have_valid_syntax_and_use_installed_commands():
    scripts_dir = REPO_ROOT / "scripts"
    runner_paths = (
        scripts_dir / "run_yeast_biogrid_ppi_example.sh",
        scripts_dir / "run_toy_ppi_example.sh",
        scripts_dir / "_run_ppi_benchmark_grid.sh",
    )
    for runner_path in runner_paths:
        subprocess.run(["bash", "-n", str(runner_path)], check=True)

    grid_text = (scripts_dir / "_run_ppi_benchmark_grid.sh").read_text(
        encoding="utf-8"
    )
    yeast_text = (scripts_dir / "run_yeast_biogrid_ppi_example.sh").read_text(
        encoding="utf-8"
    )
    toy_text = (scripts_dir / "run_toy_ppi_example.sh").read_text(
        encoding="utf-8"
    )
    assert 'VAL_SIZE="${VAL_SIZE:-0.10}"' in grid_text
    assert 'BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-exhaustive}"' in grid_text
    assert '--max-pairs "$MAX_PAIRS"' in grid_text
    assert 'BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-laptop}"' in yeast_text
    assert 'BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-exhaustive}"' in toy_text
    assert 'Unknown BENCHMARK_PROFILE' in grid_text
    assert "date -u +%Y-%m-%d_%H-%M-%S" in grid_text
    assert '--val-size "$VAL_SIZE"' in grid_text
    assert '--plot-execution-id-prefix "${EXECUTION_ID}__"' in grid_text
    assert "ppi-train" in grid_text
    assert "ppi-aggregate" in grid_text
    assert "ppi-prepare biogrid" in yeast_text
    assert "ppi-make-toy-data" in toy_text
    assert '--protein-metadata "$PROTEIN_METADATA"' in grid_text
    assert 'INCLUDE_TORCH_MLP="${INCLUDE_TORCH_MLP:-0}"' in grid_text
    assert 'SPLIT_SEEDS="${SPLIT_SEEDS:-0}"' in grid_text
    assert '--split-seed "$split_seed"' in grid_text
    assert '--model-seed "$MODEL_SEED"' in grid_text
    assert "LEARNED_CLASSIFIERS+=(torch_mlp)" in grid_text


def test_benchmark_profiles_expand_to_expected_command_grids(tmp_path):
    grid_path = REPO_ROOT / "scripts" / "_run_ppi_benchmark_grid.sh"

    def profile_commands(profile, include_torch=False, split_seeds="0"):
        suffix = "_torch" if include_torch else ""
        suffix += "_multi_seed" if " " in split_seeds else ""
        calls_path = tmp_path / f"{profile}{suffix}_calls.txt"
        out_dir = tmp_path / f"{profile}_results"
        shell_script = r'''
set -euo pipefail
CALLS_PATH="$1"
GRID_PATH="$2"
OUT_DIR="$3"
BENCHMARK_PROFILE="$4"
INCLUDE_TORCH_MLP="$5"
SPLIT_SEEDS="$6"
ppi-train() {
    printf '%s\n' "$*" >> "$CALLS_PATH"
}
ppi-aggregate() {
    :
}
PAIRS="pairs.csv"
FASTA="proteins.fasta"
NUM_RERUNS=1
MAX_ITER=100
K=2
RUN_STAMP="profile-test"
AGGREGATE_RESULTS=0
source "$GRID_PATH" --no-metrics-plots
'''
        subprocess.run(
            [
                "bash",
                "-c",
                shell_script,
                "profile-test",
                str(calls_path),
                str(grid_path),
                str(out_dir),
                profile,
                "1" if include_torch else "0",
                split_seeds,
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return calls_path.read_text(encoding="utf-8").splitlines()

    laptop_calls = profile_commands("laptop")
    laptop_torch_calls = profile_commands("laptop", include_torch=True)
    laptop_multi_seed_calls = profile_commands(
        "laptop",
        split_seeds="3 7",
    )
    exhaustive_calls = profile_commands("exhaustive")

    assert len(laptop_calls) == 12
    assert len(laptop_multi_seed_calls) == 24
    assert len(exhaustive_calls) == 24
    assert all("--n-split-trials 25" in call for call in laptop_calls)
    assert all("--val-size 0.10" in call for call in laptop_calls)
    assert all("--max-pairs 10000" in call for call in laptop_calls)
    assert all("--model-seed 0" in call for call in laptop_calls)
    assert sum(
        "--split-seed 3" in call
        for call in laptop_multi_seed_calls
    ) == 12
    assert sum(
        "--split-seed 7" in call
        for call in laptop_multi_seed_calls
    ) == 12
    assert sum(
        "seed-3_profile-test" in call
        for call in laptop_multi_seed_calls
    ) == 12
    assert sum(
        "seed-7_profile-test" in call
        for call in laptop_multi_seed_calls
    ) == 12
    laptop_learned_calls = [
        call for call in laptop_calls if "--features" in call
    ]
    assert all("--classifier sgd_logistic" in call
               for call in laptop_learned_calls)
    assert {call.split("--features ", 1)[1].split(" --classifier", 1)[0]
            for call in laptop_learned_calls} == {"tfidf", "count"}
    assert all(
        "--classifier sgd_logistic torch_mlp" in call
        for call in laptop_torch_calls
        if "--features" in call
    )

    assert all("--n-split-trials 100" in call for call in exhaustive_calls)
    assert all("--val-size 0.10" in call for call in exhaustive_calls)
    assert all("--max-pairs" not in call for call in exhaustive_calls)
    assert sum(
        "--features tfidf bm25 count binary "
        "--classifier logistic linear_svm sgd_logistic" in call
        for call in exhaustive_calls
    ) == 4
