import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

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
        "evaluation_schema_version": 1,
        "task": "ppi",
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
            "has_shared_proteins_across_splits": True,
            "has_exact_ordered_pair_overlap_across_splits": True,
            "has_unordered_pair_overlap_across_splits": True,
            "has_pair_leakage_across_splits": True,
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
    run_dir = benchmark_dir / "runs" / "random" / "seed_0" / "tfidf"
    second_run_dir = benchmark_dir / "runs" / "c3" / "seed_0" / "tfidf"

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
    assert first_manifest_row["evaluation_schema_version"] == 1
    assert first_manifest_row["task"] == "ppi"
    assert bool(
        first_manifest_row["diagnostics_has_pair_leakage_across_splits"]
    ) is True
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
    assert set(summary_df["task"]) == {"ppi"}


@pytest.mark.integration
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
        "--model-seeds", "11",
        "--max-iter", "100",
        "--train-size", "0.50",
        "--val-size", "0.0",
        "--split-seed", "11",
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
    assert "date -u +%Y-%m-%d_%H-%M-%S" in grid_text
    assert '--val-size "$VAL_SIZE"' in grid_text
    assert 'ppi-grid "${GRID_ARGS[@]}"' in grid_text
    assert "ppi-train" not in grid_text
    assert "ppi-aggregate" not in grid_text
    assert "ppi-prepare biogrid" in yeast_text
    assert "ppi-make-toy-data" in toy_text
    assert '--protein-metadata "$PROTEIN_METADATA"' in grid_text
    assert 'INCLUDE_TORCH_MLP="${INCLUDE_TORCH_MLP:-0}"' in grid_text
    assert 'INCLUDE_SGD="${INCLUDE_SGD:-1}"' in grid_text
    assert 'INCLUDE_PLM="${INCLUDE_PLM:-0}"' in grid_text
    assert 'PLM_ADAPTER="${PLM_ADAPTER:-esm2}"' in grid_text
    assert (
        'INCLUDE_LOW_RESOURCE_ESM2="${INCLUDE_LOW_RESOURCE_ESM2:-0}"'
        in grid_text
    )
    assert 'PLM_REVISION="${PLM_REVISION:-}"' in grid_text
    assert 'SPLIT_SEEDS="${SPLIT_SEEDS:-0}"' in grid_text
    assert '--split-seeds "${SPLIT_SEED_VALUES[@]}"' in grid_text
    assert '--model-seeds "${MODEL_SEED_VALUES[@]}"' in grid_text
    assert "--include-torch-mlp" in grid_text
    assert "--include-low-resource-esm2" in yeast_text
    assert "--include-low-resource-esm2" in toy_text


def test_benchmark_shell_wrapper_forwards_one_grid_command(tmp_path):
    grid_path = REPO_ROOT / "scripts" / "_run_ppi_benchmark_grid.sh"

    def wrapper_command(
            profile, include_torch=False, include_plm=False,
            include_low_resource_esm2=False,
            plm_revision="0123456789abcdef0123456789abcdef01234567",
            embedding_cache_dir=None,
            runner_args=(),
        ):
        suffix = (
            f"_{int(include_torch)}_{int(include_plm)}"
            f"_{int(include_low_resource_esm2)}"
        )
        if runner_args:
            suffix += "_" + "_".join(
                argument.lstrip("-").replace("-", "_")
                for argument in runner_args
            )
        calls_path = tmp_path / f"{profile}{suffix}_calls.txt"
        out_dir = tmp_path / f"{profile}_results"
        shell_script = r'''
set -euo pipefail
CALLS_PATH="$1"
GRID_PATH="$2"
OUT_DIR="$3"
BENCHMARK_PROFILE="$4"
INCLUDE_TORCH_MLP="$5"
INCLUDE_PLM="$6"
INCLUDE_LOW_RESOURCE_ESM2="$7"
PLM_REVISION="$8"
EMBEDDING_CACHE_DIR="$9"
ppi-grid() {
    printf '%s\n' "$*" >> "$CALLS_PATH"
}
PAIRS="pairs.csv"
FASTA="proteins.fasta"
MODEL_SEEDS="11 19"
SPLIT_SEEDS="3 7"
MAX_ITER=100
K=2
RUN_STAMP="profile-test"
AGGREGATE_RESULTS=0
shift 9
source "$GRID_PATH" --no-metrics-plots "$@"
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
                "1" if include_plm else "0",
                "1" if include_low_resource_esm2 else "0",
                plm_revision,
                str(
                    tmp_path / "shared_embeddings"
                    if embedding_cache_dir is None
                    else embedding_cache_dir
                ),
                *runner_args,
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        calls = calls_path.read_text(encoding="utf-8").splitlines()
        assert len(calls) == 1
        return calls[0]

    laptop_call = wrapper_command("laptop")
    laptop_torch_call = wrapper_command("laptop", include_torch=True)
    laptop_plm_call = wrapper_command("laptop", include_plm=True)
    laptop_esm2_call = wrapper_command(
        "laptop",
        include_low_resource_esm2=False,
        plm_revision="",
        embedding_cache_dir="",
        runner_args=("--include-low-resource-esm2",),
    )
    laptop_no_sgd_call = wrapper_command(
        "laptop",
        runner_args=("--no-sgd",),
    )
    exhaustive_call = wrapper_command("exhaustive")

    assert "--profile laptop" in laptop_call
    assert "--run-name laptop_profile-test" in laptop_call
    assert "--split-seeds 3 7" in laptop_call
    assert "--model-seeds 11 19" in laptop_call
    assert "--val-size 0.10" in laptop_call
    assert "--no-aggregate-results" in laptop_call
    assert "--no-metrics-plots" in laptop_call
    assert "--no-include-torch-mlp" in laptop_call
    assert "--include-sgd" in laptop_call
    assert "--no-include-sgd" in laptop_no_sgd_call
    assert "--include-torch-mlp" in laptop_torch_call
    assert "--include-plm" in laptop_plm_call
    assert "--plm-model facebook/esm2_t6_8M_UR50D" in laptop_plm_call
    assert "--plm-adapter esm2" in laptop_plm_call
    assert "--plm-revision 0123456789abcdef" in laptop_plm_call
    assert "--embedding-cache-dir" in laptop_plm_call
    assert "--include-plm" in laptop_esm2_call
    assert (
        "--plm-revision c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
        in laptop_esm2_call
    )
    assert "--embedding-cache-dir results/embedding_cache" in laptop_esm2_call
    assert "--plm-pooling mean" in laptop_esm2_call
    assert "--plm-device cpu" in laptop_esm2_call
    assert "--plm-precision float32" in laptop_esm2_call
    assert "--plm-max-length 1024" in laptop_esm2_call
    assert "--plm-truncation-policy truncate" in laptop_esm2_call
    assert "--plm-max-batch-tokens 1024" in laptop_esm2_call
    assert "--plm-max-batch-sequences 8" in laptop_esm2_call
    assert "--profile exhaustive" in exhaustive_call
