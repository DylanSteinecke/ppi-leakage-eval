import json

import pytest

from ppi_benchmark.cli.grid import (
    CONFIG_FILENAME,
    build_run_specs,
    resolve_grid_config,
    run_grid,
)


def grid_config(tmp_path, *extra_args):
    """Resolve a small grid with stable test paths."""
    return resolve_grid_config([
        "--pairs", str(tmp_path / "pairs.csv"),
        "--fasta", str(tmp_path / "proteins.fasta"),
        "--out-dir", str(tmp_path / "results"),
        "--run-name", "test-grid",
        *extra_args,
    ])


def test_laptop_profile_expands_to_independent_run_directories(tmp_path):
    config = grid_config(
        tmp_path,
        "--profile", "laptop",
        "--split-seeds", "3", "7",
        "--model-seeds", "11", "19",
        "--no-aggregate-results",
        "--", "--no-metrics-plots",
    )

    specs = build_run_specs(config)

    assert config.to_dict()["task"] == "ppi"
    assert len(specs) == 24
    assert len({spec.run_dir for spec in specs}) == len(specs)
    assert {spec.configuration_name for spec in specs} == {
        "baselines",
        "features-tfidf",
        "features-count",
    }
    assert all(config.benchmark_dir in spec.run_dir.parents for spec in specs)
    assert all("--append-results" not in spec.train_args for spec in specs)
    assert all("--execution-id" not in spec.train_args for spec in specs)
    assert all(spec.train_args[-1] == "--no-metrics-plots" for spec in specs)

    baseline_specs = [
        spec for spec in specs if spec.configuration_name == "baselines"
    ]
    learned_specs = [
        spec for spec in specs if spec.configuration_name != "baselines"
    ]
    assert all(
        spec.train_args[spec.train_args.index("--model-seeds") + 1] == "11"
        for spec in baseline_specs
    )
    assert all(
        spec.train_args[
            spec.train_args.index("--model-seeds") + 1:
            spec.train_args.index("--features")
        ] == ("11", "19")
        for spec in learned_specs
    )


def test_profile_and_optional_model_expansion(tmp_path):
    exhaustive = grid_config(
        tmp_path,
        "--profile", "exhaustive",
        "--no-aggregate-results",
    )
    laptop_plm = grid_config(
        tmp_path,
        "--profile", "laptop",
        "--include-torch-mlp",
        "--include-plm",
        "--plm-revision", "0123456789abcdef",
        "--embedding-cache-dir", str(tmp_path / "embeddings"),
        "--no-aggregate-results",
    )

    assert len(build_run_specs(exhaustive)) == 24
    assert exhaustive.max_pairs is None
    assert len(build_run_specs(laptop_plm)) == 16
    assert laptop_plm.max_pairs == 10_000
    assert laptop_plm.include_sgd is True
    assert laptop_plm.plm_adapter == "esm2"
    assert "torch_mlp" in laptop_plm.learned_classifiers
    plm_specs = [
        spec
        for spec in build_run_specs(laptop_plm)
        if spec.configuration_name == "features-plm"
    ]
    assert len(plm_specs) == 4
    assert all(
        spec.train_args[
            spec.train_args.index("--plm-adapter") + 1
        ] == "esm2"
        for spec in plm_specs
    )
    assert all("--plm-revision" in spec.train_args for spec in plm_specs)
    assert all("--embedding-cache-dir" in spec.train_args for spec in plm_specs)


def test_sgd_model_family_can_be_disabled_independently(tmp_path):
    baseline_only = grid_config(
        tmp_path,
        "--profile", "laptop",
        "--split-strategies", "random",
        "--no-include-sgd",
        "--no-aggregate-results",
    )

    specs = build_run_specs(baseline_only)

    assert baseline_only.include_sgd is False
    assert baseline_only.learned_classifiers == ()
    assert [spec.configuration_name for spec in specs] == ["baselines"]

    with pytest.raises(SystemExit):
        grid_config(
            tmp_path,
            "--profile", "laptop",
            "--no-include-sgd",
            "--include-plm",
            "--plm-revision", "0123456789abcdef",
        )


def test_toml_config_and_cli_overrides_are_resolved(tmp_path):
    config_path = tmp_path / "grid.toml"
    config_path.write_text(
        "\n".join([
            "[grid]",
            f'pairs = "{tmp_path / "pairs.csv"}"',
            f'fasta = "{tmp_path / "proteins.fasta"}"',
            f'out_dir = "{tmp_path / "results"}"',
            'run_name = "from-config"',
            'profile = "laptop"',
            'plm_adapter = "esm2"',
            'split_strategies = ["random", "c3"]',
            "split_seeds = [2, 5]",
            "model_seeds = [13, 17]",
            'feature_sets = [["binary"]]',
            'learned_classifiers = ["linear_svm"]',
            'train_args = ["--no-metrics-plots"]',
            "aggregate_results = false",
            "",
        ]),
        encoding="utf-8",
    )

    config = resolve_grid_config([
        "--config", str(config_path),
        "--run-name", "cli-name",
        "--full-cohort",
    ])

    assert config.run_name == "cli-name"
    assert config.max_pairs is None
    assert config.split_strategies == ("random", "c3")
    assert config.split_seeds == (2, 5)
    assert config.model_seeds == (13, 17)
    assert config.feature_sets == (("binary",),)
    assert config.learned_classifiers == ("linear_svm",)
    assert config.plm_adapter == "esm2"
    assert config.train_args == ("--no-metrics-plots",)


def test_nested_plm_config_selects_explicit_adapter(tmp_path):
    config_path = tmp_path / "plm-grid.toml"
    config_path.write_text(
        "\n".join([
            "[grid]",
            f'pairs = "{tmp_path / "pairs.csv"}"',
            f'fasta = "{tmp_path / "proteins.fasta"}"',
            f'out_dir = "{tmp_path / "results"}"',
            'run_name = "plm-config"',
            'profile = "laptop"',
            "aggregate_results = false",
            "",
            "[plm]",
            "enabled = true",
            'adapter = "esm2"',
            'model = "local/model"',
            'revision = "immutable-revision"',
            "",
        ]),
        encoding="utf-8",
    )

    config = resolve_grid_config(["--config", str(config_path)])
    plm_specs = [
        spec
        for spec in build_run_specs(config)
        if spec.configuration_name == "features-plm"
    ]

    assert config.plm_adapter == "esm2"
    assert config.plm_model == "local/model"
    assert len(plm_specs) == 4
    assert all("--plm-adapter" in spec.train_args for spec in plm_specs)


def test_extra_train_args_cannot_override_grid_dimensions(tmp_path):
    with pytest.raises(SystemExit):
        grid_config(
            tmp_path,
            "--profile", "laptop",
            "--", "--split-seed", "999",
        )


def test_run_grid_records_config_and_refuses_overwrite(tmp_path, monkeypatch):
    pairs_path = tmp_path / "pairs.csv"
    fasta_path = tmp_path / "proteins.fasta"
    pairs_path.write_text("pair_id,protein_a,protein_b,label\n", encoding="utf-8")
    fasta_path.write_text(">P1\nACDE\n", encoding="utf-8")
    config = grid_config(
        tmp_path,
        "--profile", "laptop",
        "--split-strategies", "random",
        "--model-seeds", "7", "8",
    )
    calls = []
    monkeypatch.setattr(
        "ppi_benchmark.cli.grid._run_module",
        lambda module, args: calls.append((module, tuple(args))),
    )

    run_grid(config)

    config_path = config.benchmark_dir / CONFIG_FILENAME
    recorded_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert recorded_config["model_seeds"] == [7, 8]
    assert recorded_config["benchmark_dir"] == str(config.benchmark_dir)
    assert [module for module, _ in calls].count(
        "ppi_benchmark.cli.train"
    ) == 3
    assert calls[-1] == (
        "ppi_benchmark.cli.aggregate",
        ("--benchmark-dir", str(config.benchmark_dir)),
    )
    with pytest.raises(ValueError, match="already exists"):
        run_grid(config)
