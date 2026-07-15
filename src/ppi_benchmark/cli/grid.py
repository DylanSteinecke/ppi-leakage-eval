#!/usr/bin/env python3

"""Run immutable PPI benchmark configurations from one resolved grid."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.models import CLASSIFIER_CHOICES, is_baseline_classifier
from ..features import FEATURE_CHOICES, PLM_FEATURE
from ..inputs import SPLIT_STRATEGY_CHOICES
from ..protein_encoders import (
    DEFAULT_ESM2_MODEL,
    DEFAULT_PROTEIN_ENCODER_ADAPTER,
    PROTEIN_ENCODER_ADAPTER_CHOICES,
)


DEFAULT_SPLIT_STRATEGIES = ("random", "c1", "c2", "c3")
DEFAULT_BASELINE_CLASSIFIERS = ("always_positive", "always_negative")
CONFIG_FILENAME = "benchmark_config.json"
RUNS_DIRNAME = "runs"
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
GRID_OWNED_TRAIN_FLAGS = frozenset({
    "--append-results",
    "--classifier",
    "--classifiers",
    "--execution-id",
    "--fasta",
    "--features",
    "--k",
    "--max-iter",
    "--max-pairs",
    "--model-seed",
    "--model-seeds",
    "--n-split-trials",
    "--num-reruns",
    "--pairs",
    "--plm-model",
    "--plm-adapter",
    "--plm-revision",
    "--protein-metadata",
    "--run-dir",
    "--sample-fraction",
    "--sampling-seed",
    "--seed",
    "--sequence-clusters",
    "--split-col",
    "--split-name",
    "--split-seed",
    "--split-strategy",
    "--train-size",
    "--val-size",
})


@dataclass(frozen=True)
class BenchmarkProfile:
    """Built-in feature/model defaults for one benchmark scale."""

    max_pairs: int | None
    n_split_trials: int
    feature_sets: tuple[tuple[str, ...], ...]
    learned_classifiers: tuple[str, ...]


BUILTIN_PROFILES = {
    "laptop": BenchmarkProfile(
        max_pairs=10_000,
        n_split_trials=25,
        feature_sets=(("tfidf",), ("count",)),
        learned_classifiers=("sgd_logistic",),
    ),
    "exhaustive": BenchmarkProfile(
        max_pairs=None,
        n_split_trials=100,
        feature_sets=(
            ("tfidf",),
            ("bm25",),
            ("count",),
            ("binary",),
            ("tfidf", "bm25", "count", "binary"),
        ),
        learned_classifiers=("logistic", "linear_svm", "sgd_logistic"),
    ),
}


@dataclass(frozen=True)
class BenchmarkGridConfig:
    """Complete, validated specification for one immutable benchmark grid."""

    pairs: Path
    fasta: Path
    out_dir: Path
    run_name: str
    profile: str
    protein_metadata: Path | None
    sequence_clusters: Path | None
    max_pairs: int | None
    sampling_seed: int
    train_size: float
    val_size: float
    split_strategies: tuple[str, ...]
    split_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]
    n_split_trials: int
    max_iter: int
    k: int
    feature_sets: tuple[tuple[str, ...], ...]
    baseline_classifiers: tuple[str, ...]
    learned_classifiers: tuple[str, ...]
    include_sgd: bool
    include_torch_mlp: bool
    include_plm: bool
    plm_adapter: str
    plm_model: str
    plm_revision: str | None
    embedding_cache_dir: Path | None
    aggregate_results: bool
    train_args: tuple[str, ...]

    def __post_init__(self) -> None:
        if not SAFE_NAME_PATTERN.fullmatch(self.run_name):
            raise ValueError(
                "run_name may contain only letters, numbers, '.', '_', and '-'."
            )
        if self.profile not in BUILTIN_PROFILES:
            raise ValueError(f"Unknown benchmark profile: {self.profile}")
        if self.max_pairs is not None and self.max_pairs < 1:
            raise ValueError("max_pairs must be positive when provided.")
        if not 0.0 < self.train_size < 1.0:
            raise ValueError("train_size must be between 0 and 1.")
        if not 0.0 <= self.val_size < 1.0:
            raise ValueError("val_size must be at least 0 and less than 1.")
        if self.train_size + self.val_size >= 1.0:
            raise ValueError("train_size + val_size must be less than 1.")
        if self.n_split_trials < 1 or self.max_iter < 1 or self.k < 1:
            raise ValueError("n_split_trials, max_iter, and k must be positive.")
        if not self.split_strategies:
            raise ValueError("At least one split strategy is required.")
        if not self.split_seeds or not self.model_seeds:
            raise ValueError("Split and model seed lists must not be empty.")
        if len(set(self.split_strategies)) != len(self.split_strategies):
            raise ValueError("split_strategies cannot contain duplicates.")
        if len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError("split_seeds cannot contain duplicates.")
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("model_seeds cannot contain duplicates.")
        unknown_splits = set(self.split_strategies) - set(
            SPLIT_STRATEGY_CHOICES
        )
        if unknown_splits:
            raise ValueError(f"Unknown split strategies: {sorted(unknown_splits)}")
        if not self.feature_sets:
            raise ValueError("At least one feature set is required.")
        for feature_set in self.feature_sets:
            if not feature_set:
                raise ValueError("Feature sets must not be empty.")
            unknown_features = set(feature_set) - set(FEATURE_CHOICES)
            if unknown_features:
                raise ValueError(f"Unknown features: {sorted(unknown_features)}")
            if PLM_FEATURE in feature_set and len(feature_set) != 1:
                raise ValueError("PLM must be a standalone feature set.")
        if len(set(self.feature_sets)) != len(self.feature_sets):
            raise ValueError("feature_sets cannot contain duplicates.")
        all_classifiers = (
            *self.baseline_classifiers,
            *self.learned_classifiers,
        )
        unknown_classifiers = set(all_classifiers) - set(CLASSIFIER_CHOICES)
        if unknown_classifiers:
            raise ValueError(
                f"Unknown classifiers: {sorted(unknown_classifiers)}"
            )
        if not self.baseline_classifiers:
            raise ValueError("The baseline classifier list must not be empty.")
        if len(set(all_classifiers)) != len(all_classifiers):
            raise ValueError("Classifier lists cannot contain duplicates.")
        invalid_baselines = {
            classifier
            for classifier in self.baseline_classifiers
            if not is_baseline_classifier(classifier)
        }
        if invalid_baselines:
            raise ValueError(
                "baseline_classifiers contains learned models: "
                f"{sorted(invalid_baselines)}"
            )
        learned_baselines = {
            classifier
            for classifier in self.learned_classifiers
            if is_baseline_classifier(classifier)
        }
        if learned_baselines:
            raise ValueError(
                "learned_classifiers contains baseline models: "
                f"{sorted(learned_baselines)}"
            )
        torch_requested = "torch_mlp" in self.learned_classifiers
        sgd_requested = "sgd_logistic" in self.learned_classifiers
        if self.include_sgd != sgd_requested:
            raise ValueError(
                "include_sgd must match whether sgd_logistic is in "
                "learned_classifiers."
            )
        if self.include_torch_mlp != torch_requested:
            raise ValueError(
                "include_torch_mlp must match whether torch_mlp is in "
                "learned_classifiers."
            )
        plm_requested = (PLM_FEATURE,) in self.feature_sets
        if self.include_plm != plm_requested:
            raise ValueError(
                "include_plm must match whether ['plm'] is in feature_sets."
            )
        if plm_requested and not self.plm_revision:
            raise ValueError("PLM grids require an immutable plm_revision.")
        if self.plm_adapter not in PROTEIN_ENCODER_ADAPTER_CHOICES:
            raise ValueError(
                f"Unknown protein encoder adapter: {self.plm_adapter}"
            )
        if plm_requested and not self.learned_classifiers:
            raise ValueError(
                "PLM features require at least one learned classifier. "
                "Enable SGD or another learned backend."
            )
        _validate_extra_train_args(self.train_args)

    @property
    def benchmark_dir(self) -> Path:
        """Return the immutable root for this grid invocation."""
        return self.out_dir / self.run_name

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible configuration."""
        values = asdict(self)
        values["task"] = "ppi"
        for key in (
            "pairs",
            "fasta",
            "out_dir",
            "protein_metadata",
            "sequence_clusters",
            "embedding_cache_dir",
        ):
            value = values[key]
            values[key] = None if value is None else str(value)
        values["benchmark_dir"] = str(self.benchmark_dir)
        return values


@dataclass(frozen=True)
class GridRunSpec:
    """One independent training process within a benchmark grid."""

    split_strategy: str
    split_seed: int
    configuration_name: str
    run_dir: Path
    train_args: tuple[str, ...]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _load_toml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("rb") as input_file:
        document = tomllib.load(input_file)
    unknown_sections = set(document) - {"grid", "plm"}
    if unknown_sections:
        raise ValueError(
            f"Unknown benchmark config sections: {sorted(unknown_sections)}"
        )
    values = dict(document.get("grid", {}))
    plm_values = document.get("plm", {})
    plm_key_map = {
        "enabled": "include_plm",
        "adapter": "plm_adapter",
        "model": "plm_model",
        "revision": "plm_revision",
        "embedding_cache_dir": "embedding_cache_dir",
    }
    unknown_plm_keys = set(plm_values) - set(plm_key_map)
    if unknown_plm_keys:
        raise ValueError(
            f"Unknown [plm] config keys: {sorted(unknown_plm_keys)}"
        )
    for key, value in plm_values.items():
        values[plm_key_map[key]] = value
    return values


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run laptop or exhaustive PPI grids as immutable per-configuration "
            "run directories."
        )
    )
    parser.add_argument("--config", default=None, help="Optional TOML grid config")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--fasta", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--profile", choices=tuple(BUILTIN_PROFILES), default=None)
    parser.add_argument("--protein-metadata", default=None)
    parser.add_argument("--sequence-clusters", default=None)
    cohort_group = parser.add_mutually_exclusive_group()
    cohort_group.add_argument("--max-pairs", type=int, default=None)
    cohort_group.add_argument(
        "--full-cohort", action="store_true", default=None,
        help="Override a profile's pair limit and use the full eligible cohort",
    )
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--train-size", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument(
        "--split-strategies", nargs="+", choices=SPLIT_STRATEGY_CHOICES,
        default=None,
    )
    parser.add_argument("--split-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-split-trials", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--include-sgd", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--include-torch-mlp", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--include-plm", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--plm-adapter",
        choices=PROTEIN_ENCODER_ADAPTER_CHOICES,
        default=None,
    )
    parser.add_argument("--plm-model", default=None)
    parser.add_argument("--plm-revision", default=None)
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument(
        "--aggregate-results", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Additional ppi-train arguments after '--'",
    )
    return parser


def _value(
        cli_value: Any, config_values: Mapping[str, Any], key: str,
        default: Any,
    ) -> Any:
    if cli_value is not None:
        return cli_value
    return config_values.get(key, default)


def _nested_feature_sets(value: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("feature_sets must be an array of string arrays.")
    feature_sets = []
    for feature_set in value:
        if isinstance(feature_set, (str, bytes)):
            raise ValueError("Each feature_set must be an array of strings.")
        feature_sets.append(tuple(str(item) for item in feature_set))
    return tuple(feature_sets)


def _validate_extra_train_args(train_args: Sequence[str]) -> None:
    """Reject extra arguments that would override grid-owned dimensions."""
    conflicting_flags = sorted({
        token.partition("=")[0]
        for token in train_args
        if token.startswith("--")
        and token.partition("=")[0] in GRID_OWNED_TRAIN_FLAGS
    })
    if conflicting_flags:
        raise ValueError(
            "Additional ppi-train arguments cannot override grid-owned "
            f"flags: {conflicting_flags}"
        )


def resolve_grid_config(argv: Sequence[str] | None = None) -> BenchmarkGridConfig:
    """Resolve built-ins, TOML configuration, and CLI overrides."""
    parser = _cli_parser()
    args = parser.parse_args(argv)
    try:
        config_values = _load_toml(args.config)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        parser.error(str(exc))

    allowed_grid_keys = {
        "pairs", "fasta", "out_dir", "run_name", "profile",
        "protein_metadata", "sequence_clusters", "max_pairs",
        "full_cohort", "sampling_seed", "train_size", "val_size",
        "split_strategies", "split_seeds", "model_seeds",
        "n_split_trials", "max_iter", "k", "include_sgd",
        "include_torch_mlp",
        "include_plm", "plm_adapter", "plm_model", "plm_revision",
        "embedding_cache_dir", "aggregate_results", "feature_sets",
        "baseline_classifiers", "learned_classifiers", "train_args",
    }
    unknown_grid_keys = set(config_values) - allowed_grid_keys
    if unknown_grid_keys:
        parser.error(f"Unknown [grid] config keys: {sorted(unknown_grid_keys)}")

    profile_name = _value(args.profile, config_values, "profile", "exhaustive")
    if profile_name not in BUILTIN_PROFILES:
        parser.error(f"Unknown benchmark profile: {profile_name}")
    profile = BUILTIN_PROFILES[profile_name]
    pairs = _value(args.pairs, config_values, "pairs", None)
    fasta = _value(args.fasta, config_values, "fasta", None)
    out_dir = _value(args.out_dir, config_values, "out_dir", None)
    if pairs is None or fasta is None or out_dir is None:
        parser.error("pairs, fasta, and out_dir are required via CLI or config.")

    full_cohort = bool(_value(
        args.full_cohort, config_values, "full_cohort", False
    ))
    max_pairs = _value(
        args.max_pairs, config_values, "max_pairs", profile.max_pairs
    )
    if args.max_pairs is not None:
        full_cohort = False
    if full_cohort:
        max_pairs = None
    feature_sets = _nested_feature_sets(
        config_values.get("feature_sets", profile.feature_sets)
    )
    learned_classifiers = tuple(
        str(value) for value in config_values.get(
            "learned_classifiers", profile.learned_classifiers
        )
    )
    if args.include_sgd is not None:
        include_sgd = args.include_sgd
    elif "include_sgd" in config_values:
        include_sgd = bool(config_values["include_sgd"])
    else:
        include_sgd = "sgd_logistic" in learned_classifiers
    if include_sgd and "sgd_logistic" not in learned_classifiers:
        learned_classifiers = (*learned_classifiers, "sgd_logistic")
    elif not include_sgd:
        learned_classifiers = tuple(
            classifier
            for classifier in learned_classifiers
            if classifier != "sgd_logistic"
        )
    include_sgd = "sgd_logistic" in learned_classifiers

    if args.include_torch_mlp is not None:
        include_torch_mlp = args.include_torch_mlp
    elif "include_torch_mlp" in config_values:
        include_torch_mlp = bool(config_values["include_torch_mlp"])
    else:
        include_torch_mlp = "torch_mlp" in learned_classifiers
    if include_torch_mlp and "torch_mlp" not in learned_classifiers:
        learned_classifiers = (*learned_classifiers, "torch_mlp")
    elif not include_torch_mlp:
        learned_classifiers = tuple(
            classifier
            for classifier in learned_classifiers
            if classifier != "torch_mlp"
        )
    include_torch_mlp = "torch_mlp" in learned_classifiers

    if args.include_plm is not None:
        include_plm = args.include_plm
    elif "include_plm" in config_values:
        include_plm = bool(config_values["include_plm"])
    else:
        include_plm = (PLM_FEATURE,) in feature_sets
    if include_plm and (PLM_FEATURE,) not in feature_sets:
        feature_sets = (*feature_sets, (PLM_FEATURE,))
    elif not include_plm:
        feature_sets = tuple(
            feature_set
            for feature_set in feature_sets
            if feature_set != (PLM_FEATURE,)
        )
    include_plm = (PLM_FEATURE,) in feature_sets
    run_name = str(_value(
        args.run_name,
        config_values,
        "run_name",
        f"{profile_name}_{_timestamp()}",
    ))
    cli_train_args = tuple(args.train_args)
    if cli_train_args and cli_train_args[0] == "--":
        cli_train_args = cli_train_args[1:]
    configured_train_args = tuple(
        str(value) for value in config_values.get("train_args", ())
    )
    protein_metadata = _value(
        args.protein_metadata, config_values, "protein_metadata", None
    )
    sequence_clusters = _value(
        args.sequence_clusters, config_values, "sequence_clusters", None
    )
    embedding_cache_dir = _value(
        args.embedding_cache_dir,
        config_values,
        "embedding_cache_dir",
        None,
    )

    try:
        return BenchmarkGridConfig(
            pairs=Path(pairs),
            fasta=Path(fasta),
            out_dir=Path(out_dir),
            run_name=run_name,
            profile=profile_name,
            protein_metadata=(
                None if protein_metadata is None else Path(protein_metadata)
            ),
            sequence_clusters=(
                None if sequence_clusters is None else Path(sequence_clusters)
            ),
            max_pairs=max_pairs,
            sampling_seed=int(_value(
                args.sampling_seed, config_values, "sampling_seed", 0
            )),
            train_size=float(_value(
                args.train_size, config_values, "train_size", 0.80
            )),
            val_size=float(_value(args.val_size, config_values, "val_size", 0.10)),
            split_strategies=tuple(
                str(value) for value in _value(
                    args.split_strategies,
                    config_values,
                    "split_strategies",
                    DEFAULT_SPLIT_STRATEGIES,
                )
            ),
            split_seeds=tuple(int(value) for value in _value(
                args.split_seeds, config_values, "split_seeds", (0,)
            )),
            model_seeds=tuple(int(value) for value in _value(
                args.model_seeds, config_values, "model_seeds", (0,)
            )),
            n_split_trials=int(_value(
                args.n_split_trials,
                config_values,
                "n_split_trials",
                profile.n_split_trials,
            )),
            max_iter=int(_value(args.max_iter, config_values, "max_iter", 1000)),
            k=int(_value(args.k, config_values, "k", 3)),
            feature_sets=feature_sets,
            baseline_classifiers=tuple(
                str(value) for value in config_values.get(
                    "baseline_classifiers", DEFAULT_BASELINE_CLASSIFIERS
                )
            ),
            learned_classifiers=learned_classifiers,
            include_sgd=include_sgd,
            include_torch_mlp=include_torch_mlp,
            include_plm=include_plm,
            plm_adapter=str(_value(
                args.plm_adapter,
                config_values,
                "plm_adapter",
                DEFAULT_PROTEIN_ENCODER_ADAPTER,
            )),
            plm_model=str(_value(
                args.plm_model, config_values, "plm_model", DEFAULT_ESM2_MODEL
            )),
            plm_revision=_value(
                args.plm_revision, config_values, "plm_revision", None
            ),
            embedding_cache_dir=(
                None
                if embedding_cache_dir is None
                else Path(embedding_cache_dir)
            ),
            aggregate_results=bool(_value(
                args.aggregate_results,
                config_values,
                "aggregate_results",
                True,
            )),
            train_args=(*configured_train_args, *cli_train_args),
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def _feature_name(feature_set: Sequence[str]) -> str:
    return "features-" + "+".join(feature_set)


def build_run_specs(config: BenchmarkGridConfig) -> tuple[GridRunSpec, ...]:
    """Expand a resolved grid into independent ppi-train invocations."""
    specs = []
    for split_strategy in config.split_strategies:
        for split_seed in config.split_seeds:
            split_root = (
                config.benchmark_dir
                / RUNS_DIRNAME
                / split_strategy
                / f"split_seed_{split_seed}"
            )
            common_args = [
                "--pairs", str(config.pairs),
                "--fasta", str(config.fasta),
                "--train-size", str(config.train_size),
                "--val-size", str(config.val_size),
                "--sampling-seed", str(config.sampling_seed),
                "--split-strategy", split_strategy,
                "--split-seed", str(split_seed),
                "--n-split-trials", str(config.n_split_trials),
                "--max-iter", str(config.max_iter),
                "--k", str(config.k),
            ]
            if config.protein_metadata is not None:
                common_args.extend([
                    "--protein-metadata", str(config.protein_metadata)
                ])
            if config.sequence_clusters is not None:
                common_args.extend([
                    "--sequence-clusters", str(config.sequence_clusters)
                ])
            if config.max_pairs is not None:
                common_args.extend(["--max-pairs", str(config.max_pairs)])

            baseline_dir = split_root / "baselines"
            specs.append(GridRunSpec(
                split_strategy=split_strategy,
                split_seed=split_seed,
                configuration_name="baselines",
                run_dir=baseline_dir,
                train_args=tuple([
                    *common_args,
                    "--run-dir", str(baseline_dir),
                    "--model-seeds", str(config.model_seeds[0]),
                    "--classifier", *config.baseline_classifiers,
                    *config.train_args,
                ]),
            ))

            for feature_set in config.feature_sets:
                if not config.learned_classifiers:
                    continue
                configuration_name = _feature_name(feature_set)
                run_dir = split_root / configuration_name
                feature_args = [
                    *common_args,
                    "--run-dir", str(run_dir),
                    "--model-seeds", *(str(seed) for seed in config.model_seeds),
                    "--features", *feature_set,
                    "--classifier", *config.learned_classifiers,
                ]
                if feature_set == (PLM_FEATURE,):
                    feature_args.extend([
                        "--plm-adapter", config.plm_adapter,
                        "--plm-model", config.plm_model,
                        "--plm-revision", str(config.plm_revision),
                    ])
                    if config.embedding_cache_dir is not None:
                        feature_args.extend([
                            "--embedding-cache-dir",
                            str(config.embedding_cache_dir),
                        ])
                feature_args.extend(config.train_args)
                specs.append(GridRunSpec(
                    split_strategy=split_strategy,
                    split_seed=split_seed,
                    configuration_name=configuration_name,
                    run_dir=run_dir,
                    train_args=tuple(feature_args),
                ))
    return tuple(specs)


def _run_module(module: str, args: Sequence[str]) -> None:
    command = [sys.executable, "-m", module, *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_grid(config: BenchmarkGridConfig) -> None:
    """Execute a benchmark grid and optionally aggregate its outputs."""
    benchmark_dir = config.benchmark_dir
    input_paths = {
        "pairs": config.pairs,
        "fasta": config.fasta,
        "protein_metadata": config.protein_metadata,
        "sequence_clusters": config.sequence_clusters,
    }
    missing_inputs = [
        f"{name}={path}"
        for name, path in input_paths.items()
        if path is not None and not path.is_file()
    ]
    if missing_inputs:
        raise ValueError(
            "Benchmark input files do not exist: " + ", ".join(missing_inputs)
        )
    if benchmark_dir.exists():
        raise ValueError(
            f"Benchmark directory already exists: {benchmark_dir}. "
            "Choose a new --run-name; grid runs are immutable."
        )
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / CONFIG_FILENAME).write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    specs = build_run_specs(config)
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Independent configurations: {len(specs)}")
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] split={spec.split_strategy}; "
            f"seed={spec.split_seed}; config={spec.configuration_name}",
            flush=True,
        )
        _run_module("ppi_benchmark.cli.train", spec.train_args)
    if config.aggregate_results:
        _run_module(
            "ppi_benchmark.cli.aggregate",
            ("--benchmark-dir", str(benchmark_dir)),
        )


def main() -> None:
    """Resolve and execute a benchmark grid."""
    config = resolve_grid_config()
    try:
        run_grid(config)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
