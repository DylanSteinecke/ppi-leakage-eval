#!/usr/bin/env python3

"""Task-neutral benchmark utility commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..reporting.run_comparison import compare_run_directories
from ..run_integrity import RunIntegrityError


STATUS_ORDER = (
    "run_contract_match",
    "encoder_contract_match",
    "embedding_table_match",
    "matrix_contract_match",
    "matrix_row_identity_match",
    "matrix_value_match",
    "matrix_match",
    "model_configuration_match",
    "model_output_match",
)


def argument_parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the umbrella benchmark command line."""
    parser = argparse.ArgumentParser(prog="protein-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare two completed runs through realized model inputs.",
    )
    compare_parser.add_argument("reference_run")
    compare_parser.add_argument("candidate_run")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    """Run one task-neutral benchmark utility command."""
    args = argument_parser(argv)
    if args.command != "compare-runs":  # pragma: no cover - argparse guards it
        raise AssertionError(f"Unhandled command: {args.command}")
    try:
        comparison = compare_run_directories(
            args.reference_run,
            args.candidate_run,
        )
    except (OSError, ValueError, RunIntegrityError) as exc:
        print(f"compare-runs: {exc}", file=sys.stderr)
        return 2

    for name in STATUS_ORDER:
        print(f"{name}: {comparison.statuses[name]}")
    for difference in comparison.differences:
        print(f"difference: {difference}")
    if comparison.matches_realized_inputs:
        print("overall: match_through_realized_model_inputs")
        return 0
    print("overall: checked_scientific_identity_differs")
    return 1


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
