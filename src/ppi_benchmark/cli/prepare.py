#!/usr/bin/env python3

"""
Prepare raw PPI datasets into canonical pipeline inputs.
"""

import argparse
from collections.abc import Sequence

from ..datasets import biogrid, generic_edges


def argument_parser() -> argparse.ArgumentParser:
    """
    Return the dataset-preparation argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prepare raw PPI datasets into pairs.csv, proteins.fasta, "
            "protein_metadata.csv, and dataset_metadata.json."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="loader_name",
        required=True,
    )
    generic_edges.register_subcommand(subparsers)
    biogrid.register_subcommand(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the selected dataset-preparation loader.
    """
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
