#!/usr/bin/env python3

"""
Compatibility entry point for benchmark aggregation.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from ppi_benchmark.cli.aggregate import main  # noqa: E402


if __name__ == "__main__":
    main()
