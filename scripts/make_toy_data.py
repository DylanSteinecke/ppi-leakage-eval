#!/usr/bin/env python3

"""
Compatibility entry point for synthetic PPI dataset generation.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from ppi_benchmark.cli.make_toy_data import main  # noqa: E402


if __name__ == "__main__":
    main()
