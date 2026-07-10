#!/usr/bin/env python3

"""
Compatibility entry point for PPI dataset preparation.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from ppi_benchmark.cli.prepare import main  # noqa: E402


if __name__ == "__main__":
    main()
