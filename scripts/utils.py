"""
Compatibility imports for older scripts.

New code should import from the ppi_benchmark package. This file remains only
for external scripts that still use the legacy utility import.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from ppi_benchmark.features import BM25Vectorizer  # noqa: E402


__all__ = ["BM25Vectorizer"]
