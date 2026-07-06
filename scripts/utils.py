"""
Compatibility imports for older scripts.

New code should import from focused modules such as ppi_features, ppi_inputs,
ppi_models, and ppi_results. This file should eventually disappear once no
notebooks or external scripts rely on legacy imports.
"""

from ppi_features import BM25Vectorizer


__all__ = ["BM25Vectorizer"]
