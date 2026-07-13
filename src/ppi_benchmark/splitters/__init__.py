"""
Dataset splitting algorithms.
"""

from .protein_disjoint import (
    SplitResult,
    split_pairs,
    split_pairs_three_way,
    ThreeWaySplitResult,
)


__all__ = [
    "SplitResult",
    "ThreeWaySplitResult",
    "split_pairs",
    "split_pairs_three_way",
]
