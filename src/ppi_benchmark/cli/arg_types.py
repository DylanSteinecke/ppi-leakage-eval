"""Reusable scalar parsers for command-line arguments."""

import argparse
import math


def positive_int(value: str) -> int:
    """Parse a positive integer."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def finite_float(value: str) -> float:
    """Parse a finite float."""
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed_value


def positive_float(value: str) -> float:
    """Parse a positive finite float."""
    parsed_value = finite_float(value)
    if parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed_value


def auto_or_positive_float(value: str) -> float | None:
    """Parse ``auto`` or a positive finite float."""
    if value.strip().lower() == "auto":
        return None
    return positive_float(value)


def auto_or_cluster_mode(value: str) -> int | None:
    """Parse ``auto`` or an MMseqs2 cluster mode from zero through three."""
    if value.strip().lower() == "auto":
        return None
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be 'auto' or an integer from 0 through 3"
        ) from exc
    if parsed_value not in range(4):
        raise argparse.ArgumentTypeError(
            "value must be 'auto' or an integer from 0 through 3"
        )
    return parsed_value


def nonnegative_float(value: str) -> float:
    """Parse a nonnegative finite float."""
    parsed_value = finite_float(value)
    if parsed_value < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed_value


def unit_interval(value: str) -> float:
    """Parse a finite float between zero and one, inclusive."""
    parsed_value = finite_float(value)
    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed_value


def proportion(value: str) -> float:
    """Parse a finite float strictly between zero and one."""
    parsed_value = finite_float(value)
    if not 0.0 < parsed_value < 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed_value


def nonnegative_proportion(value: str) -> float:
    """Parse a finite float greater than or equal to zero and below one."""
    parsed_value = finite_float(value)
    if not 0.0 <= parsed_value < 1.0:
        raise argparse.ArgumentTypeError(
            "value must be at least 0 and less than 1"
        )
    return parsed_value


def non_empty_string(value: str) -> str:
    """Parse a nonempty string after trimming whitespace."""
    parsed_value = value.strip()
    if not parsed_value:
        raise argparse.ArgumentTypeError("value must not be empty")
    return parsed_value
