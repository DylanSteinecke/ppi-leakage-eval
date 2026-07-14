"""Minimal contracts shared by PPI and future PTM task adapters."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ..backends.base import SupervisedSplit
from ..training import SplitEvaluation


@runtime_checkable
class TaskAdapter(Protocol):
    """Convert task examples to shared splits and prediction records."""

    name: str

    def make_split(
            self, name: str, examples: Any, inputs: Any,
        ) -> SupervisedSplit:
        """Build one backend-neutral supervised split."""
        ...

    def prediction_frame(
            self, examples: Any, evaluation: SplitEvaluation,
            model_metadata: Mapping[str, Any],
        ) -> Any:
        """Return standardized plus task-specific prediction records."""
        ...
