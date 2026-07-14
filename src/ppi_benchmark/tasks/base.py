"""Minimal contracts shared by PPI and future PTM task adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol, runtime_checkable

from ..backends.base import SupervisedSplit
from ..training import SplitEvaluation


def stable_task_data_signature(
        task_name: str, schema_version: int,
        records: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
    """Return an order-sensitive signature for task-owned examples."""
    canonical = json.dumps(
        {
            "task": task_name,
            "schema_version": schema_version,
            "records": records,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "task": task_name,
        "schema_version": schema_version,
        "n_examples": len(records),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


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

    def data_signature(self, examples: Any) -> Mapping[str, Any]:
        """Return the task-specific example identity used by checkpoints."""
        ...
