"""Task adapters that connect examples to shared training infrastructure."""

from ..schema import (
    EVALUATION_SCHEMA_VERSION,
    STANDARD_PREDICTION_COLUMNS,
)
from .base import TaskAdapter
from .ppi import (
    PPI_TASK,
    PPIPairBatch,
    PPIPairCollator,
    PPIPairDataset,
    PPIPairExample,
    PPITask,
    SymmetricPairComposer,
)

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "PPI_TASK",
    "STANDARD_PREDICTION_COLUMNS",
    "PPIPairBatch",
    "PPIPairCollator",
    "PPIPairDataset",
    "PPIPairExample",
    "PPITask",
    "SymmetricPairComposer",
    "TaskAdapter",
]
