"""Task adapters that connect examples to shared training infrastructure."""

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
from .ptm import (
    PTM_TASK,
    PTMResidueDataset,
    PTMResidueExample,
    PTMTask,
    PTMWindowBatch,
    PTMWindowCollator,
    split_ptm_examples_by_protein,
)

__all__ = [
    "PPI_TASK",
    "PPIPairBatch",
    "PPIPairCollator",
    "PPIPairDataset",
    "PPIPairExample",
    "PPITask",
    "PTM_TASK",
    "PTMResidueDataset",
    "PTMResidueExample",
    "PTMTask",
    "PTMWindowBatch",
    "PTMWindowCollator",
    "SymmetricPairComposer",
    "TaskAdapter",
    "split_ptm_examples_by_protein",
]


def __getattr__(name: str):
    """Load optional Torch heads only when explicitly requested."""
    if name in {"PPIPairHead", "TorchSymmetricPairComposer"}:
        from .ppi_torch import PPIPairHead, TorchSymmetricPairComposer

        return {
            "PPIPairHead": PPIPairHead,
            "TorchSymmetricPairComposer": TorchSymmetricPairComposer,
        }[name]
    if name == "PTMResidueHead":
        from .ptm_torch import PTMResidueHead

        return PTMResidueHead
    raise AttributeError(name)


__all__.extend([
    "PPIPairHead",
    "PTMResidueHead",
    "TorchSymmetricPairComposer",
])
