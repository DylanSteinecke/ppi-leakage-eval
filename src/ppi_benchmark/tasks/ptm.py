"""PTM residue examples, protein-aware splitting, and aligned windows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backends.base import SupervisedSplit
from ..protein_encoders.representations import (
    ResidueAlignmentProvider,
    TokenizedProteinBatch,
)
from ..schema import EVALUATION_SCHEMA_VERSION
from ..training import SplitEvaluation
from .base import stable_task_data_signature


PTM_REQUIRED_COLUMNS = ("protein_id", "residue_index", "label")


@dataclass(frozen=True)
class PTMResidueExample:
    """One zero-based target residue and its supervised label."""

    protein_id: str
    residue_index: int
    target: int
    example_id: str
    ptm_type: str = "ptm"
    residue: str | None = None

    def __post_init__(self) -> None:
        if not self.protein_id:
            raise ValueError("PTM protein_id must not be empty.")
        if self.residue_index < 0:
            raise ValueError("PTM residue_index must be zero or greater.")
        if not self.example_id:
            raise ValueError("PTM example_id must not be empty.")
        if not self.ptm_type:
            raise ValueError("PTM ptm_type must not be empty.")
        if self.residue is not None and len(self.residue) != 1:
            raise ValueError("PTM residue must be one amino-acid character.")


class PTMResidueDataset(Sequence[PTMResidueExample]):
    """Immutable residue examples created before window collation."""

    def __init__(self, examples: Sequence[PTMResidueExample]):
        self.examples = tuple(examples)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame) -> "PTMResidueDataset":
        """Parse task examples without creating sequence windows."""
        missing = set(PTM_REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"PTM examples are missing columns: {sorted(missing)}")
        examples = []
        for position, row in frame.reset_index(drop=True).iterrows():
            protein_id = str(row["protein_id"])
            residue_index = int(row["residue_index"])
            ptm_type = (
                "ptm"
                if "ptm_type" not in frame.columns
                or pd.isna(row["ptm_type"])
                else str(row["ptm_type"])
            )
            example_id = (
                f"{protein_id}:{residue_index}:{ptm_type}"
                if "example_id" not in frame.columns
                or pd.isna(row["example_id"])
                else str(row["example_id"])
            )
            residue = (
                None
                if "residue" not in frame.columns or pd.isna(row["residue"])
                else str(row["residue"]).upper()
            )
            examples.append(PTMResidueExample(
                protein_id=protein_id,
                residue_index=residue_index,
                target=int(row["label"]),
                example_id=example_id,
                ptm_type=ptm_type,
                residue=residue,
            ))
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PTMResidueExample:
        return self.examples[index]


def split_ptm_examples_by_protein(
        examples: Sequence[PTMResidueExample],
        protein_to_split: Mapping[str, str],
    ) -> dict[str, PTMResidueDataset]:
    """Assign residues by protein before any windows are generated."""
    split_examples: dict[str, list[PTMResidueExample]] = {}
    missing_proteins = sorted({
        example.protein_id
        for example in examples
        if example.protein_id not in protein_to_split
    })
    if missing_proteins:
        raise ValueError(
            "PTM examples contain proteins without split assignments: "
            f"{missing_proteins[:10]}"
        )
    for example in examples:
        split_name = str(protein_to_split[example.protein_id]).strip()
        if not split_name:
            raise ValueError("PTM protein split names must not be empty.")
        split_examples.setdefault(split_name, []).append(example)
    return {
        split_name: PTMResidueDataset(rows)
        for split_name, rows in split_examples.items()
    }


@dataclass(frozen=True)
class PTMWindowBatch:
    """Tokenized windows with exact target-residue token positions."""

    example_ids: tuple[str, ...]
    protein_ids: tuple[str, ...]
    residue_indices: np.ndarray
    window_starts: np.ndarray
    window_sequences: tuple[str, ...]
    target_window_residue_indices: np.ndarray
    target_token_indices: np.ndarray
    targets: np.ndarray
    tokenized: TokenizedProteinBatch


class PTMWindowCollator:
    """Create bounded windows only after splitting, preserving alignment."""

    def __init__(
            self, protein_sequences: Mapping[str, str], window_radius: int,
            alignment_provider: ResidueAlignmentProvider,
        ):
        if window_radius < 0:
            raise ValueError("PTM window_radius must be nonnegative.")
        self.protein_sequences = protein_sequences
        self.window_radius = window_radius
        self.alignment_provider = alignment_provider

    def __call__(self, examples: Sequence[PTMResidueExample]) -> PTMWindowBatch:
        if not examples:
            raise ValueError("Cannot collate an empty PTM batch.")
        window_sequences = []
        window_starts = []
        target_window_indices = []
        for example in examples:
            if example.protein_id not in self.protein_sequences:
                raise ValueError(
                    f"Missing sequence for PTM protein {example.protein_id!r}."
                )
            sequence = "".join(
                str(self.protein_sequences[example.protein_id]).split()
            ).upper()
            if example.residue_index >= len(sequence):
                raise ValueError(
                    f"PTM residue index {example.residue_index} is outside "
                    f"protein {example.protein_id!r}."
                )
            observed_residue = sequence[example.residue_index]
            if example.residue is not None and observed_residue != example.residue:
                raise ValueError(
                    f"PTM residue mismatch for {example.example_id}: expected "
                    f"{example.residue}, found {observed_residue}."
                )
            start = max(0, example.residue_index - self.window_radius)
            stop = min(
                len(sequence),
                example.residue_index + self.window_radius + 1,
            )
            window_sequences.append(sequence[start:stop])
            window_starts.append(start)
            target_window_indices.append(example.residue_index - start)

        tokenized = self.alignment_provider.tokenize_with_alignment(
            window_sequences
        )
        if tokenized.sequences != tuple(window_sequences):
            raise ValueError(
                "Alignment provider changed PTM window sequence contents."
            )
        expected_counts = np.asarray(
            [len(sequence) for sequence in window_sequences],
            dtype=np.int64,
        )
        if not np.array_equal(
            tokenized.alignment.residue_counts,
            expected_counts,
        ):
            raise ValueError(
                "PTM windows were truncated or do not map one-to-one to "
                "residue tokens."
            )
        target_token_indices = np.asarray([
            tokenized.alignment.token_index(batch_index, residue_index)
            for batch_index, residue_index in enumerate(target_window_indices)
        ], dtype=np.int64)
        return PTMWindowBatch(
            example_ids=tuple(example.example_id for example in examples),
            protein_ids=tuple(example.protein_id for example in examples),
            residue_indices=np.asarray([
                example.residue_index for example in examples
            ], dtype=np.int64),
            window_starts=np.asarray(window_starts, dtype=np.int64),
            window_sequences=tuple(window_sequences),
            target_window_residue_indices=np.asarray(
                target_window_indices,
                dtype=np.int64,
            ),
            target_token_indices=target_token_indices,
            targets=np.asarray([
                example.target for example in examples
            ], dtype=np.int64),
            tokenized=tokenized,
        )


class PTMTask:
    """Connect residue-level PTM examples to shared evaluation outputs."""

    name = "ptm"

    @staticmethod
    def _dataset(examples: pd.DataFrame) -> PTMResidueDataset:
        return PTMResidueDataset.from_dataframe(examples)

    def make_split(
            self, name: str, examples: pd.DataFrame, inputs: Any,
        ) -> SupervisedSplit:
        dataset = self._dataset(examples)
        n_inputs = inputs.shape[0] if hasattr(inputs, "shape") else len(inputs)
        if n_inputs != len(dataset):
            raise ValueError(
                f"PTM {name} examples and backend inputs have different lengths."
            )
        return SupervisedSplit(
            name=name,
            inputs=inputs,
            targets=np.asarray([
                example.target for example in dataset
            ], dtype=np.int64),
        )

    def prediction_frame(
            self, examples: pd.DataFrame, evaluation: SplitEvaluation,
            model_metadata: Mapping[str, Any],
        ) -> pd.DataFrame:
        dataset = self._dataset(examples)
        lengths = {
            len(dataset),
            len(evaluation.targets),
            len(evaluation.scores),
            len(evaluation.predictions),
        }
        if len(lengths) != 1:
            raise ValueError(
                "PTM examples and evaluated predictions have different lengths."
            )
        frame = pd.DataFrame({
            "protein_id": [example.protein_id for example in dataset],
            "residue_index": [example.residue_index for example in dataset],
            "residue": [example.residue for example in dataset],
            "ptm_type": [example.ptm_type for example in dataset],
        })
        for position, (column, value) in enumerate(model_metadata.items()):
            frame.insert(position, column, value)
        frame["evaluation_schema_version"] = EVALUATION_SCHEMA_VERSION
        frame["task"] = self.name
        frame["split"] = evaluation.name
        frame["example_id"] = [example.example_id for example in dataset]
        frame["target"] = evaluation.targets
        frame["score"] = evaluation.scores
        frame["prediction"] = evaluation.predictions
        return frame

    def data_signature(self, examples: pd.DataFrame) -> dict[str, Any]:
        dataset = self._dataset(examples)
        records = [
            {
                "example_id": example.example_id,
                "protein_id": example.protein_id,
                "residue_index": example.residue_index,
                "target": example.target,
                "ptm_type": example.ptm_type,
                "residue": example.residue,
            }
            for example in dataset
        ]
        return stable_task_data_signature(
            task_name=self.name,
            schema_version=EVALUATION_SCHEMA_VERSION,
            records=records,
        )


PTM_TASK = PTMTask()
