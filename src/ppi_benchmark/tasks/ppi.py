"""PPI examples, collation, prediction formatting, and pair composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack

from ..backends.base import SupervisedSplit
from ..schema import EVALUATION_SCHEMA_VERSION
from ..training import SplitEvaluation
from .base import stable_task_data_signature


PPI_REQUIRED_COLUMNS = ("protein_a", "protein_b", "label")
DEFAULT_PAIR_COMPOSITION_CHUNK_SIZE = 8192


@dataclass(frozen=True)
class PPIPairExample:
    """One labeled protein-pair example."""

    example_id: str
    protein_a: str
    protein_b: str
    target: int


class PPIPairDataset(Sequence[PPIPairExample]):
    """A lightweight dataframe view for future batched PPI models."""

    def __init__(self, examples: pd.DataFrame):
        missing = set(PPI_REQUIRED_COLUMNS) - set(examples.columns)
        if missing:
            raise ValueError(f"PPI examples are missing columns: {sorted(missing)}")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PPIPairExample:
        row = self.examples.iloc[index]
        if "pair_id" in self.examples.columns and pd.notna(row["pair_id"]):
            example_id = str(row["pair_id"])
        elif "source_row_index" in self.examples.columns:
            example_id = str(row["source_row_index"])
        else:
            example_id = str(index)
        return PPIPairExample(
            example_id=example_id,
            protein_a=str(row["protein_a"]),
            protein_b=str(row["protein_b"]),
            target=int(row["label"]),
        )


@dataclass(frozen=True)
class PPIPairBatch:
    """Pairs indexed into the unique proteins encoded for one batch."""

    example_ids: tuple[str, ...]
    protein_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    protein_a_indices: np.ndarray
    protein_b_indices: np.ndarray
    targets: np.ndarray


class PPIPairCollator:
    """Deduplicate proteins before a shared encoder processes a pair batch."""

    def __init__(self, protein_sequences: Mapping[str, str]):
        self.protein_sequences = protein_sequences

    def __call__(self, examples: Sequence[PPIPairExample]) -> PPIPairBatch:
        if not examples:
            raise ValueError("Cannot collate an empty PPI batch.")
        protein_ids = tuple(dict.fromkeys(
            protein_id
            for example in examples
            for protein_id in (example.protein_a, example.protein_b)
        ))
        missing = [
            protein_id
            for protein_id in protein_ids
            if protein_id not in self.protein_sequences
        ]
        if missing:
            raise ValueError(
                f"PPI batch contains proteins without sequences: {missing[:10]}"
            )
        index_by_protein = {
            protein_id: index
            for index, protein_id in enumerate(protein_ids)
        }
        return PPIPairBatch(
            example_ids=tuple(example.example_id for example in examples),
            protein_ids=protein_ids,
            sequences=tuple(
                self.protein_sequences[protein_id]
                for protein_id in protein_ids
            ),
            protein_a_indices=np.asarray([
                index_by_protein[example.protein_a]
                for example in examples
            ], dtype=np.int64),
            protein_b_indices=np.asarray([
                index_by_protein[example.protein_b]
                for example in examples
            ], dtype=np.int64),
            targets=np.asarray([
                example.target for example in examples
            ], dtype=np.int64),
        )


@dataclass(frozen=True)
class SymmetricPairComposer:
    """Compose sum, absolute-difference, and product pair features."""

    chunk_size: int = DEFAULT_PAIR_COMPOSITION_CHUNK_SIZE

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("Pair composition chunk_size must be positive.")

    def compose(
            self, examples: pd.DataFrame, protein_ids: pd.Index,
            protein_features: Any,
        ) -> Any:
        """Return symmetric features without full dense intermediates."""
        protein_a_rows = protein_ids.get_indexer(examples["protein_a"])
        protein_b_rows = protein_ids.get_indexer(examples["protein_b"])
        if (protein_a_rows < 0).any() or (protein_b_rows < 0).any():
            raise ValueError("Pair dataframe contains proteins without features.")

        if sparse.issparse(protein_features):
            protein_a = protein_features[protein_a_rows]
            protein_b = protein_features[protein_b_rows]
            return hstack([
                protein_a + protein_b,
                np.abs(protein_a - protein_b),
                protein_a.multiply(protein_b),
            ], format="csr")

        dense_features = np.asarray(protein_features)
        if dense_features.ndim != 2:
            raise ValueError(
                "Protein features must be a two-dimensional matrix."
            )
        n_pairs = len(examples)
        n_features = dense_features.shape[1]
        pair_features = np.empty(
            (n_pairs, n_features * 3),
            dtype=dense_features.dtype,
        )
        for start in range(0, n_pairs, self.chunk_size):
            stop = min(start + self.chunk_size, n_pairs)
            protein_a = dense_features[protein_a_rows[start:stop]]
            protein_b = dense_features[protein_b_rows[start:stop]]
            output = pair_features[start:stop]
            np.add(protein_a, protein_b, out=output[:, :n_features])
            np.subtract(
                protein_a,
                protein_b,
                out=output[:, n_features:2 * n_features],
            )
            np.abs(
                output[:, n_features:2 * n_features],
                out=output[:, n_features:2 * n_features],
            )
            np.multiply(
                protein_a,
                protein_b,
                out=output[:, 2 * n_features:],
            )
        return pair_features


class PPITask:
    """Connect PPI dataframes to shared training and evaluation."""

    name = "ppi"

    @staticmethod
    def _validate_examples(examples: pd.DataFrame) -> None:
        missing = set(PPI_REQUIRED_COLUMNS) - set(examples.columns)
        if missing:
            raise ValueError(f"PPI examples are missing columns: {sorted(missing)}")

    def make_split(
            self, name: str, examples: pd.DataFrame, inputs: Any,
        ) -> SupervisedSplit:
        """Extract PPI labels while leaving backend inputs unrestricted."""
        self._validate_examples(examples)
        if hasattr(inputs, "shape"):
            n_inputs = inputs.shape[0]
        else:
            n_inputs = len(inputs)
        if n_inputs != len(examples):
            raise ValueError(
                f"PPI {name} examples and backend inputs have different "
                "lengths."
            )
        return SupervisedSplit(
            name=name,
            inputs=inputs,
            targets=examples["label"].to_numpy(),
        )

    def prediction_frame(
            self, examples: pd.DataFrame, evaluation: SplitEvaluation,
            model_metadata: Mapping[str, Any],
        ) -> pd.DataFrame:
        """Return the legacy PPI fields plus a stable common schema."""
        self._validate_examples(examples)
        evaluated_lengths = {
            len(evaluation.targets),
            len(evaluation.scores),
            len(evaluation.predictions),
        }
        if evaluated_lengths != {len(examples)}:
            raise ValueError(
                "PPI examples and evaluated predictions have different lengths."
            )
        base_columns = [
            column
            for column in (
                "source_row_index",
                "protein_a",
                "protein_b",
                "label",
            )
            if column in examples.columns
        ]
        frame = examples[base_columns].copy().reset_index(drop=True)
        metadata_items = list(model_metadata.items())
        # Preserve source_row_index as the fifth legacy prediction column.
        legacy_prefix = metadata_items[:4]
        legacy_suffix = metadata_items[4:]
        for position, (column, value) in enumerate(legacy_prefix):
            frame.insert(position, column, value)
        suffix_position = len(legacy_prefix) + int(
            "source_row_index" in frame.columns
        )
        for offset, (column, value) in enumerate(legacy_suffix):
            frame.insert(suffix_position + offset, column, value)
        frame["pred_score"] = evaluation.scores
        frame["pred_label"] = evaluation.predictions

        dataset = PPIPairDataset(examples)
        frame["evaluation_schema_version"] = EVALUATION_SCHEMA_VERSION
        frame["task"] = self.name
        frame["split"] = evaluation.name
        frame["example_id"] = [
            dataset[index].example_id
            for index in range(len(dataset))
        ]
        frame["target"] = evaluation.targets
        frame["score"] = evaluation.scores
        frame["prediction"] = evaluation.predictions
        return frame

    def data_signature(self, examples: pd.DataFrame) -> dict[str, Any]:
        """Hash ordered PPI identities, endpoints, and labels."""
        self._validate_examples(examples)
        dataset = PPIPairDataset(examples)
        records = [
            {
                "example_id": example.example_id,
                "protein_a": example.protein_a,
                "protein_b": example.protein_b,
                "target": example.target,
            }
            for example in dataset
        ]
        return stable_task_data_signature(
            task_name=self.name,
            schema_version=EVALUATION_SCHEMA_VERSION,
            records=records,
        )


PPI_TASK = PPITask()
