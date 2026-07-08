import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from ppi_inputs import prepare_input_data
from ppi_splits import (
    add_source_row_index,
    compute_split_metadata,
    make_split_assignments,
    write_split_artifacts,
)


def test_source_row_index_and_drop_reasons_are_preserved():
    pairs = pd.DataFrame({
        "pair_id": ["p0", "p1", "p2", "p3", "p4"],
        "protein_a": ["A", "A", "X", "B", "X"],
        "protein_b": ["B", "C", "A", "Y", "Y"],
        "label": [1, 0, 0, 1, 0],
    })
    sequences = {
        "A": "AAAA",
        "B": "BBBB",
        "C": "CCCC",
    }

    indexed_pairs = add_source_row_index(pairs)
    prepared_pairs, dropped_pairs = prepare_input_data(
        indexed_pairs,
        sequences,
    )

    assert prepared_pairs["source_row_index"].tolist() == [0, 1]
    assert dropped_pairs.columns.tolist() == [
        "source_row_index",
        "drop_reason",
        "pair_id",
    ]
    assert dropped_pairs["source_row_index"].tolist() == [2, 3, 4]
    assert dropped_pairs["drop_reason"].tolist() == [
        "missing_protein_a_sequence",
        "missing_protein_b_sequence",
        "missing_both_sequences",
    ]


def test_add_source_row_index_rejects_existing_reserved_column():
    pairs = pd.DataFrame({
        "source_row_index": [10],
        "protein_a": ["A"],
        "protein_b": ["B"],
        "label": [1],
    })

    with pytest.raises(ValueError, match="already contains source_row_index"):
        add_source_row_index(pairs)


def test_split_assignments_are_minimal_with_optional_pair_id():
    split_df = pd.DataFrame({
        "source_row_index": [2, 0],
        "pair_id": ["p2", "p0"],
        "protein_a": ["A", "B"],
        "protein_b": ["C", "D"],
        "label": [1, 0],
    })

    assignments = make_split_assignments(
        train_df=split_df.iloc[[0]],
        val_df=None,
        test_df=split_df.iloc[[1]],
    )

    assert assignments.columns.tolist() == [
        "source_row_index",
        "split",
        "pair_id",
    ]
    assert assignments["source_row_index"].tolist() == [0, 2]


def test_split_assignments_are_minimal_without_pair_id():
    split_df = pd.DataFrame({
        "source_row_index": [0, 1],
        "protein_a": ["A", "B"],
        "protein_b": ["C", "D"],
        "label": [1, 0],
    })

    assignments = make_split_assignments(
        train_df=split_df.iloc[[0]],
        val_df=None,
        test_df=split_df.iloc[[1]],
    )

    assert assignments.columns.tolist() == ["source_row_index", "split"]


def test_metadata_contains_required_audit_fields(tmp_path):
    pairs_path = tmp_path / "pairs.csv"
    fasta_path = tmp_path / "seqs.fasta"
    pairs_path.write_text(
        "protein_a,protein_b,label\nA,B,1\nA,C,0\n",
        encoding="utf-8",
    )
    fasta_path.write_text(
        ">A\nAAAA\n>B\nBBBB\n>C\nCCCC\n",
        encoding="utf-8",
    )
    split_dir = tmp_path / "run" / "splits"
    output_paths = SimpleNamespace(
        run_dir=tmp_path / "run",
        train_metrics_path=tmp_path / "run" / "train_metrics.csv",
        val_metrics_path=tmp_path / "run" / "val_metrics.csv",
        test_metrics_path=tmp_path / "run" / "test_metrics.csv",
        predictions_path=tmp_path / "run" / "predictions.csv",
        split_assignments_path=split_dir / "split_assignments.csv",
        dropped_pairs_path=split_dir / "dropped_pairs.csv",
        split_metadata_path=split_dir / "split_metadata.json",
    )
    args = argparse.Namespace(
        pairs=str(pairs_path),
        fasta=str(fasta_path),
        train_size=0.5,
        val_size=0.0,
        effective_split_strategy="random",
        split_name=None,
        split_col=None,
        seed=7,
        evaluate_test_metrics=True,
        n_connected_components=None,
        n_pruned_pairs=0,
        pruned_pair_fraction=0.0,
    )
    pairs = pd.DataFrame({
        "source_row_index": [0, 1],
        "protein_a": ["A", "A"],
        "protein_b": ["B", "C"],
        "label": [1, 0],
    })

    metadata = compute_split_metadata(
        args=args,
        output_paths=output_paths,
        protein_pairs=pairs,
        dropped_pairs=pd.DataFrame(
            columns=["source_row_index", "drop_reason"]),
        train_df=pairs.iloc[[0]],
        val_df=None,
        test_df=pairs.iloc[[1]],
        execution_id="test-execution",
        n_input_pairs_before_filtering=2,
    )

    required_keys = {
        "execution_id",
        "timestamp_utc",
        "run_dir",
        "command",
        "argv",
        "python_executable",
        "working_directory",
        "pairs_path",
        "fasta_path",
        "split_strategy",
        "split_seed",
        "target_train_size",
        "target_val_size",
        "target_test_size",
        "n_train",
        "n_val",
        "n_test",
        "actual_train_size",
        "actual_val_size",
        "actual_test_size",
        "label_counts_train",
        "label_counts_val",
        "label_counts_test",
        "n_shared_proteins_train_val",
        "n_shared_proteins_train_test",
        "n_shared_proteins_val_test",
        "pairs_file_sha256",
        "fasta_file_sha256",
    }
    assert required_keys <= set(metadata)
    assert metadata["execution_id"] == "test-execution"
    assert metadata["n_val"] == 0
    assert metadata["actual_val_size"] == 0.0


def test_append_rejects_different_split_assignments(tmp_path):
    split_dir = tmp_path / "splits"
    output_paths = SimpleNamespace(
        split_assignments_path=split_dir / "split_assignments.csv",
        dropped_pairs_path=split_dir / "dropped_pairs.csv",
        split_metadata_path=split_dir / "split_metadata.json",
    )
    assignments = pd.DataFrame({
        "source_row_index": [0, 1],
        "split": ["train", "test"],
    })
    dropped_pairs = pd.DataFrame(
        columns=["source_row_index", "drop_reason"])
    metadata = {"execution_id": "test-execution"}

    write_split_artifacts(
        split_assignments=assignments,
        dropped_pairs=dropped_pairs,
        split_metadata=metadata,
        output_paths=output_paths,
        append_results=False,
    )
    write_split_artifacts(
        split_assignments=assignments.copy(),
        dropped_pairs=dropped_pairs,
        split_metadata=metadata,
        output_paths=output_paths,
        append_results=True,
    )

    changed_assignments = assignments.copy()
    changed_assignments.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="different split"):
        write_split_artifacts(
            split_assignments=changed_assignments,
            dropped_pairs=dropped_pairs,
            split_metadata=metadata,
            output_paths=output_paths,
            append_results=True,
        )
