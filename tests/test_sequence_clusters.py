import json

import pandas as pd
import pytest

from ppi_benchmark.datasets.common import file_sha256
from ppi_benchmark.inputs import load_sequence_cluster_mapping


def test_sequence_cluster_loader_normalizes_and_rejects_duplicate_proteins(
        tmp_path):
    mapping_path = tmp_path / "clusters.csv"
    pd.DataFrame({
        "protein_id": [" P2 ", "P1"],
        "cluster_id": [" C1 ", "C0"],
        "ignored": [1, 2],
    }).to_csv(mapping_path, index=False)

    mapping, normalized, metadata = load_sequence_cluster_mapping(
        mapping_path,
        required_proteins={"P1", "P2"},
    )

    assert mapping == {"P2": "C1", "P1": "C0"}
    assert normalized.to_dict("records") == [
        {"protein_id": "P1", "cluster_id": "C0"},
        {"protein_id": "P2", "cluster_id": "C1"},
    ]
    assert metadata == {
        "n_mapped_proteins": 2,
        "n_sequence_clusters": 2,
    }

    duplicate_path = tmp_path / "duplicate_clusters.csv"
    pd.DataFrame({
        "protein_id": ["P1", "P1"],
        "cluster_id": ["C0", "C1"],
    }).to_csv(duplicate_path, index=False)
    with pytest.raises(ValueError, match="one row per protein"):
        load_sequence_cluster_mapping(duplicate_path)


def test_c3_cli_uses_sequence_clusters_as_atomic_homology_groups(
        tmp_path, run_cli):
    n_proteins = 18
    pairs_path = tmp_path / "pairs.csv"
    fasta_path = tmp_path / "proteins.fasta"
    clusters_path = tmp_path / "sequence_clusters.csv"
    run_dir = tmp_path / "clustered_c3"

    pair_rows = ["pair_id,protein_a,protein_b,label"]
    for left_index in range(n_proteins):
        for right_index in range(left_index + 1, n_proteins):
            pair_rows.append(
                f"pair_{left_index}_{right_index},P{left_index},"
                f"P{right_index},{(left_index + right_index) % 2}"
            )
    pairs_path.write_text("\n".join(pair_rows) + "\n", encoding="utf-8")
    fasta_path.write_text(
        "".join(
            f">P{protein_index}\nACDEFGHIKLMNPQRSTVWY\n"
            for protein_index in range(n_proteins)
        ),
        encoding="utf-8",
    )
    cluster_rows = pd.DataFrame({
        "protein_id": [
            f"P{protein_index}"
            for protein_index in reversed(range(n_proteins))
        ],
        "cluster_id": [
            f"cluster_{protein_index // 2}"
            for protein_index in reversed(range(n_proteins))
        ],
    })
    cluster_rows.to_csv(clusters_path, index=False)

    run_cli(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--sequence-clusters", clusters_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive",
        "--num-reruns", "1",
        "--train-size", "0.50",
        "--val-size", "0.25",
        "--split-seed", "5",
        "--split-strategy", "c3",
        "--n-split-trials", "100",
        "--no-metrics-plots",
    )

    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"))
    audit = metadata["split_audit"]
    snapshot_path = run_dir / "splits" / "sequence_cluster_assignments.csv"
    snapshot = pd.read_csv(snapshot_path, dtype="string")

    assert audit["grouping_type"] == "sequence_cluster"
    assert audit["n_groups_input"] == 9
    assert audit["n_shared_groups_train_val"] == 0
    assert audit["n_shared_groups_train_test"] == 0
    assert audit["n_shared_groups_val_test"] == 0
    assert audit["all_invariants_passed"] is True
    assert metadata["sequence_clusters"] == {
        "applied_to_split": True,
        "file_sha256": file_sha256(clusters_path),
        "file_size_bytes": clusters_path.stat().st_size,
        "n_mapped_proteins": 18,
        "n_sequence_clusters": 9,
        "path": str(clusters_path),
    }
    assert metadata["sequence_cluster_assignments_path"] == str(
        snapshot_path)
    assert snapshot["protein_id"].tolist() == sorted(
        cluster_rows["protein_id"].tolist())


def test_c2_c3_cluster_mapping_must_cover_the_pre_sampling_cohort(
        tmp_path, ppi_test_data, run_cli):
    pairs_path, fasta_path = ppi_test_data
    pairs = pd.read_csv(pairs_path)
    protein_ids = sorted(
        set(pairs["protein_a"]) | set(pairs["protein_b"])
    )
    clusters_path = tmp_path / "incomplete_clusters.csv"
    pd.DataFrame({
        "protein_id": protein_ids[:-1],
        "cluster_id": [
            f"cluster_{index // 2}"
            for index in range(len(protein_ids) - 1)
        ],
    }).to_csv(clusters_path, index=False)

    completed_process = run_cli(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--sequence-clusters", clusters_path,
        "--max-pairs", "12",
        "--run-dir", tmp_path / "missing_cluster",
        "--classifier", "always_positive",
        "--split-strategy", "c3",
        "--no-metrics-plots",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "must map every eligible protein" in completed_process.stderr

