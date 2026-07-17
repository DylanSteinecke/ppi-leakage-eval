import json

import pandas as pd
import pytest

from ppi_benchmark.cli.train import argument_parser
from ppi_benchmark.artifact_io import file_sha256
from ppi_benchmark.splitting import grouping
from ppi_benchmark.splitting.grouping import (
    load_sequence_cluster_mapping,
    SequenceClusterParameters,
)


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


def test_c3_cli_uses_supplied_sequence_clusters_as_atomic_groups(
        tmp_path, run_train):
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

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--sequence-clusters", clusters_path,
        "--run-dir", run_dir,
        "--classifier", "always_positive",
        "--train-size", "0.50",
        "--val-size", "0.25",
        "--split-seed", "5",
        "--model-seeds", "5",
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
    assert audit["grouping_kind"] == "sequence_cluster"
    assert audit["n_groups_input"] == 9
    assert audit["n_shared_groups_train_val"] == 0
    assert audit["n_shared_groups_train_test"] == 0
    assert audit["n_shared_groups_val_test"] == 0
    assert audit["all_invariants_passed"] is True
    grouping_metadata = metadata["sequence_clusters"]
    assert grouping_metadata["applied_to_split"] is True
    assert grouping_metadata["grouping_kind"] == "sequence_cluster"
    assert grouping_metadata["grouping_source"] == "supplied_csv"
    assert grouping_metadata["method"] is None
    assert grouping_metadata["file_sha256"] == file_sha256(clusters_path)
    assert grouping_metadata["file_size_bytes"] == clusters_path.stat().st_size
    assert grouping_metadata["n_mapped_proteins"] == 18
    assert grouping_metadata["n_sequence_clusters"] == 9
    assert grouping_metadata["path"] == str(clusters_path)
    assert grouping_metadata["mapping_sha256"]
    assert metadata["sequence_cluster_assignments_path"] == str(
        snapshot_path)
    assert snapshot["protein_id"].tolist() == sorted(
        cluster_rows["protein_id"].tolist())


def test_c2_c3_cluster_mapping_must_cover_the_pre_sampling_cohort(
        tmp_path, ppi_test_data, run_train):
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

    with pytest.raises(ValueError, match="must map every eligible protein"):
        run_train(
            "--pairs", pairs_path,
            "--fasta", fasta_path,
            "--sequence-clusters", clusters_path,
            "--max-pairs", "12",
            "--run-dir", tmp_path / "missing_cluster",
            "--classifier", "always_positive",
            "--split-strategy", "c3",
            "--no-metrics-plots",
        )


@pytest.mark.parametrize("split_strategy", ["random", "c1"])
@pytest.mark.parametrize("source_option", ["supplied", "generated"])
def test_cluster_sources_are_rejected_for_incompatible_protocols(
    tmp_path,
    capsys,
    split_strategy,
    source_option,
):
    source_args = (
        ["--sequence-clusters", str(tmp_path / "clusters.csv")]
        if source_option == "supplied"
        else ["--sequence-cluster-method", "mmseqs2"]
    )
    with pytest.raises(SystemExit) as error:
        argument_parser([
            "--pairs", str(tmp_path / "pairs.csv"),
            "--fasta", str(tmp_path / "proteins.fasta"),
            "--run-dir", str(tmp_path / "run"),
            "--split-strategy", split_strategy,
            *source_args,
        ])
    assert error.value.code == 2
    assert "incompatible" in capsys.readouterr().err


def test_cluster_sources_are_rejected_for_provided_and_mutually_exclusive(
    tmp_path,
    capsys,
):
    common = [
        "--pairs", str(tmp_path / "pairs.csv"),
        "--fasta", str(tmp_path / "proteins.fasta"),
        "--run-dir", str(tmp_path / "run"),
    ]
    with pytest.raises(SystemExit):
        argument_parser([
            *common,
            "--split-col", "split",
            "--sequence-clusters", str(tmp_path / "clusters.csv"),
        ])
    assert "incompatible" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        argument_parser([
            *common,
            "--split-strategy", "c2",
            "--sequence-clusters", str(tmp_path / "clusters.csv"),
            "--sequence-cluster-method", "mmseqs2",
        ])
    assert "cannot be combined" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        argument_parser([
            *common,
            "--split-strategy", "c2",
            "--sequence-cluster-threads", "4",
        ])
    assert "requires --sequence-cluster-method" in capsys.readouterr().err


def test_automatic_cli_defaults_are_explicit_and_validated(tmp_path):
    args = argument_parser([
        "--pairs", str(tmp_path / "pairs.csv"),
        "--fasta", str(tmp_path / "proteins.fasta"),
        "--run-dir", str(tmp_path / "run"),
        "--split-strategy", "c2",
        "--sequence-cluster-method", "mmseqs2",
    ])

    assert args.sequence_cluster_min_seq_id == 0.30
    assert args.sequence_cluster_coverage == 0.80
    assert args.sequence_cluster_cov_mode == 0
    assert args.sequence_cluster_evalue == 0.001
    assert args.sequence_cluster_sensitivity is None
    assert args.sequence_cluster_cluster_mode is None
    assert args.sequence_cluster_threads == 1


def test_generated_c2_c3_share_cache_and_record_provenance(
    tmp_path,
    monkeypatch,
    run_train,
):
    n_proteins = 18
    pairs_path = tmp_path / "pairs.csv"
    fasta_path = tmp_path / "proteins.fasta"
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
            f">P{protein_index}\nACDEFGHIKLMNPQRSTV{protein_index % 10}\n"
            for protein_index in range(n_proteins)
        ),
        encoding="utf-8",
    )
    clustering_calls = []
    monkeypatch.setattr(grouping.shutil, "which", lambda _: "/tools/mmseqs")
    monkeypatch.setattr(grouping, "mmseqs_version", lambda _: "17-b804f")

    def fake_run(executable, sequences, parameters, work_dir):
        clustering_calls.append(set(sequences))
        rows = []
        protein_ids = sorted(
            sequences,
            key=lambda protein_id: int(protein_id.removeprefix("P")),
        )
        for start in range(0, len(protein_ids), 2):
            members = protein_ids[start:start + 2]
            cluster_id = grouping._stable_cluster_id(members)
            rows.extend(
                {"protein_id": protein_id, "cluster_id": cluster_id}
                for protein_id in members
            )
        assignments = pd.DataFrame(rows).sort_values("protein_id")
        command = grouping.build_mmseqs_command(
            executable,
            work_dir / "input.fasta",
            work_dir / "clusters",
            work_dir / "tmp",
            parameters,
        )
        return assignments, command, "", ""

    monkeypatch.setattr(grouping, "_run_mmseqs", fake_run)
    metadata_by_strategy = {}
    for split_strategy in ("c2", "c3"):
        run_dir = tmp_path / split_strategy
        run_train(
            "--pairs", pairs_path,
            "--fasta", fasta_path,
            "--sequence-cluster-method", "mmseqs2",
            "--sequence-cluster-cache-dir", tmp_path / "cache",
            "--sequence-cluster-threads", "2",
            "--max-pairs", "100",
            "--run-dir", run_dir,
            "--classifier", "always_positive",
            "--train-size", "0.50",
            "--val-size", "0.25",
            "--split-seed", "5",
            "--model-seeds", "5",
            "--split-strategy", split_strategy,
            "--n-split-trials", "100",
            "--no-metrics-plots",
        )
        metadata_by_strategy[split_strategy] = json.loads(
            (run_dir / "splits" / "split_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        snapshot = pd.read_csv(
            run_dir / "splits" / "sequence_cluster_assignments.csv"
        )
        assert snapshot["protein_id"].tolist() == sorted(
            snapshot["protein_id"].tolist()
        )
        audit = metadata_by_strategy[split_strategy]["split_audit"]
        assert audit["grouping_kind"] == "sequence_cluster"
        assert (
            audit["n_assigned_train_groups"]
            + audit["n_assigned_val_groups"]
            + audit["n_assigned_test_groups"]
        ) == audit["n_groups_input"]
        if split_strategy == "c3":
            assert audit["n_shared_groups_train_val"] == 0
            assert audit["n_shared_groups_train_test"] == 0
            assert audit["n_shared_groups_val_test"] == 0
        assert audit["all_invariants_passed"] is True

    assert len(clustering_calls) == 1
    assert clustering_calls[0] == {f"P{index}" for index in range(n_proteins)}
    c2_grouping = metadata_by_strategy["c2"]["sequence_clusters"]
    c3_grouping = metadata_by_strategy["c3"]["sequence_clusters"]
    assert c2_grouping["grouping_source"] == "generated"
    assert c2_grouping["method"] == "mmseqs2"
    assert c2_grouping["workflow"] == "easy-cluster"
    assert c2_grouping["tool_version"] == "17-b804f"
    assert c2_grouping["parameters"] == SequenceClusterParameters(
        threads=2
    ).to_dict()
    assert c2_grouping["cache_hit"] is False
    assert c3_grouping["cache_hit"] is True
    assert c2_grouping["cache_fingerprint"] == c3_grouping[
        "cache_fingerprint"
    ]
    assert "sequence-cluster-disjoint" in c2_grouping["scientific_warning"]
