from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import pandas as pd
import pytest

from ppi_benchmark.splitting import grouping
from ppi_benchmark.splitting.grouping import (
    SEQUENCE_CLUSTER_ASSIGNMENTS_FILENAME,
    SEQUENCE_CLUSTER_METADATA_FILENAME,
    SequenceClusterParameters,
    build_mmseqs_command,
    default_sequence_cluster_cache_dir,
    parse_mmseqs_cluster_output,
    resolve_sequence_clusters,
)


def _assignments(sequences):
    rows = []
    members_by_sequence = {}
    for protein_id, sequence in sequences.items():
        members_by_sequence.setdefault(sequence, []).append(protein_id)
    for members in members_by_sequence.values():
        cluster_id = grouping._stable_cluster_id(members)
        rows.extend(
            {"protein_id": protein_id, "cluster_id": cluster_id}
            for protein_id in members
        )
    return pd.DataFrame(rows).sort_values("protein_id").reset_index(drop=True)


def _install_fake_mmseqs(monkeypatch, *, delay=0.0):
    calls = []
    calls_lock = threading.Lock()

    monkeypatch.setattr(grouping.shutil, "which", lambda _: "/tools/mmseqs")
    monkeypatch.setattr(grouping, "mmseqs_version", lambda _: "17-b804f")

    def fake_run(executable, sequences, parameters, work_dir):
        with calls_lock:
            calls.append((dict(sequences), parameters))
        if delay:
            time.sleep(delay)
        command = build_mmseqs_command(
            executable,
            work_dir / "input.fasta",
            work_dir / "clusters",
            work_dir / "tmp",
            parameters,
        )
        return _assignments(sequences), command, "", ""

    monkeypatch.setattr(grouping, "_run_mmseqs", fake_run)
    return calls


def test_sequence_cluster_parameters_have_validated_public_defaults():
    parameters = SequenceClusterParameters()

    assert parameters.to_dict() == {
        "method": "mmseqs2",
        "workflow": "easy-cluster",
        "min_seq_id": 0.30,
        "coverage": 0.80,
        "cov_mode": 0,
        "evalue": 0.001,
        "sensitivity": None,
        "sensitivity_mode": "automatic",
        "cluster_mode": None,
        "cluster_mode_setting": "automatic",
        "threads": 1,
    }
    with pytest.raises(ValueError, match="between 0 and 1"):
        SequenceClusterParameters(min_seq_id=1.1)
    with pytest.raises(ValueError, match="between 0 and 5"):
        SequenceClusterParameters(cov_mode=6)
    with pytest.raises(ValueError, match="at least 1"):
        SequenceClusterParameters(threads=0)


def test_mmseqs_command_omits_auto_settings_and_forwards_explicit_values():
    automatic = build_mmseqs_command(
        "/tools/mmseqs",
        "proteins.fasta",
        "clusters",
        "tmp",
        SequenceClusterParameters(),
    )
    explicit = build_mmseqs_command(
        "/tools/mmseqs",
        "proteins.fasta",
        "clusters",
        "tmp",
        SequenceClusterParameters(
            sensitivity=7.5,
            cluster_mode=1,
            threads=8,
        ),
    )

    assert automatic[:5] == [
        "/tools/mmseqs", "easy-cluster", "proteins.fasta", "clusters", "tmp"
    ]
    assert "-s" not in automatic
    assert "--cluster-mode" not in automatic
    assert explicit[explicit.index("-s") + 1] == "7.5"
    assert explicit[explicit.index("--cluster-mode") + 1] == "1"
    assert explicit[explicit.index("--threads") + 1] == "8"


def test_mmseqs_parser_is_stable_and_rejects_invalid_membership(tmp_path):
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    first.write_text("P2\tP2\nP2\tP1\nP3\tP3\n", encoding="utf-8")
    second.write_text("P3\tP3\nP2\tP1\nP2\tP2\n", encoding="utf-8")

    parsed_first = parse_mmseqs_cluster_output(first, {"P1", "P2", "P3"})
    parsed_second = parse_mmseqs_cluster_output(second, {"P1", "P2", "P3"})

    pd.testing.assert_frame_equal(parsed_first, parsed_second)
    assert parsed_first["protein_id"].tolist() == ["P1", "P2", "P3"]
    assert all(
        cluster_id.startswith("sequence_cluster_")
        and len(cluster_id) == len("sequence_cluster_") + 64
        for cluster_id in parsed_first["cluster_id"]
    )

    invalid_cases = {
        "malformed.tsv": "P1 P1\n",
        "unknown.tsv": "P1\tP1\nP2\tUNKNOWN\nP3\tP3\n",
        "duplicate.tsv": "P1\tP1\nP1\tP2\nP3\tP2\nP3\tP3\n",
        "missing.tsv": "P1\tP1\nP2\tP2\n",
        "bad_representative.tsv": "P1\tP2\nP2\tP1\nP3\tP3\n",
    }
    for filename, contents in invalid_cases.items():
        invalid_path = tmp_path / filename
        invalid_path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError):
            parse_mmseqs_cluster_output(
                invalid_path,
                {"P1", "P2", "P3"},
            )


def test_mmseqs_version_records_output_and_wraps_failures(monkeypatch):
    monkeypatch.setattr(
        grouping.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="17-b804f\n", stderr=""
        ),
    )
    assert grouping.mmseqs_version("/tools/mmseqs") == "17-b804f"

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr="version failed",
        )

    monkeypatch.setattr(grouping.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Could not determine"):
        grouping.mmseqs_version("/tools/mmseqs")


def test_mmseqs_execution_writes_canonical_fasta_and_validates_output(
    monkeypatch,
    tmp_path,
):
    observed = {}

    def succeed(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        output_path = Path(f"{command[3]}_cluster.tsv")
        output_path.write_text("P1\tP1\nP2\tP2\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="tool stdout",
            stderr="tool stderr",
        )

    monkeypatch.setenv("MMSEQS_NUM_THREADS", "99")
    monkeypatch.setattr(grouping.subprocess, "run", succeed)
    work_dir = tmp_path / "success"
    work_dir.mkdir()
    assignments, command, stdout, stderr = grouping._run_mmseqs(
        "/tools/mmseqs",
        {"P1": "AAAA", "P2": "CCCC"},
        SequenceClusterParameters(),
        work_dir,
    )

    assert command == observed["command"]
    assert "MMSEQS_NUM_THREADS" not in observed["environment"]
    assert (work_dir / "eligible_proteins.fasta").read_text(
        encoding="utf-8"
    ) == ">P1\nAAAA\n>P2\nCCCC\n"
    assert assignments["protein_id"].tolist() == ["P1", "P2"]
    assert stdout == "tool stdout"
    assert stderr == "tool stderr"

    monkeypatch.setattr(
        grouping.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )
    missing_output_dir = tmp_path / "missing-output"
    missing_output_dir.mkdir()
    with pytest.raises(RuntimeError, match="without producing"):
        grouping._run_mmseqs(
            "/tools/mmseqs",
            {"P1": "AAAA"},
            SequenceClusterParameters(),
            missing_output_dir,
        )


def test_default_cache_directory_precedence(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("PPI_SEQUENCE_CLUSTER_CACHE_DIR", str(explicit))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert default_sequence_cluster_cache_dir() == explicit

    monkeypatch.delenv("PPI_SEQUENCE_CLUSTER_CACHE_DIR")
    assert default_sequence_cluster_cache_dir() == (
        xdg / "ppi-leakage" / "sequence_clusters"
    )


def test_generated_grouping_caches_validated_artifacts_and_invalidates(
    monkeypatch,
    tmp_path,
):
    calls = _install_fake_mmseqs(monkeypatch)
    sequences = {"P2": "AAAA", "P1": "AAAA", "P3": "CCCC"}
    cache_dir = tmp_path / "cache"

    first = resolve_sequence_clusters(
        sequences,
        {"P1", "P2", "P3"},
        parameters=SequenceClusterParameters(),
        cache_dir=cache_dir,
    )
    second = resolve_sequence_clusters(
        sequences,
        {"P3", "P2", "P1"},
        parameters=SequenceClusterParameters(),
        cache_dir=cache_dir,
    )

    assert first is not None and second is not None
    assert len(calls) == 1
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert first.metadata["cache_fingerprint"] == second.metadata[
        "cache_fingerprint"
    ]
    assert first.metadata["tool_version"] == "17-b804f"
    assert first.metadata["grouping_source"] == "generated"
    assert first.metadata["grouping_kind"] == "sequence_cluster"
    assert first.metadata["mapping_sha256"]
    assert "eligible_protein_ids" not in first.metadata
    cache_entry = Path(first.metadata["cache_path"])
    assert {path.name for path in cache_entry.iterdir()} == {
        SEQUENCE_CLUSTER_ASSIGNMENTS_FILENAME,
        SEQUENCE_CLUSTER_METADATA_FILENAME,
    }
    cache_metadata = json.loads(
        (cache_entry / SEQUENCE_CLUSTER_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert cache_metadata["eligible_protein_ids"] == ["P1", "P2", "P3"]

    changed = resolve_sequence_clusters(
        sequences,
        {"P1", "P2", "P3"},
        parameters=SequenceClusterParameters(threads=2),
        cache_dir=cache_dir,
    )
    assert changed is not None
    assert len(calls) == 2
    assert changed.metadata["cache_fingerprint"] != first.metadata[
        "cache_fingerprint"
    ]

    changed_sequence = resolve_sequence_clusters(
        {**sequences, "P3": "DDDD"},
        {"P1", "P2", "P3"},
        parameters=SequenceClusterParameters(),
        cache_dir=cache_dir,
    )
    changed_membership = resolve_sequence_clusters(
        sequences,
        {"P1", "P2"},
        parameters=SequenceClusterParameters(),
        cache_dir=cache_dir,
    )
    assert changed_sequence is not None and changed_membership is not None
    assert len(calls) == 4
    assert changed_sequence.metadata["cache_fingerprint"] != first.metadata[
        "cache_fingerprint"
    ]
    assert changed_membership.metadata["cache_fingerprint"] != first.metadata[
        "cache_fingerprint"
    ]


def test_every_effective_parameter_changes_the_cache_fingerprint():
    base = SequenceClusterParameters()
    base_fingerprint = grouping._cache_fingerprint(
        "universe", "version", base
    )
    variants = (
        replace(base, min_seq_id=0.31),
        replace(base, coverage=0.79),
        replace(base, cov_mode=1),
        replace(base, evalue=0.002),
        replace(base, sensitivity=5.7),
        replace(base, cluster_mode=1),
        replace(base, threads=2),
    )
    assert all(
        grouping._cache_fingerprint("universe", "version", variant)
        != base_fingerprint
        for variant in variants
    )
    assert grouping._cache_fingerprint(
        "universe", "other-version", base
    ) != base_fingerprint


def test_corrupt_cache_entry_is_rejected_without_recomputation(
    monkeypatch,
    tmp_path,
):
    calls = _install_fake_mmseqs(monkeypatch)
    result = resolve_sequence_clusters(
        {"P1": "AAAA", "P2": "CCCC"},
        {"P1", "P2"},
        parameters=SequenceClusterParameters(),
        cache_dir=tmp_path / "cache",
    )
    assert result is not None
    assignments_path = Path(result.metadata["path"])
    assignments_path.write_text("protein_id,cluster_id\nP1,broken\n")

    with pytest.raises(ValueError, match="Corrupt|mismatch"):
        resolve_sequence_clusters(
            {"P1": "AAAA", "P2": "CCCC"},
            {"P1", "P2"},
            parameters=SequenceClusterParameters(),
            cache_dir=tmp_path / "cache",
        )
    assert len(calls) == 1


def test_concurrent_resolvers_publish_one_complete_cache_entry(
    monkeypatch,
    tmp_path,
):
    calls = _install_fake_mmseqs(monkeypatch, delay=0.1)

    def resolve():
        return resolve_sequence_clusters(
            {"P1": "AAAA", "P2": "CCCC"},
            {"P1", "P2"},
            parameters=SequenceClusterParameters(),
            cache_dir=tmp_path / "cache",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve(), range(2)))

    assert len(calls) == 1
    assert {result.metadata["cache_hit"] for result in results} == {
        False,
        True,
    }
    assert results[0].protein_to_group == results[1].protein_to_group


def test_generated_resolver_validates_sequences_and_executable(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(ValueError, match="missing a sequence"):
        resolve_sequence_clusters(
            {"P1": "AAAA"},
            {"P1", "P2"},
            parameters=SequenceClusterParameters(),
            cache_dir=tmp_path,
        )

    monkeypatch.setattr(grouping.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires the 'mmseqs' executable"):
        resolve_sequence_clusters(
            {"P1": "AAAA"},
            {"P1"},
            parameters=SequenceClusterParameters(),
            cache_dir=tmp_path,
        )


@pytest.mark.external_tool
@pytest.mark.skipif(shutil.which("mmseqs") is None, reason="MMseqs2 unavailable")
def test_real_mmseqs_easy_cluster_smoke(tmp_path):
    result = resolve_sequence_clusters(
        {
            "P1": "ACDEFGHIKLMNPQRSTVWY",
            "P2": "ACDEFGHIKLMNPQRSTVWY",
            "P3": "YYYYYYYYYYYYYYYYYYYY",
        },
        {"P1", "P2", "P3"},
        parameters=SequenceClusterParameters(min_seq_id=0.9, coverage=0.8),
        cache_dir=tmp_path / "cache",
    )

    assert result is not None
    assert set(result.protein_to_group) == {"P1", "P2", "P3"}
    assert result.protein_to_group["P1"] == result.protein_to_group["P2"]
    assert result.metadata["method"] == "mmseqs2"
