import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from ppi_benchmark.cli.aggregate import run_directories
from ppi_benchmark.cli.benchmark import run as run_benchmark
from ppi_benchmark.reporting.run_comparison import (
    compare_run_directories,
    compare_run_identities,
)
from ppi_benchmark.run_integrity import (
    RUN_CLAIM_FILENAME,
    RUN_FINGERPRINT_FILENAME,
    RunIntegrityError,
    claim_run_directory,
    read_run_fingerprint,
    write_run_fingerprint,
)


def run_identity(
    *,
    task_id="ppi",
    encoder_contract="encoder-contract",
    embedding_table="embedding-table",
    matrix_contract="matrix-contract",
    matrix_rows="matrix-rows",
    matrix_values="matrix-values",
):
    return {
        "task": {
            "task_id": task_id,
            "task_schema_version": 1,
            "evaluation_schema_version": 3,
        },
        "run_contract": {"inputs": {"pairs": "pairs-hash"}},
        "encoder": {
            "embedding_contract_sha256": encoder_contract,
            "embedding_table_sha256": embedding_table,
        },
        "matrices": {
            "configured_features": {
                "train": {
                    "matrix_source": "configured_features",
                    "split": "train",
                    "matrix_contract_sha256": matrix_contract,
                    "row_identity_sha256": matrix_rows,
                    "matrix_sha256": matrix_values,
                }
            }
        },
        "models": {
            "tfidf__logistic": {
                "17": {
                    "configuration_id": "tfidf__logistic",
                    "model_seed": 17,
                }
            }
        },
    }


def completed_run(path, identity=None, artifact_text="artifact"):
    claim_run_directory(path, task_id=(identity or run_identity())["task"][
        "task_id"
    ])
    split_dir = path / "splits"
    split_dir.mkdir()
    (split_dir / "split_metadata.json").write_text(
        artifact_text,
        encoding="utf-8",
    )
    write_run_fingerprint(path, identity=identity or run_identity())
    return path


def test_concurrent_claims_allow_exactly_one_owner(tmp_path):
    run_dir = tmp_path / "run"

    def claim():
        try:
            claim_run_directory(run_dir, task_id="ppi")
        except RunIntegrityError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: claim(), range(2)))

    assert sorted(outcomes) == [False, True]
    assert (run_dir / RUN_CLAIM_FILENAME).is_file()
    assert not (run_dir / RUN_FINGERPRINT_FILENAME).exists()
    with pytest.raises(RunIntegrityError, match="not empty"):
        claim_run_directory(run_dir, task_id="ppi")


def test_claim_accepts_precreated_empty_directory_and_rejects_content(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert claim_run_directory(empty, task_id="ppi") == empty

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RunIntegrityError, match="not empty"):
        claim_run_directory(nonempty, task_id="ppi")
    assert not (nonempty / RUN_CLAIM_FILENAME).exists()


def test_fingerprint_detects_changed_missing_and_extra_artifacts(tmp_path):
    run_dir = completed_run(tmp_path / "changed")
    read_run_fingerprint(run_dir)
    metadata_path = run_dir / "splits" / "split_metadata.json"
    metadata_path.write_text("changed", encoding="utf-8")
    with pytest.raises(RunIntegrityError, match="changed="):
        read_run_fingerprint(run_dir)

    missing_run = completed_run(tmp_path / "missing")
    (missing_run / "splits" / "split_metadata.json").unlink()
    with pytest.raises(RunIntegrityError, match="missing="):
        read_run_fingerprint(missing_run)

    extra_run = completed_run(tmp_path / "extra")
    (extra_run / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RunIntegrityError, match="extra="):
        read_run_fingerprint(extra_run)


def test_fingerprint_rejects_symlinks_and_payload_tampering(tmp_path):
    linked_run = tmp_path / "linked"
    claim_run_directory(linked_run, task_id="ppi")
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    (linked_run / "linked.txt").symlink_to(target)
    with pytest.raises(RunIntegrityError, match="symlinks"):
        write_run_fingerprint(linked_run, identity=run_identity())

    operational_link_run = tmp_path / "operational-link"
    claim_run_directory(operational_link_run, task_id="ppi")
    (operational_link_run / "ignored.lock").symlink_to(target)
    with pytest.raises(RunIntegrityError, match="symlinks"):
        write_run_fingerprint(
            operational_link_run,
            identity=run_identity(),
        )

    run_dir = completed_run(tmp_path / "tampered")
    fingerprint_path = run_dir / RUN_FINGERPRINT_FILENAME
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    fingerprint["payload"]["identity"]["run_contract"] = {"changed": True}
    fingerprint_path.write_text(json.dumps(fingerprint), encoding="utf-8")
    with pytest.raises(RunIntegrityError, match="payload hash mismatch"):
        read_run_fingerprint(run_dir)

    symlinked_fingerprint = completed_run(tmp_path / "symlinked-fingerprint")
    fingerprint_path = symlinked_fingerprint / RUN_FINGERPRINT_FILENAME
    target_fingerprint = tmp_path / "fingerprint-target.json"
    fingerprint_path.replace(target_fingerprint)
    fingerprint_path.symlink_to(target_fingerprint)
    with pytest.raises(RunIntegrityError, match="must not be a symlink"):
        read_run_fingerprint(symlinked_fingerprint)


def test_comparison_reports_independent_matrix_layers(tmp_path):
    reference = completed_run(tmp_path / "reference")
    candidate = completed_run(
        tmp_path / "candidate",
        identity=run_identity(
            encoder_contract="other-contract",
            matrix_rows="other-rows",
        ),
        artifact_text="different model output",
    )

    comparison = compare_run_directories(reference, candidate)

    assert comparison.statuses["run_contract_match"] == "true"
    assert comparison.statuses["encoder_contract_match"] == "false"
    assert comparison.statuses["embedding_table_match"] == "true"
    assert comparison.statuses["matrix_contract_match"] == "true"
    assert comparison.statuses["matrix_row_identity_match"] == "false"
    assert comparison.statuses["matrix_value_match"] == "true"
    assert comparison.statuses["matrix_match"] == "false"
    assert comparison.statuses["model_output_match"] == "not_checked"
    assert comparison.matches_realized_inputs is False


def test_comparison_marks_absent_hash_layers_not_applicable():
    reference = run_identity()
    candidate = run_identity()
    for identity in (reference, candidate):
        identity["encoder"].pop("embedding_table_sha256")
        identity["matrices"]["configured_features"]["train"].pop(
            "row_identity_sha256"
        )

    comparison = compare_run_identities(reference, candidate)

    assert comparison.statuses["embedding_table_match"] == "not_applicable"
    assert (
        comparison.statuses["matrix_row_identity_match"]
        == "not_applicable"
    )
    assert comparison.statuses["matrix_match"] == "true"


def test_compare_cli_ignores_artifact_differences_but_rejects_tasks(
        tmp_path, capsys):
    reference = completed_run(
        tmp_path / "reference", artifact_text="first output")
    candidate = completed_run(
        tmp_path / "candidate", artifact_text="second output")

    assert run_benchmark((
        "compare-runs", str(reference), str(candidate),
    )) == 0
    assert "match_through_realized_model_inputs" in capsys.readouterr().out

    ptm = completed_run(
        tmp_path / "ptm",
        identity=run_identity(task_id="ptm"),
    )
    assert run_benchmark(("compare-runs", str(reference), str(ptm))) == 2
    assert "incomparable" in capsys.readouterr().err


def test_aggregation_discovers_current_historical_and_incomplete_runs(tmp_path):
    benchmark_dir = tmp_path / "benchmark"
    current = completed_run(benchmark_dir / "current")
    historical = benchmark_dir / "historical"
    (historical / "splits").mkdir(parents=True)
    (historical / "splits" / "split_metadata.json").write_text(
        "{}", encoding="utf-8")

    assert run_directories(benchmark_dir) == [current, historical]

    incomplete = benchmark_dir / "incomplete"
    claim_run_directory(incomplete, task_id="ppi")
    with pytest.raises(ValueError, match="incomplete"):
        run_directories(benchmark_dir)


def test_aggregation_rejects_fingerprint_without_claim(tmp_path):
    benchmark_dir = tmp_path / "benchmark"
    malformed = completed_run(benchmark_dir / "malformed")
    (malformed / RUN_CLAIM_FILENAME).unlink()

    with pytest.raises(ValueError, match="fingerprint without"):
        run_directories(benchmark_dir)
