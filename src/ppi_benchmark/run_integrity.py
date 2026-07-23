"""Atomic run ownership and integrity-checked completion artifacts."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import (
    canonical_json_sha256,
    canonical_json_snapshot,
    file_sha256,
)


RUN_CLAIM_FILENAME = ".run_claim"
RUN_FINGERPRINT_FILENAME = "run_fingerprint.json"
RUN_FINGERPRINT_SCHEMA_ID = "protein_benchmark.run_fingerprint"
RUN_FINGERPRINT_SCHEMA_VERSION = 1


class RunIntegrityError(ValueError):
    """Raised when a run is incomplete, malformed, or modified."""


def validate_run_identity(identity: Mapping[str, Any]) -> None:
    """Validate canonical task-neutral keys in one scientific identity."""
    task = identity.get("task")
    if (
        not isinstance(task, Mapping)
        or not isinstance(task.get("task_id"), str)
        or not task["task_id"].strip()
    ):
        raise RunIntegrityError("Run identity requires a nonempty task_id.")

    encoder = identity.get("encoder")
    if encoder is not None and not isinstance(encoder, Mapping):
        raise RunIntegrityError("Run identity encoder must be null or an object.")

    matrices = identity.get("matrices") or {}
    if not isinstance(matrices, Mapping):
        raise RunIntegrityError("Run identity matrices must be an object.")
    for source, split_records in matrices.items():
        if not isinstance(split_records, Mapping):
            raise RunIntegrityError(
                f"Matrix source {source!r} must contain split records."
            )
        for split, record in split_records.items():
            if not isinstance(record, Mapping):
                raise RunIntegrityError(
                    f"Matrix {source}/{split} must be an object."
                )
            if record.get("matrix_source") != source:
                raise RunIntegrityError(
                    f"Matrix {source}/{split} has inconsistent source metadata."
                )
            if record.get("split") != split:
                raise RunIntegrityError(
                    f"Matrix {source}/{split} has inconsistent split metadata."
                )

    models = identity.get("models") or {}
    if not isinstance(models, Mapping):
        raise RunIntegrityError("Run identity models must be an object.")
    for configuration_id, seed_records in models.items():
        if not isinstance(seed_records, Mapping):
            raise RunIntegrityError(
                f"Model {configuration_id!r} must contain seed records."
            )
        for seed_key, record in seed_records.items():
            if not isinstance(record, Mapping):
                raise RunIntegrityError(
                    f"Model {configuration_id}/{seed_key} must be an object."
                )
            if record.get("configuration_id") != configuration_id:
                raise RunIntegrityError(
                    f"Model {configuration_id}/{seed_key} has an inconsistent "
                    "configuration_id."
                )
            model_seed = record.get("model_seed")
            if model_seed is None:
                expected_seed = "none"
            elif isinstance(model_seed, bool) or not isinstance(model_seed, int):
                raise RunIntegrityError(
                    f"Model {configuration_id}/{seed_key} has a non-integer "
                    "model_seed."
                )
            else:
                expected_seed = str(model_seed)
            if seed_key != expected_seed:
                raise RunIntegrityError(
                    f"Model {configuration_id}/{seed_key} has an inconsistent "
                    "model_seed."
                )

    run_contract = identity.get("run_contract") or {}
    if not isinstance(run_contract, Mapping):
        raise RunIntegrityError("Run identity run_contract must be an object.")
    protocols = run_contract.get("protocols") or {}
    if not isinstance(protocols, Mapping):
        raise RunIntegrityError("Run identity protocols must be an object.")
    for protocol_id, instance_records in protocols.items():
        if not isinstance(instance_records, Mapping):
            raise RunIntegrityError(
                f"Protocol {protocol_id!r} must contain instance records."
            )
        for instance_hash, record in instance_records.items():
            if not isinstance(record, Mapping):
                raise RunIntegrityError(
                    f"Protocol {protocol_id}/{instance_hash} must be an object."
                )
            if record.get("protocol_id") != protocol_id:
                raise RunIntegrityError(
                    f"Protocol {protocol_id}/{instance_hash} has an "
                    "inconsistent protocol_id."
                )
            unhashed_record = dict(record)
            recorded_hash = unhashed_record.pop(
                "protocol_instance_sha256", None)
            if (
                recorded_hash != instance_hash
                or canonical_json_sha256(unhashed_record) != instance_hash
            ):
                raise RunIntegrityError(
                    f"Protocol {protocol_id}/{instance_hash} has an "
                    "inconsistent instance hash."
                )


def claim_run_directory(run_dir: str | Path, *, task_id: str) -> Path:
    """Atomically claim one empty run directory for an immutable execution."""
    run_path = Path(run_dir)
    if run_path.is_symlink():
        raise RunIntegrityError(f"Run directory must not be a symlink: {run_path}")
    if run_path.exists():
        if not run_path.is_dir():
            raise RunIntegrityError(f"Run path is not a directory: {run_path}")
        if any(run_path.iterdir()):
            raise RunIntegrityError(
                f"Run directory is not empty: {run_path}. Choose a new "
                "--run-dir; runs are immutable."
            )
    else:
        run_path.mkdir(parents=True, exist_ok=True)

    claim_path = run_path / RUN_CLAIM_FILENAME
    claim = {
        "schema_id": "protein_benchmark.run_claim",
        "schema_version": 1,
        "task_id": str(task_id),
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as exc:
        raise RunIntegrityError(
            f"Run directory is already claimed: {run_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(claim, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
    except Exception:
        # Preserve the exclusive claim even if recording its metadata fails.
        raise

    unexpected = sorted(
        path.name for path in run_path.iterdir() if path != claim_path
    )
    if unexpected:
        raise RunIntegrityError(
            f"Run directory changed while it was being claimed: {run_path}; "
            f"unexpected entries: {unexpected}."
        )
    return run_path


def _is_operational_path(relative_path: Path) -> bool:
    name = relative_path.name
    return (
        relative_path == Path(RUN_FINGERPRINT_FILENAME)
        or name.endswith(".lock")
        or (name.startswith(".") and name.endswith(".tmp"))
    )


def _artifact_role(relative_path: Path) -> str:
    if relative_path == Path(RUN_CLAIM_FILENAME):
        return "run_claim"
    if relative_path.parts[0] in {"splits", "sampling"}:
        return "split_artifact"
    if relative_path.name == "protein_encoder.json":
        return "encoder_provenance"
    if relative_path.parts[0] == "checkpoints" or relative_path.name in {
        "predictions.csv",
        "train_metrics.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "training_history.csv",
        "val_degree_metrics.csv",
        "test_degree_metrics.csv",
    }:
        return "model_output"
    if relative_path.parts[0] == "plots" or "summary" in relative_path.name:
        return "report"
    if relative_path.name in {"performance.jsonl", "invocations.jsonl"}:
        return "runtime_provenance"
    return "other"


def artifact_manifest(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Return hashes and roles for all certifiable files in one run."""
    run_path = Path(run_dir)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(run_path.rglob("*")):
        relative_path = path.relative_to(run_path)
        if path.is_symlink():
            raise RunIntegrityError(
                f"Completed runs must not contain symlinks: {relative_path}"
            )
        if _is_operational_path(relative_path):
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise RunIntegrityError(
                f"Unsupported run artifact type: {relative_path}"
            )
        key = relative_path.as_posix()
        records[key] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "role": _artifact_role(relative_path),
        }
    return records


def write_run_fingerprint(
    run_dir: str | Path,
    *,
    identity: Mapping[str, Any],
) -> Path:
    """Finalize a claimed run with an atomic integrity fingerprint."""
    run_path = Path(run_dir)
    claim_path = run_path / RUN_CLAIM_FILENAME
    if not claim_path.is_file():
        raise RunIntegrityError(
            f"Cannot finalize an unclaimed run directory: {run_path}"
        )
    output_path = run_path / RUN_FINGERPRINT_FILENAME
    if output_path.exists():
        raise RunIntegrityError(f"Run is already finalized: {run_path}")

    validate_run_identity(identity)
    payload = canonical_json_snapshot({
        "identity": identity,
        "artifacts": artifact_manifest(run_path),
    })
    fingerprint = {
        "schema_id": RUN_FINGERPRINT_SCHEMA_ID,
        "schema_version": RUN_FINGERPRINT_SCHEMA_VERSION,
        "payload": payload,
        "payload_sha256": canonical_json_sha256(payload),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".run_fingerprint.",
        suffix=".tmp",
        dir=run_path,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(fingerprint, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def read_run_fingerprint(
    run_dir: str | Path,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Read and validate one current-format completed run."""
    run_path = Path(run_dir)
    claim_path = run_path / RUN_CLAIM_FILENAME
    fingerprint_path = run_path / RUN_FINGERPRINT_FILENAME
    if run_path.is_symlink():
        raise RunIntegrityError(
            f"Run directory must not be a symlink: {run_path}"
        )
    for control_path in (claim_path, fingerprint_path):
        if control_path.is_symlink():
            raise RunIntegrityError(
                f"Run control artifact must not be a symlink: {control_path}"
            )
    if not claim_path.is_file():
        raise RunIntegrityError(
            f"Current-format run lacks {RUN_CLAIM_FILENAME}: {run_path}"
        )
    if not fingerprint_path.is_file():
        raise RunIntegrityError(
            f"Claimed run is incomplete; missing {RUN_FINGERPRINT_FILENAME}: "
            f"{run_path}"
        )
    try:
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunIntegrityError(
            f"Invalid run fingerprint at {fingerprint_path}: {exc}"
        ) from exc
    if fingerprint.get("schema_id") != RUN_FINGERPRINT_SCHEMA_ID:
        raise RunIntegrityError(
            f"Unsupported run fingerprint schema at {fingerprint_path}: "
            f"{fingerprint.get('schema_id')!r}."
        )
    if fingerprint.get("schema_version") != RUN_FINGERPRINT_SCHEMA_VERSION:
        raise RunIntegrityError(
            f"Unsupported run fingerprint version at {fingerprint_path}: "
            f"{fingerprint.get('schema_version')!r}."
        )
    payload = fingerprint.get("payload")
    if not isinstance(payload, dict):
        raise RunIntegrityError(
            f"Run fingerprint payload must be an object: {fingerprint_path}"
        )
    observed_payload_hash = canonical_json_sha256(payload)
    if fingerprint.get("payload_sha256") != observed_payload_hash:
        raise RunIntegrityError(
            f"Run fingerprint payload hash mismatch: {fingerprint_path}"
        )
    identity = payload.get("identity")
    expected_artifacts = payload.get("artifacts")
    if not isinstance(identity, dict) or not isinstance(expected_artifacts, dict):
        raise RunIntegrityError(
            f"Run fingerprint requires identity and artifacts objects: "
            f"{fingerprint_path}"
        )
    validate_run_identity(identity)
    if verify_artifacts:
        observed_artifacts = artifact_manifest(run_path)
        if observed_artifacts != expected_artifacts:
            expected_paths = set(expected_artifacts)
            observed_paths = set(observed_artifacts)
            missing = sorted(expected_paths - observed_paths)
            extra = sorted(observed_paths - expected_paths)
            changed = sorted(
                path
                for path in expected_paths & observed_paths
                if expected_artifacts[path] != observed_artifacts[path]
            )
            raise RunIntegrityError(
                f"Run artifact integrity check failed for {run_path}: "
                f"missing={missing}, extra={extra}, changed={changed}."
            )
    return fingerprint
