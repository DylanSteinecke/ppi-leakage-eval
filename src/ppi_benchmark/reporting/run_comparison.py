"""Task-neutral comparison of completed benchmark run identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..run_integrity import read_run_fingerprint


TRUE = "true"
FALSE = "false"
NOT_APPLICABLE = "not_applicable"
NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class RunComparison:
    """Layered comparison result for two valid, comparable runs."""

    statuses: Mapping[str, str]
    differences: tuple[str, ...]

    @property
    def matches_realized_inputs(self) -> bool:
        """Return whether every applicable checked layer matches."""
        return all(
            status != FALSE
            for name, status in self.statuses.items()
            if name != "model_output_match"
        )


def _optional_hash_match(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
    field: str,
) -> str:
    if first is None and second is None:
        return NOT_APPLICABLE
    if first is None or second is None:
        return FALSE
    first_hash = first.get(field)
    second_hash = second.get(field)
    if first_hash is None and second_hash is None:
        return NOT_APPLICABLE
    if first_hash is None or second_hash is None:
        return FALSE
    return TRUE if first_hash == second_hash else FALSE


def _matrix_records(identity: Mapping[str, Any]) -> dict[tuple[str, str], Any]:
    matrices = identity.get("matrices") or {}
    records: dict[tuple[str, str], Any] = {}
    if not isinstance(matrices, dict):
        raise ValueError("Run identity matrices must be an object.")
    for source, split_records in matrices.items():
        if not isinstance(split_records, dict):
            raise ValueError(
                f"Run identity matrix source {source!r} must be an object."
            )
        for split, record in split_records.items():
            if not isinstance(record, dict):
                raise ValueError(
                    f"Run identity matrix {source}/{split} must be an object."
                )
            records[(str(source), str(split))] = record
    return records


def compare_run_identities(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> RunComparison:
    """Compare scientific identity layers without comparing run artifacts."""
    reference_task = reference.get("task") or {}
    candidate_task = candidate.get("task") or {}
    if reference_task.get("task_id") != candidate_task.get("task_id"):
        raise ValueError(
            "Run tasks are incomparable: "
            f"{reference_task.get('task_id')!r} != "
            f"{candidate_task.get('task_id')!r}."
        )

    statuses: dict[str, str] = {
        "run_contract_match": (
            TRUE
            if reference.get("run_contract") == candidate.get("run_contract")
            and reference_task == candidate_task
            else FALSE
        ),
        "encoder_contract_match": _optional_hash_match(
            reference.get("encoder"),
            candidate.get("encoder"),
            "embedding_contract_sha256",
        ),
        "embedding_table_match": _optional_hash_match(
            reference.get("encoder"),
            candidate.get("encoder"),
            "embedding_table_sha256",
        ),
        "model_configuration_match": (
            TRUE
            if reference.get("models") == candidate.get("models")
            else FALSE
        ),
        "model_output_match": NOT_CHECKED,
    }
    differences = [
        name for name, status in statuses.items() if status == FALSE
    ]

    reference_matrices = _matrix_records(reference)
    candidate_matrices = _matrix_records(candidate)
    matrix_keys = sorted(reference_matrices.keys() | candidate_matrices.keys())
    matrix_fields = {
        "matrix_contract_match": "matrix_contract_sha256",
        "matrix_row_identity_match": "row_identity_sha256",
        "matrix_value_match": "matrix_sha256",
    }
    for status_name, hash_field in matrix_fields.items():
        if not matrix_keys:
            statuses[status_name] = NOT_APPLICABLE
            continue
        mismatched_keys = []
        applicable_keys = []
        for key in matrix_keys:
            first = reference_matrices.get(key)
            second = candidate_matrices.get(key)
            if first is None or second is None:
                mismatched_keys.append(f"{key[0]}/{key[1]}")
                applicable_keys.append(key)
                continue
            first_hash = first.get(hash_field)
            second_hash = second.get(hash_field)
            if first_hash is None and second_hash is None:
                continue
            applicable_keys.append(key)
            if first_hash is None or second_hash is None or first_hash != second_hash:
                mismatched_keys.append(f"{key[0]}/{key[1]}")
        statuses[status_name] = (
            NOT_APPLICABLE
            if not applicable_keys
            else (FALSE if mismatched_keys else TRUE)
        )
        differences.extend(
            f"{status_name}:{key}" for key in mismatched_keys
        )
    matrix_statuses = [
        statuses[name] for name in matrix_fields
    ]
    statuses["matrix_match"] = (
        NOT_APPLICABLE
        if all(status == NOT_APPLICABLE for status in matrix_statuses)
        else (
            TRUE
            if all(status in {TRUE, NOT_APPLICABLE} for status in matrix_statuses)
            else FALSE
        )
    )
    return RunComparison(
        statuses=statuses,
        differences=tuple(differences),
    )


def compare_run_directories(
    reference_run: str | Path,
    candidate_run: str | Path,
) -> RunComparison:
    """Validate and compare two completed run directories."""
    reference = read_run_fingerprint(reference_run)["payload"]["identity"]
    candidate = read_run_fingerprint(candidate_run)["payload"]["identity"]
    return compare_run_identities(reference, candidate)
