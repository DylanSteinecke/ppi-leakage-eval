"""Scientific run-identity construction for the PPI task."""

from __future__ import annotations

from typing import Any, Mapping

from ..artifact_io import canonical_json_sha256, canonical_json_snapshot


_NON_SCIENTIFIC_KEYS = frozenset({
    "cache_hit",
    "cache_path",
    "created_at_utc",
    "path",
    "selected_examples_path",
    "work_dir",
})


def _scientific_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _scientific_value(item)
            for key, item in sorted(value.items())
            if str(key) not in _NON_SCIENTIFIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_value(item) for item in value]
    return value


def _input_hashes(split_metadata: Mapping[str, Any]) -> dict[str, str]:
    fields = {
        "pairs": "pairs_file_sha256",
        "fasta": "fasta_file_sha256",
        "dataset_metadata": "dataset_metadata_file_sha256",
        "protein_metadata": "protein_metadata_file_sha256",
        "sequence_clusters": "sequence_clusters_file_sha256",
    }
    return {
        role: str(split_metadata[field])
        for role, field in fields.items()
        if split_metadata.get(field)
    }


def _matrix_identities(
    matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    fields = (
        "matrix_source",
        "split",
        "matrix_schema_id",
        "pair_composition_schema_id",
        "matrix_contract_sha256",
        "row_identity_sha256",
        "matrix_sha256",
    )
    identities: dict[str, dict[str, dict[str, Any]]] = {}
    for source, split_records in sorted(matrices.items()):
        identities[source] = {}
        for split, record in sorted(split_records.items()):
            if record.get("matrix_source") != source:
                raise ValueError(
                    f"Matrix {source}/{split} has inconsistent source metadata."
                )
            if record.get("split") != split:
                raise ValueError(
                    f"Matrix {source}/{split} has inconsistent split metadata."
                )
            identities[source][split] = {
                field: record.get(field) for field in fields
            }
    return identities


def _model_identities(
    model_runs: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    fields = (
        "model_name",
        "estimator_id",
        "estimator_params",
        "feature_spec_sha256",
        "feature_identity",
        "configuration_id",
        "features",
        "matrix_source",
        "matrix_schema_id",
        "pair_composition_schema_id",
        "reporting_group",
        "backend",
        "model_seed",
    )
    identities: dict[str, dict[str, dict[str, Any]]] = {}
    for run in model_runs:
        configuration_id = str(run["configuration_id"])
        model_seed = run.get("model_seed")
        seed_key = "none" if model_seed is None else str(int(model_seed))
        identity = {
            field: _scientific_value(run.get(field)) for field in fields
        }
        seed_records = identities.setdefault(configuration_id, {})
        if seed_key in seed_records:
            raise ValueError(
                f"Duplicate model identity {configuration_id}/{seed_key}."
            )
        seed_records[seed_key] = identity
    return {
        configuration: dict(sorted(seed_records.items()))
        for configuration, seed_records in sorted(identities.items())
    }


def build_ppi_run_identity(
    *,
    task_schema_version: int,
    split_metadata: Mapping[str, Any],
    protocol_instance: Mapping[str, Any],
    evaluation_cohort_hashes: Mapping[str, str],
    split_assignments_sha256: str,
    matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_runs: list[dict[str, Any]],
    encoder_metadata: Mapping[str, Any] | None,
    threshold_selection: str,
    has_validation_split: bool,
    evaluate_test_metrics: bool,
) -> dict[str, Any]:
    """Return path-independent scientific identity for one PPI run."""
    protocol_record = _scientific_value(protocol_instance)
    protocol_instance_hash = canonical_json_sha256(protocol_record)
    protocol_id = str(protocol_record["protocol_id"])
    protocols = {
        protocol_id: {
            protocol_instance_hash: {
                **protocol_record,
                "protocol_instance_sha256": protocol_instance_hash,
            }
        }
    }
    encoder = None
    if encoder_metadata is not None:
        encoder = {
            "encoder_fingerprint": encoder_metadata["encoder_fingerprint"],
            "encoder_identity_strength": encoder_metadata[
                "encoder_identity_strength"
            ],
            "sequence_normalization_schema_id": encoder_metadata[
                "sequence_normalization_schema_id"
            ],
            "unique_embedding_inputs_sha256": encoder_metadata[
                "unique_embedding_inputs_sha256"
            ],
            "embedding_contract_sha256": encoder_metadata[
                "embedding_contract_sha256"
            ],
            "embedding_table_sha256": encoder_metadata[
                "embedding_table_sha256"
            ],
        }
    sampling = split_metadata.get("sampling") or {}
    sampling_identity = {
        key: _scientific_value(sampling.get(key))
        for key in (
            "applied",
            "fraction",
            "max_examples",
            "n_eligible",
            "n_excluded",
            "n_selected",
            "seed",
        )
        if key in sampling
    }
    return canonical_json_snapshot({
        "task": {
            "task_id": "ppi",
            "task_schema_version": task_schema_version,
        },
        "run_contract": {
            "inputs": _input_hashes(split_metadata),
            "protocols": protocols,
            "cohorts": dict(sorted(evaluation_cohort_hashes.items())),
            "split_assignments_sha256": split_assignments_sha256,
            "negative_construction": _scientific_value(
                split_metadata.get("negative_construction")
            ),
            "sampling": sampling_identity,
            "evaluation_policy": {
                "threshold_selection": threshold_selection,
                "has_validation_split": bool(has_validation_split),
                "evaluate_test_metrics": bool(evaluate_test_metrics),
            },
        },
        "encoder": encoder,
        "matrices": _matrix_identities(matrices),
        "models": _model_identities(model_runs),
    })
