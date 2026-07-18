"""Backward-compatible normalization for evaluation result tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..schema import EVALUATION_SCHEMA_VERSION
from ..tasks.ppi_models import (
    CONTROL_GROUP,
    LEGACY_CLASSIFIER_ESTIMATOR_IDS,
    TRAINING_DEGREE_MATRIX,
    ppi_model_spec,
)


PREFERENTIAL_ATTACHMENT_MODEL = "preferential_attachment"


def _source_label(source: str | Path | None) -> str:
    return "evaluation table" if source is None else str(source)


def _legacy_identity(model_name: str) -> dict[str, Any]:
    if model_name == PREFERENTIAL_ATTACHMENT_MODEL:
        return {
            "estimator_id": PREFERENTIAL_ATTACHMENT_MODEL,
            "estimator_params": "{}",
            "matrix_source": TRAINING_DEGREE_MATRIX,
            "reporting_group": CONTROL_GROUP,
        }
    try:
        estimator_id = LEGACY_CLASSIFIER_ESTIMATOR_IDS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown schema-v1 classifier {model_name!r}; add an explicit "
            "compatibility mapping before aggregating this artifact."
        ) from exc
    spec = ppi_model_spec(model_name)
    return {
        "estimator_id": estimator_id,
        "estimator_params": json.dumps(
            dict(spec.estimator_params),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "matrix_source": spec.matrix_source,
        "reporting_group": spec.reporting_group,
    }


def normalize_evaluation_frame(
    frame: pd.DataFrame,
    *,
    source: str | Path | None = None,
    allow_missing_version: bool = False,
) -> pd.DataFrame:
    """Return schema-v2 identities for a schema-v1 or schema-v2 table."""
    if frame.empty:
        return frame.copy()
    normalized = frame.copy()
    if "evaluation_schema_version" not in normalized.columns:
        if not allow_missing_version or "classifier" not in normalized.columns:
            raise ValueError(
                f"{_source_label(source)} lacks evaluation_schema_version."
            )
        normalized["evaluation_schema_version"] = 1
    versions = set(
        pd.to_numeric(
            normalized["evaluation_schema_version"],
            errors="raise",
        ).astype(int)
    )
    if len(versions) != 1:
        raise ValueError(
            f"{_source_label(source)} mixes evaluation schema versions: "
            f"{sorted(versions)}."
        )
    version = versions.pop()
    if version == EVALUATION_SCHEMA_VERSION:
        if "classifier" in normalized.columns:
            raise ValueError(
                f"Schema-v{version} {_source_label(source)} must not contain "
                "the legacy classifier field."
            )
        return normalized
    if version != 1:
        raise ValueError(
            f"Unsupported evaluation schema version {version} in "
            f"{_source_label(source)}."
        )
    required = {"classifier", "model_name"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(
            f"Schema-v1 {_source_label(source)} lacks columns: "
            f"{sorted(missing)}."
        )

    legacy_configuration = normalized["model_name"].astype(str)
    public_models = normalized["classifier"].astype(str)
    identities = [_legacy_identity(name) for name in public_models]
    normalized["model_name"] = public_models
    normalized["configuration_id"] = legacy_configuration
    for field in (
        "estimator_id",
        "estimator_params",
        "matrix_source",
        "reporting_group",
    ):
        normalized[field] = [identity[field] for identity in identities]
    normalized["feature_identity"] = normalized.get("features", pd.NA)
    for field in (
        "feature_spec_sha256",
        "fitted_extractor_sha256",
        "matrix_schema_id",
        "pair_composition_schema_id",
        "matrix_contract_sha256",
        "row_identity_sha256",
        "matrix_sha256",
        "matrix_persisted",
    ):
        normalized[field] = pd.NA
    normalized = normalized.drop(columns="classifier")
    return normalized


def validate_feature_identity_collisions(
    frame: pd.DataFrame,
    *,
    source: str | Path | None = None,
) -> None:
    """Reject one shortened feature identity resolving to multiple specs."""
    required = {"feature_identity", "feature_spec_sha256"}
    if frame.empty or not required <= set(frame.columns):
        return
    identities = frame[list(required)].dropna().drop_duplicates()
    collisions = (
        identities.groupby("feature_identity")["feature_spec_sha256"]
        .nunique()
    )
    collisions = collisions[collisions > 1]
    if not collisions.empty:
        raise ValueError(
            f"{_source_label(source)} contains shortened feature identity "
            f"collisions: {sorted(collisions.index.astype(str))}."
        )
