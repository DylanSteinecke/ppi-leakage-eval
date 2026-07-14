"""Stable identifiers shared by result-producing pipeline modules."""


EVALUATION_SCHEMA_VERSION = 1
STANDARD_PREDICTION_COLUMNS = (
    "evaluation_schema_version",
    "task",
    "split",
    "example_id",
    "target",
    "score",
    "prediction",
)
