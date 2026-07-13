import numpy as np
import pytest
from sklearn.metrics import f1_score

from ppi_benchmark.thresholds import (
    fixed_threshold_selection,
    predictions_at_threshold,
    select_validation_f1_threshold,
)


def test_validation_f1_selection_replaces_a_suboptimal_default_threshold():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.10, 0.40, 0.35, 0.45])

    selection = select_validation_f1_threshold(
        labels=labels,
        scores=scores,
        default_threshold=0.5,
    )
    predictions = predictions_at_threshold(scores, selection.threshold)

    assert selection.threshold == pytest.approx(0.35)
    assert selection.default_threshold == 0.5
    assert selection.strategy == "validation_f1"
    assert selection.metric_name == "f1"
    assert selection.metric_value == pytest.approx(0.8)
    assert f1_score(labels, predictions) == pytest.approx(0.8)


def test_validation_f1_keeps_the_backend_default_when_it_is_optimal():
    labels = np.asarray([0, 1, 1])
    scores = np.asarray([-1.0, 0.0, 1.0])

    selection = select_validation_f1_threshold(
        labels=labels,
        scores=scores,
        default_threshold=0.0,
    )

    assert selection.threshold == 0.0
    assert selection.metric_value == 1.0


def test_fixed_threshold_reports_validation_f1_without_tuning():
    labels = np.asarray([0, 1, 1])
    scores = np.asarray([0.2, 0.4, 0.9])

    selection = fixed_threshold_selection(
        default_threshold=0.5,
        labels=labels,
        scores=scores,
    )

    assert selection.threshold == 0.5
    assert selection.strategy == "fixed"
    assert selection.metric_value == pytest.approx(2.0 / 3.0)
