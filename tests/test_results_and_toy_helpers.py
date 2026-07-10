import argparse
import math
import random

import pandas as pd
import pytest

from ppi_benchmark.cli.make_toy_data import (
    PairCandidate,
    positive_float,
    probability,
    weighted_sample_without_replacement,
)
from ppi_benchmark.results import summarize_metrics


def test_summarize_metrics_supports_plain_metric_tables(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame({
        "accuracy": [0.5, 1.0],
        "f1": [0.4, 0.8],
    }).to_csv(metrics_path, index=False)

    summary = summarize_metrics(metrics_path)

    assert summary["n_runs"].tolist() == [2]
    assert summary["accuracy_mean"].tolist() == [0.75]
    assert summary["f1_mean"].tolist() == [pytest.approx(0.6)]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_toy_numeric_parsers_reject_non_finite_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(value)
    with pytest.raises(argparse.ArgumentTypeError):
        probability(value)


def test_weighted_sampling_is_reproducible_without_replacement():
    candidates = [
        PairCandidate(
            protein_a=f"P{index}",
            protein_b=f"Q{index}",
            component_id=0,
            relation="compatible",
            latent_score=float(index),
            sample_weight=math.sqrt(index + 1),
        )
        for index in range(100)
    ]
    random.seed(17)
    first_sample = weighted_sample_without_replacement(candidates, 20)
    random.seed(17)
    second_sample = weighted_sample_without_replacement(candidates, 20)

    assert first_sample == second_sample
    assert len(set(first_sample)) == 20
