import pandas as pd
import pytest

from ppi_benchmark.splitting.cohort import SamplingSpec, select_examples


def test_unstratified_sampling_is_reproducible_and_without_replacement():
    example_ids = pd.Series(range(100))
    spec = SamplingSpec(max_examples=20, seed=17)

    first = select_examples(example_ids, spec)
    second = select_examples(example_ids, spec)

    assert first.selected_positions.tolist() == second.selected_positions.tolist()
    assert first.sampling_ranks.tolist() == second.sampling_ranks.tolist()
    assert len(set(first.selected_positions)) == 20
    assert first.metadata["n_selected"] == 20
    assert first.metadata["n_excluded"] == 80
    assert first.metadata["applied"] is True


def test_stratified_samples_are_nested_for_increasing_counts():
    example_ids = pd.Series(range(200))
    strata = pd.DataFrame({
        "label": [0] * 120 + [1] * 80,
    })

    smaller = select_examples(
        example_ids,
        SamplingSpec(max_examples=40, seed=9, minimum_per_stratum=1),
        strata=strata,
    )
    larger = select_examples(
        example_ids,
        SamplingSpec(max_examples=80, seed=9, minimum_per_stratum=1),
        strata=strata,
    )

    assert set(smaller.selected_positions) < set(larger.selected_positions)
    smaller_labels = strata.iloc[smaller.selected_positions]["label"]
    assert smaller_labels.value_counts().to_dict() == {0: 24, 1: 16}


def test_sampling_can_stratify_a_ptm_shaped_cohort():
    ptm_examples = pd.DataFrame({
        "site_id": [f"site-{index}" for index in range(24)],
        "protein_id": [f"P{index // 4}" for index in range(24)],
        "ptm_type": ["phosphorylation"] * 16 + ["acetylation"] * 8,
        "label": [0, 1] * 12,
    })
    strata = ptm_examples[["ptm_type", "label"]]

    result = select_examples(
        ptm_examples["site_id"],
        SamplingSpec(max_examples=12, seed=3, minimum_per_stratum=1),
        strata=strata,
    )

    sampled = ptm_examples.iloc[result.selected_positions]
    assert len(sampled) == 12
    assert set(sampled["ptm_type"]) == {"phosphorylation", "acetylation"}
    assert set(sampled["label"]) == {0, 1}
    assert result.metadata["stratum_columns"] == ["ptm_type", "label"]


def test_fraction_uses_floor_with_a_minimum_of_one():
    result = select_examples(
        ["a", "b", "c"],
        SamplingSpec(fraction=0.5, seed=1),
    )

    assert len(result.selected_positions) == 1


def test_cap_at_or_above_cohort_size_is_an_audited_noop():
    result = select_examples(
        ["a", "b", "c"],
        SamplingSpec(max_examples=10, seed=1),
    )

    assert result.selected_positions.tolist() == [0, 1, 2]
    assert result.metadata["requested"] is True
    assert result.metadata["applied"] is False
    assert result.metadata["n_excluded"] == 0


def test_minimum_per_stratum_fails_clearly_when_sample_is_too_small():
    with pytest.raises(ValueError, match="at least 4 examples are required"):
        select_examples(
            range(8),
            SamplingSpec(max_examples=3, seed=0, minimum_per_stratum=1),
            strata=pd.DataFrame({"group": list("AAAABBCD")}),
        )


@pytest.mark.parametrize(
    "example_ids,spec,error",
    [
        (["a", "a"], SamplingSpec(max_examples=1), "must be unique"),
        (["a"], SamplingSpec(max_examples=0), "at least 1"),
        (["a"], SamplingSpec(fraction=1.0), "less than 1"),
    ],
)
def test_invalid_sampling_inputs_fail_clearly(example_ids, spec, error):
    with pytest.raises(ValueError, match=error):
        select_examples(example_ids, spec)
