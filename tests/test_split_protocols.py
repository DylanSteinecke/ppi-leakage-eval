from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pandas as pd
import pytest

from ppi_benchmark.splitting import dispatch
from ppi_benchmark.splitting.dispatch import load_or_make_split
from ppi_benchmark.splitting.protocols import (
    C1_SPLIT_STRATEGY,
    C2_SPLIT_STRATEGY,
    C3_SPLIT_STRATEGY,
    PROVIDED_SPLIT_STRATEGY,
    RANDOM_SPLIT_STRATEGY,
    SPLIT_STRATEGY_CHOICES,
    SplitStrategySpec,
    get_split_strategy,
    registered_split_strategies,
)


EXPECTED_PPI_PROTOCOLS = {
    PROVIDED_SPLIT_STRATEGY: "ppi.provided.v1",
    RANDOM_SPLIT_STRATEGY: "ppi.random_pair.v1",
    C1_SPLIT_STRATEGY: "ppi.c1.v1",
    C2_SPLIT_STRATEGY: "ppi.c2.v1",
    C3_SPLIT_STRATEGY: "ppi.c3.v1",
}


def test_ppi_registry_has_expected_task_qualified_protocols():
    assert registered_split_strategies("ppi") == tuple(
        sorted(EXPECTED_PPI_PROTOCOLS)
    )
    for strategy_name, protocol_id in EXPECTED_PPI_PROTOCOLS.items():
        spec = get_split_strategy("ppi", strategy_name)
        assert spec.task_id == "ppi"
        assert isinstance(spec, SplitStrategySpec)
        assert spec.strategy_name == strategy_name
        assert spec.protocol_id == protocol_id
        assert spec.protocol_version == 1
        assert protocol_id.startswith(f"{spec.task_id}.")
        assert protocol_id.endswith(f".v{spec.protocol_version}")
        assert spec.prediction_unit
        assert spec.assignment_entity
        assert spec.projection_kind


def test_dispatch_reexports_strategy_constants():
    assert dispatch.PROVIDED_SPLIT_STRATEGY == PROVIDED_SPLIT_STRATEGY
    assert dispatch.RANDOM_SPLIT_STRATEGY == RANDOM_SPLIT_STRATEGY
    assert dispatch.C1_SPLIT_STRATEGY == C1_SPLIT_STRATEGY
    assert dispatch.C2_SPLIT_STRATEGY == C2_SPLIT_STRATEGY
    assert dispatch.C3_SPLIT_STRATEGY == C3_SPLIT_STRATEGY
    assert dispatch.SPLIT_STRATEGY_CHOICES == SPLIT_STRATEGY_CHOICES


def test_c2_and_c3_declare_grouping_and_projection_drops():
    c2 = get_split_strategy("ppi", "c2")
    c3 = get_split_strategy("ppi", "c3")

    for spec in (c2, c3):
        assert spec.assignment_entity == "protein_group"
        assert spec.default_grouping_kind == "protein_identity"
        assert spec.allowed_grouping_kinds == (
            "protein_identity",
            "sequence_cluster",
        )
        assert spec.projection_drop_reasons

    assert c2.projection_drop_reasons == (
        "c2_two_validation_endpoints",
        "c2_two_test_endpoints",
        "c2_validation_test_edge",
    )
    assert c3.projection_drop_reasons == ("c3_cross_partition_edge",)


def test_strategy_spec_is_frozen():
    spec = get_split_strategy("ppi", "random")

    with pytest.raises(FrozenInstanceError):
        spec.strategy_name = "changed"


def test_unknown_ppi_strategy_lists_supported_legacy_names():
    with pytest.raises(ValueError) as error:
        get_split_strategy("ppi", "unknown")

    message = str(error.value)
    assert "Split strategy 'unknown' is not supported for task 'ppi'" in message
    for strategy_name in EXPECTED_PPI_PROTOCOLS:
        assert strategy_name in message


def test_ppi_strategy_requested_for_ptm_fails_without_future_suggestions():
    with pytest.raises(ValueError) as error:
        get_split_strategy("ptm", "c2")

    message = str(error.value)
    assert "registered for task(s): ppi" in message
    assert "No split strategies are registered for task 'ptm'" in message
    assert "protein_disjoint" not in message
    assert "homology_disjoint" not in message


def test_ppi_boundary_rejects_mismatched_resolved_strategy():
    args = SimpleNamespace(effective_split_strategy="random")
    split_spec = get_split_strategy("ppi", "c2")

    with pytest.raises(ValueError, match="does not match"):
        load_or_make_split(
            pd.DataFrame(),
            args,
            split_spec=split_spec,
        )


def test_ppi_boundary_rejects_non_ppi_specification():
    args = SimpleNamespace(effective_split_strategy="c2")
    split_spec = replace(get_split_strategy("ppi", "c2"), task_id="ptm")

    with pytest.raises(ValueError, match="cannot execute task 'ptm'"):
        load_or_make_split(
            pd.DataFrame(),
            args,
            split_spec=split_spec,
        )


def test_legacy_ppi_boundary_resolves_same_random_split():
    pairs = pd.DataFrame({
        "protein_a": [f"A{index}" for index in range(20)],
        "protein_b": [f"B{index}" for index in range(20)],
        "label": [0, 1] * 10,
    })

    def args():
        return SimpleNamespace(
            effective_split_strategy="random",
            seed=7,
            split_seed=7,
            split_col=None,
            train_size=0.5,
            val_size=0.0,
        )

    implicit = load_or_make_split(pairs, args(), None)
    explicit = load_or_make_split(
        pairs,
        args(),
        None,
        split_spec=get_split_strategy("ppi", "random"),
    )

    for implicit_frame, explicit_frame in zip(implicit, explicit):
        if implicit_frame is None:
            assert explicit_frame is None
        else:
            assert implicit_frame.index.tolist() == explicit_frame.index.tolist()


def test_c1_audit_does_not_claim_a_protein_grouping_relation():
    pairs = pd.DataFrame([
        (f"P{left}", f"P{right}", (left + right) % 2)
        for left in range(12)
        for right in range(left + 1, 12)
    ], columns=["protein_a", "protein_b", "label"])
    args = SimpleNamespace(
        effective_split_strategy="c1",
        seed=7,
        split_seed=7,
        split_col=None,
        train_size=0.7,
        val_size=0.0,
        n_split_trials=20,
    )

    load_or_make_split(pairs, args)

    assert args.split_audit["grouping_kind"] is None
    assert args.split_audit["grouping_type"] is None
