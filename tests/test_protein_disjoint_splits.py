import json

import pandas as pd
import pytest

from ppi_benchmark.splitters.protein_disjoint import (
    split_pairs,
    split_pairs_three_way,
)


def complete_graph_pairs(n_proteins=14):
    """
    Return a dense balanced graph suitable for every split regime.
    """
    rows = [
        (f"P{left}", f"P{right}", (left + right) % 2)
        for left in range(n_proteins)
        for right in range(left + 1, n_proteins)
    ]

    return pd.DataFrame(
        rows,
        columns=["protein_a", "protein_b", "label"],
    )


def proteins(pairs):
    """
    Return proteins represented by one pair table.
    """
    return set(pairs["protein_a"]) | set(pairs["protein_b"])


def unordered_pairs(pairs):
    """
    Return unordered pair identities from one pair table.
    """
    return {
        tuple(sorted((protein_a, protein_b)))
        for protein_a, protein_b in zip(
            pairs["protein_a"],
            pairs["protein_b"],
        )
    }


def paired_group_mapping(n_proteins=14):
    """
    Group consecutive proteins into indivisible two-protein units.
    """
    return {
        f"P{protein_index}": f"G{protein_index // 2}"
        for protein_index in range(n_proteins)
    }


def test_c1_is_edge_disjoint_and_all_test_proteins_appear_in_train():
    pairs = complete_graph_pairs()
    duplicate = pairs.iloc[[0]].copy()
    duplicate[["protein_a", "protein_b"]] = duplicate[
        ["protein_b", "protein_a"]].to_numpy()
    pairs = pd.concat([pairs, duplicate], ignore_index=True)

    result = split_pairs(
        pairs,
        mode="c1",
        test_size=0.25,
        seed=3,
        n_trials=30,
    )

    assert unordered_pairs(result.train).isdisjoint(
        unordered_pairs(result.test))
    assert proteins(result.test) <= proteins(result.train)
    assert result.dropped.empty
    assert result.audit["all_invariants_passed"] is True


def test_c2_has_exactly_one_train_group_per_test_edge():
    pairs = complete_graph_pairs()
    protein_to_group = paired_group_mapping()

    result = split_pairs(
        pairs,
        mode="c2",
        test_size=0.35,
        protein_to_group=protein_to_group,
        seed=4,
        n_trials=30,
    )
    train_groups = {
        protein_to_group[protein_id]
        for protein_id in proteins(result.train)
    }

    assert all(
        (protein_to_group[protein_a] in train_groups)
        ^ (protein_to_group[protein_b] in train_groups)
        for protein_a, protein_b in zip(
            result.test["protein_a"],
            result.test["protein_b"],
        )
    )
    assert all(
        protein_to_group[protein_a] not in train_groups
        and protein_to_group[protein_b] not in train_groups
        for protein_a, protein_b in zip(
            result.dropped["protein_a"],
            result.dropped["protein_b"],
        )
    )
    assert result.audit["all_invariants_passed"] is True


def test_c3_has_no_train_test_group_overlap_and_discards_cross_edges():
    pairs = complete_graph_pairs()
    protein_to_group = paired_group_mapping()

    result = split_pairs(
        pairs,
        mode="c3",
        test_size=0.30,
        protein_to_group=protein_to_group,
        seed=5,
        n_trials=30,
    )
    train_groups = {
        protein_to_group[protein_id]
        for protein_id in proteins(result.train)
    }
    test_groups = {
        protein_to_group[protein_id]
        for protein_id in proteins(result.test)
    }

    assert train_groups.isdisjoint(test_groups)
    assert all(
        (protein_to_group[protein_a] in train_groups)
        ^ (protein_to_group[protein_b] in train_groups)
        for protein_a, protein_b in zip(
            result.dropped["protein_a"],
            result.dropped["protein_b"],
        )
    )
    assert result.audit["n_shared_groups_train_test"] == 0
    assert result.audit["all_invariants_passed"] is True


@pytest.mark.parametrize("mode", ["c1", "c2", "c3"])
def test_split_is_deterministic_for_fixed_seed(mode):
    pairs = complete_graph_pairs()
    protein_to_group = None if mode == "c1" else paired_group_mapping()
    kwargs = {
        "mode": mode,
        "test_size": 0.3,
        "protein_to_group": protein_to_group,
        "seed": 17,
        "n_trials": 20,
    }

    first = split_pairs(pairs, **kwargs)
    second = split_pairs(pairs, **kwargs)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)
    pd.testing.assert_frame_equal(first.dropped, second.dropped)
    assert first.audit == second.audit
    json.dumps(first.audit)


def test_group_mapping_must_cover_every_c2_c3_protein():
    pairs = complete_graph_pairs(6)

    with pytest.raises(ValueError, match="must map every protein"):
        split_pairs(
            pairs,
            mode="c3",
            protein_to_group={"P0": "G0"},
        )


def test_audit_contains_requested_split_quality_sections():
    result = split_pairs(
        complete_graph_pairs(),
        mode="c2",
        test_size=0.3,
        seed=9,
        n_trials=10,
    )

    assert {
        "n_shared_proteins_train_test",
        "n_retained_edges",
        "n_dropped_edges",
        "class_distribution",
        "degree_statistics",
        "invariant_checks",
        "selection_score",
    } <= set(result.audit)


def test_three_way_c1_is_edge_disjoint_with_all_endpoints_seen_in_train():
    pairs = complete_graph_pairs()
    duplicate = pairs.iloc[[0]].copy()
    duplicate[["protein_a", "protein_b"]] = duplicate[
        ["protein_b", "protein_a"]].to_numpy()
    pairs = pd.concat([pairs, duplicate], ignore_index=True)

    result = split_pairs_three_way(
        pairs,
        mode="c1",
        val_size=0.1,
        test_size=0.1,
        seed=3,
        n_trials=40,
    )

    train_pairs = unordered_pairs(result.train)
    val_pairs = unordered_pairs(result.val)
    test_pairs = unordered_pairs(result.test)
    assert train_pairs.isdisjoint(val_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert val_pairs.isdisjoint(test_pairs)
    assert proteins(result.val) <= proteins(result.train)
    assert proteins(result.test) <= proteins(result.train)
    assert result.dropped.empty
    assert result.audit["all_invariants_passed"] is True


def test_three_way_c2_uses_distinct_validation_and_test_novel_proteins():
    result = split_pairs_three_way(
        complete_graph_pairs(),
        mode="c2",
        val_size=0.1,
        test_size=0.1,
        seed=4,
        n_trials=40,
    )
    train_proteins = proteins(result.train)
    val_novel_proteins = proteins(result.val) - train_proteins
    test_novel_proteins = proteins(result.test) - train_proteins

    assert val_novel_proteins
    assert test_novel_proteins
    assert val_novel_proteins.isdisjoint(test_novel_proteins)
    assert all(
        (protein_a in train_proteins) ^ (protein_b in train_proteins)
        for protein_a, protein_b in zip(
            result.val["protein_a"], result.val["protein_b"])
    )
    assert all(
        (protein_a in train_proteins) ^ (protein_b in train_proteins)
        for protein_a, protein_b in zip(
            result.test["protein_a"], result.test["protein_b"])
    )
    assert result.audit["all_invariants_passed"] is True


def test_three_way_c3_has_pairwise_disjoint_proteins():
    result = split_pairs_three_way(
        complete_graph_pairs(),
        mode="c3",
        val_size=0.1,
        test_size=0.1,
        seed=5,
        n_trials=60,
    )
    train_proteins = proteins(result.train)
    val_proteins = proteins(result.val)
    test_proteins = proteins(result.test)

    assert train_proteins.isdisjoint(val_proteins)
    assert train_proteins.isdisjoint(test_proteins)
    assert val_proteins.isdisjoint(test_proteins)
    assert result.audit["n_shared_groups_train_val"] == 0
    assert result.audit["n_shared_groups_train_test"] == 0
    assert result.audit["n_shared_groups_val_test"] == 0
    assert result.audit["all_invariants_passed"] is True


@pytest.mark.parametrize("mode", ["c1", "c2", "c3"])
def test_three_way_split_is_deterministic(mode):
    kwargs = {
        "mode": mode,
        "val_size": 0.1,
        "test_size": 0.1,
        "seed": 17,
        "n_trials": 40,
    }

    first = split_pairs_three_way(complete_graph_pairs(), **kwargs)
    second = split_pairs_three_way(complete_graph_pairs(), **kwargs)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.val, second.val)
    pd.testing.assert_frame_equal(first.test, second.test)
    pd.testing.assert_frame_equal(first.dropped, second.dropped)
    assert first.audit == second.audit
    json.dumps(first.audit)
