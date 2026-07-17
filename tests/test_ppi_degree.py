import json

import numpy as np
import pandas as pd
import pytest

from ppi_benchmark.reporting.degree import (
    BENCHMARK_DEGREE_CONTROL_SELECTION_FILENAME,
    build_degree_lift,
    select_primary_degree_control,
    selected_control_for_context,
)
from ppi_benchmark.cli.aggregate import aggregate_benchmark_results
from ppi_benchmark.tasks.ppi_degree import (
    DEGREE_FEATURE_NAMES,
    PREFERENTIAL_ATTACHMENT_CLASSIFIER,
    build_degree_evaluation_plan,
    build_training_degree_profile,
    degree_feature_matrix,
    degree_metric_rows as evaluate_degree_plan,
    evaluation_cohort_sha256,
    preferential_attachment_scores,
)


def degree_metric_rows(examples, profile, *, split_strategy, **kwargs):
    plan = build_degree_evaluation_plan(examples, profile, split_strategy)
    return evaluate_degree_plan(plan, **kwargs)


def training_fixture():
    train = pd.DataFrame({
        "source_row_index": [0, 1, 2, 3],
        "protein_a": ["A", "B", "A", "D"],
        "protein_b": ["B", "A", "C", "A"],
        "label": [1, 1, 1, 0],
    })
    assignments = pd.DataFrame({
        "source_row_index": [0, 1, 2, 3, 4, 5],
        "split": ["train", "train", "train", "train", "val", "test"],
    })
    return train, assignments


def test_training_degree_profile_rejects_held_out_rows_and_counts_partners():
    train, assignments = training_fixture()

    profile = build_training_degree_profile(train, assignments)
    repeated = build_training_degree_profile(train.copy(), assignments.copy())

    assert profile.positive_degree == {"A": 2, "B": 1, "C": 1, "D": 0}
    assert profile.training_exposure == {"A": 4, "B": 2, "C": 1, "D": 1}
    assert profile.degree_bin("D") == "zero"
    assert profile.degree_bin("held_out_only") == "unseen"
    assert profile.exposure_bin("held_out_only") == "unseen"
    assert profile.metadata == repeated.metadata
    pd.testing.assert_frame_equal(profile.frame, repeated.frame)

    injected = pd.concat([
        train,
        pd.DataFrame({
            "source_row_index": [4],
            "protein_a": ["X"],
            "protein_b": ["Y"],
            "label": [1],
        }),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="non-training/extra rows"):
        build_training_degree_profile(injected, assignments)


def test_held_out_perturbations_cannot_change_training_degree_provenance():
    train, assignments = training_fixture()
    profile_before = build_training_degree_profile(train, assignments)
    held_out = pd.DataFrame({
        "source_row_index": [4, 5],
        "protein_a": ["A", "X"],
        "protein_b": ["X", "Y"],
        "label": [1, 0],
    })
    perturbed = held_out.assign(
        protein_a=["NEW_A", "NEW_B"],
        protein_b=["NEW_C", "NEW_D"],
        label=[0, 1],
    )

    # Held-out content is deliberately not an input to profile construction.
    profile_after = build_training_degree_profile(train, assignments)

    assert not held_out.equals(perturbed)
    assert profile_before.metadata == profile_after.metadata
    pd.testing.assert_frame_equal(profile_before.frame, profile_after.frame)


def test_degree_features_are_symmetric_and_leave_positive_edge_out():
    train, assignments = training_fixture()
    profile = build_training_degree_profile(train, assignments)
    pair = pd.DataFrame({
        "protein_a": ["A", "C"],
        "protein_b": ["C", "A"],
        "label": [1, 1],
    })

    complete = degree_feature_matrix(
        pair, profile, leave_one_positive_edge_out=False
    )
    leave_one_out = degree_feature_matrix(
        pair, profile, leave_one_positive_edge_out=True
    )

    assert len(DEGREE_FEATURE_NAMES) == 4
    np.testing.assert_allclose(complete[0], complete[1])
    np.testing.assert_allclose(leave_one_out[0], leave_one_out[1])
    np.testing.assert_allclose(
        leave_one_out[0],
        np.log1p([0, 1, 1, 0]),
    )
    assert not np.array_equal(complete, leave_one_out)


def test_pair_order_has_identical_degree_bin_assignment():
    train, assignments = training_fixture()
    profile = build_training_degree_profile(train, assignments)
    forward = pd.DataFrame({
        "protein_a": ["A"],
        "protein_b": ["D"],
        "label": [0],
    })
    reverse = pd.DataFrame({
        "protein_a": ["D"],
        "protein_b": ["A"],
        "label": [0],
    })
    kwargs = {
        "profile": profile,
        "scores": [0.25],
        "predictions": [0],
        "global_threshold": 0.5,
        "split_strategy": "random",
        "metadata": {"model_name": "model", "split": "test"},
    }

    forward_rows = degree_metric_rows(forward, **kwargs)
    reverse_rows = degree_metric_rows(reverse, **kwargs)
    columns = ["stratification_axis", "stratum"]

    pd.testing.assert_frame_equal(
        forward_rows[columns].sort_values(columns).reset_index(drop=True),
        reverse_rows[columns].sort_values(columns).reset_index(drop=True),
    )


def test_degree_evaluation_plan_is_reusable_and_immutable():
    train, assignments = training_fixture()
    profile = build_training_degree_profile(train, assignments)
    examples = pd.DataFrame({
        "protein_a": ["A", "D"],
        "protein_b": ["B", "A"],
        "label": [1, 0],
    })
    plan = build_degree_evaluation_plan(examples, profile, "random")
    kwargs = {
        "scores": np.asarray([0.8, 0.2]),
        "predictions": np.asarray([1, 0]),
        "global_threshold": 0.5,
        "metadata": {"model_name": "model", "split": "val"},
    }

    first = evaluate_degree_plan(plan, **kwargs)
    second = evaluate_degree_plan(plan, **kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert not plan.targets.flags.writeable
    assert not plan.preferential_attachment_scores.flags.writeable
    assert all(not stratum.mask.flags.writeable for stratum in plan.strata)


def test_quantile_ties_are_deterministic_and_can_leave_empty_middle_bins():
    train = pd.DataFrame({
        "source_row_index": [0, 1, 2],
        "protein_a": ["A", "B", "C"],
        "protein_b": ["B", "C", "A"],
        "label": [1, 1, 1],
    })
    assignments = pd.DataFrame({
        "source_row_index": [0, 1, 2],
        "split": ["train", "train", "train"],
    })

    profile = build_training_degree_profile(train, assignments)

    assert profile.positive_degree_cutoffs == (2.0, 2.0)
    assert set(profile.frame["positive_degree_bin"]) == {"low"}


def test_c2_and_c3_reporting_apply_protocol_specific_degree_contracts():
    train = pd.DataFrame({
        "source_row_index": [0, 1],
        "protein_a": ["S", "S"],
        "protein_b": ["T", "U"],
        "label": [1, 0],
    })
    assignments = pd.DataFrame({
        "source_row_index": [0, 1],
        "split": ["train", "train"],
    })
    profile = build_training_degree_profile(train, assignments)
    c2 = pd.DataFrame({
        "protein_a": ["S", "X"],
        "protein_b": ["X", "S"],
        "label": [1, 0],
    })
    pa_scores = preferential_attachment_scores(c2, profile)
    metadata = {
        "model_name": PREFERENTIAL_ATTACHMENT_CLASSIFIER,
        "split": "test",
    }

    c2_rows = degree_metric_rows(
        c2,
        profile,
        scores=pa_scores,
        predictions=None,
        global_threshold=None,
        split_strategy="c2",
        metadata=metadata,
    )
    global_row = c2_rows[c2_rows["stratification_axis"] == "global"].iloc[0]

    assert np.all(pa_scores == 0.0)
    assert global_row["auprc"] == pytest.approx(0.5)
    assert global_row["auroc"] == pytest.approx(0.5)
    assert set(c2_rows["endpoint_novelty"]) == {"one_seen_by_protocol"}
    assert set(c2_rows["count"]) == {2}
    assert c2_rows["stratification_note"].str.contains(
        "structurally zero"
    ).any()

    c3 = pd.DataFrame({
        "protein_a": ["X", "Y"],
        "protein_b": ["Y", "Z"],
        "label": [1, 0],
    })
    c3_rows = degree_metric_rows(
        c3,
        profile,
        scores=preferential_attachment_scores(c3, profile),
        predictions=None,
        global_threshold=None,
        split_strategy="c3",
        metadata=metadata,
    )

    assert c3_rows["stratification_axis"].tolist() == ["global"]
    assert not bool(c3_rows.iloc[0]["stratification_applicable"])
    assert "not applicable" in c3_rows.iloc[0]["stratification_note"]
    c3_features = degree_feature_matrix(
        c3,
        profile,
        leave_one_positive_edge_out=False,
    )
    assert np.all(c3_features == 0.0)
    with pytest.raises(RuntimeError, match="C3 degree-control scores"):
        degree_metric_rows(
            c3,
            profile,
            scores=[0.0, 1.0],
            predictions=None,
            global_threshold=None,
            split_strategy="c3",
            metadata=metadata,
        )


def test_degree_strata_reuse_global_predictions_and_report_one_class_counts():
    train, assignments = training_fixture()
    profile = build_training_degree_profile(train, assignments)
    examples = pd.DataFrame({
        "protein_a": ["A", "D", "X"],
        "protein_b": ["B", "A", "Y"],
        "label": [1, 0, 0],
    })
    scores = np.asarray([0.8, 0.4, 0.2])
    predictions = (scores >= 0.6).astype(int)

    rows = degree_metric_rows(
        examples,
        profile,
        scores=scores,
        predictions=predictions,
        global_threshold=0.6,
        split_strategy="random",
        metadata={"model_name": "model", "split": "test"},
    )

    assert (rows["global_threshold"] == 0.6).all()
    assert (rows["count"] == rows["positives"] + rows["negatives"]).all()
    assert rows["prevalence"].between(0.0, 1.0).all()
    assert rows["auroc"].isna().any()
    assert {
        "positive_degree_least",
        "positive_degree_greatest",
        "positive_degree_pair",
        "training_exposure_least",
        "training_exposure_greatest",
        "training_exposure_pair",
        "endpoint_novelty",
    }.issubset(set(rows["stratification_axis"]))


def test_evaluation_cohort_hash_is_pair_order_symmetric_but_label_sensitive():
    examples = pd.DataFrame({
        "source_row_index": [8, 9],
        "protein_a": ["A", "C"],
        "protein_b": ["B", "D"],
        "label": [1, 0],
    })
    reversed_pairs = examples.rename(columns={
        "protein_a": "protein_b",
        "protein_b": "protein_a",
    })
    changed_label = examples.assign(label=[0, 0])

    assert evaluation_cohort_sha256(examples) == evaluation_cohort_sha256(
        reversed_pairs
    )
    assert evaluation_cohort_sha256(examples) != evaluation_cohort_sha256(
        changed_label
    )


def control_selection_rows(hgb_delta):
    rows = []
    for split_seed in range(5):
        for model_name, auprc in (
            ("degree_logistic", 0.60),
            ("degree_hgb", 0.60 + hgb_delta),
        ):
            rows.append({
                "split": "val",
                "stratification_axis": "global",
                "model_name": model_name,
                "split_strategy": "random",
                "split_seed": split_seed,
                "auprc": auprc,
            })
    return pd.DataFrame(rows)


def test_prepublication_control_rule_is_predeclared_and_defaults_to_logistic():
    promoted = select_primary_degree_control(control_selection_rows(0.03))
    retained = select_primary_degree_control(control_selection_rows(0.02))
    missing = select_primary_degree_control(control_selection_rows(0.03)[:-2])

    assert promoted["selected_control"] == "degree_hgb"
    assert retained["selected_control"] == "degree_logistic"
    assert missing["selected_control"] == "degree_logistic"


def test_control_selection_is_context_scoped_and_lift_requires_coverage():
    promoted = control_selection_rows(0.03).assign(
        control_selection_context_sha256="context-a"
    )
    defaulted = control_selection_rows(0.03)
    defaulted = defaulted[
        defaulted["model_name"] == "degree_logistic"
    ].assign(control_selection_context_sha256="context-b")
    selection = select_primary_degree_control(
        pd.concat((promoted, defaulted), ignore_index=True)
    )

    assert selected_control_for_context(selection, "context-a") == "degree_hgb"
    assert (
        selected_control_for_context(selection, "context-b")
        == "degree_logistic"
    )

    rows = []
    for context, selected_control in (
        ("context-a", "degree_hgb"),
        ("context-b", "degree_logistic"),
    ):
        common = {
            "task": "ppi",
            "split": "test",
            "split_strategy": "random",
            "protocol_id": "ppi.random_pair.v1",
            "protocol_version": 1,
            "split_seed": 0,
            "dataset_sha256": f"dataset-{context}",
            "training_positive_edges_sha256": f"positive-{context}",
            "training_examples_sha256": f"training-{context}",
            "split_assignments_sha256": f"assignments-{context}",
            "evaluation_cohort_sha256": f"cohort-{context}",
            "negative_construction_sha256": "negatives",
            "grouping_sha256": "grouping",
            "protocol_instance_sha256": f"protocol-{context}",
            "degree_provenance_sha256": f"degree-{context}",
            "control_selection_context_sha256": context,
            "stratification_axis": "global",
            "features": "none",
            "run_number": 1,
            "model_seed": 0,
        }
        rows.extend((
            {
                **common,
                "model_name": "logistic",
                "configuration_id": f"real-{context}",
                "model_role": "predictive_model",
                "auprc": 0.8,
            },
            {
                **common,
                "model_name": selected_control,
                "configuration_id": selected_control,
                "model_role": "degree_control",
                "auprc": 0.6,
            },
            {
                **common,
                "model_name": PREFERENTIAL_ATTACHMENT_CLASSIFIER,
                "configuration_id": PREFERENTIAL_ATTACHMENT_CLASSIFIER,
                "model_role": "degree_reference",
                "auprc": 0.5,
            },
        ))
    metrics = pd.DataFrame(rows)
    lift = build_degree_lift(metrics, selection)

    assert len(lift) == 2
    assert set(lift["selected_fitted_control"]) == {
        "degree_hgb",
        "degree_logistic",
    }
    missing_control = metrics[
        ~(
            (metrics["control_selection_context_sha256"] == "context-b")
            & (metrics["model_name"] == "degree_logistic")
        )
    ]
    with pytest.raises(ValueError, match="required degree_logistic rows"):
        build_degree_lift(missing_control, selection)


@pytest.mark.parametrize("mismatch_field", [
    "evaluation_cohort_sha256",
    "split_assignments_sha256",
    "negative_construction_sha256",
    "grouping_sha256",
    "training_positive_edges_sha256",
])
def test_lift_join_refuses_mismatched_identity_gates(mismatch_field):
    common = {
        "task": "ppi",
        "split": "test",
        "split_strategy": "random",
        "protocol_id": "ppi.random_pair.v1",
        "protocol_version": 1,
        "split_seed": 0,
        "dataset_sha256": "dataset",
        "training_positive_edges_sha256": "positive",
        "training_examples_sha256": "training",
        "split_assignments_sha256": "assignments",
        "negative_construction_sha256": "negatives",
        "grouping_sha256": "grouping",
        "protocol_instance_sha256": "protocol",
        "degree_provenance_sha256": "degree",
        "control_selection_context_sha256": "selection-context",
        "stratification_axis": "global",
        "features": "none",
        "run_number": 1,
        "model_seed": 0,
    }
    model_row = {
            **common,
            "model_name": "logistic",
            "configuration_id": "real",
            "model_role": "predictive_model",
            "evaluation_cohort_sha256": "cohort-a",
            "auprc": 0.8,
        }
    control_row = {
            **common,
            "model_name": "degree_logistic",
            "configuration_id": "degree_logistic",
            "model_role": "primary_degree_control",
            "evaluation_cohort_sha256": "cohort-a",
            "auprc": 0.6,
        }
    control_row[mismatch_field] = f"different-{mismatch_field}"
    pa_row = {
            **common,
            "model_name": PREFERENTIAL_ATTACHMENT_CLASSIFIER,
            "configuration_id": PREFERENTIAL_ATTACHMENT_CLASSIFIER,
            "model_role": "degree_reference",
            "evaluation_cohort_sha256": "cohort-a",
            "auprc": 0.5,
        }
    metrics = pd.DataFrame([model_row, control_row, pa_row])

    with pytest.raises(ValueError, match=mismatch_field):
        build_degree_lift(metrics, "degree_logistic")


def test_cli_degree_artifact_lifecycle_and_registered_control(
    tmp_path,
    ppi_test_data,
    run_train,
):
    pairs_path, fasta_path = ppi_test_data
    run_dir = tmp_path / "degree-run"

    run_train(
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--run-dir", run_dir,
        "--classifier", "degree_logistic",
        "--train-size", "0.70",
        "--val-size", "0.25",
        "--split-seed", "4",
        "--model-seeds", "7",
        "--no-metrics-plots",
    )

    profile_path = run_dir / "splits" / "training_positive_degree.csv"
    val_metrics_path = run_dir / "val_degree_metrics.csv"
    assert profile_path.exists()
    assert val_metrics_path.exists()
    assert (run_dir / "val_degree_metrics_summary.csv").exists()
    assert not (run_dir / "test_degree_metrics.csv").exists()
    val_metrics = pd.read_csv(val_metrics_path)
    assert {
        PREFERENTIAL_ATTACHMENT_CLASSIFIER,
        "degree_logistic",
    } == set(val_metrics["model_name"])
    assert val_metrics[
        val_metrics["model_name"] == "degree_logistic"
    ]["global_threshold"].notna().all()
    metadata = json.loads(
        (run_dir / "splits" / "split_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostic = metadata["diagnostics"]["degree_diagnostic"]
    assert diagnostic["degree_diagnostic_schema_version"] == 1
    assert diagnostic["primary_fitted_control"] == "degree_logistic"
    assert diagnostic["coefficient_interpretation_allowed"] is False


def test_benchmark_aggregation_writes_hash_gated_degree_lift(
    tmp_path,
    ppi_test_data,
    run_train,
):
    pairs_path, fasta_path = ppi_test_data
    benchmark_dir = tmp_path / "benchmark"
    baseline_dir = benchmark_dir / "runs" / "random" / "seed0" / "baselines"
    model_dir = benchmark_dir / "runs" / "random" / "seed0" / "features"
    common = (
        "--pairs", pairs_path,
        "--fasta", fasta_path,
        "--train-size", "0.70",
        "--val-size", "0.25",
        "--eval-test-set",
        "--split-strategy", "random",
        "--split-seed", "0",
        "--model-seeds", "0",
        "--no-metrics-plots",
    )
    run_train(
        *common,
        "--run-dir", baseline_dir,
        "--classifier", "degree_logistic",
    )
    run_train(
        *common,
        "--run-dir", model_dir,
        "--features", "count",
        "--classifier", "logistic",
    )

    aggregate_benchmark_results(benchmark_dir)

    summary = pd.read_csv(benchmark_dir / "benchmark_degree_summary.csv")
    lift = pd.read_csv(benchmark_dir / "benchmark_degree_lift.csv")
    selection_path = (
        benchmark_dir / BENCHMARK_DEGREE_CONTROL_SELECTION_FILENAME
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert {"degree_logistic", "logistic"}.issubset(
        set(summary["model_name"])
    )
    assert not lift.empty
    assert set(lift["selected_fitted_control"]) == {"degree_logistic"}
    assert np.allclose(
        lift["residual_over_degree_control"],
        lift["absolute_lift"],
    )
    assert set(lift["split"]) == {"val", "test"}
    assert selection["degree_control_selection_schema_version"] == 1
    assert "selection_evidence_sha256" in selection
    assert "control_selection_evidence" not in summary.columns

    aggregate_benchmark_results(benchmark_dir)
    baseline_degree_path = baseline_dir / "val_degree_metrics.csv"
    baseline_degree = pd.read_csv(baseline_degree_path)
    changed = (
        (baseline_degree["model_name"] == "degree_logistic")
        & (baseline_degree["stratification_axis"] == "global")
    )
    baseline_degree.loc[changed, "auprc"] += 0.01
    baseline_degree.to_csv(baseline_degree_path, index=False)
    with pytest.raises(ValueError, match="locked fitted degree control"):
        aggregate_benchmark_results(benchmark_dir)
