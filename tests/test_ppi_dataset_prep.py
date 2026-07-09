import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ppi_dataset_utils import (
    canonicalize_filter_and_assign,
    canonicalize_pairs,
    infer_separator,
    read_fasta,
    sample_negative_pairs,
    validate_negative_ratio,
    write_fasta,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP_SCRIPT = REPO_ROOT / "scripts" / "prepare_ppi_dataset.py"


def write_text(path, text):
    """
    Write text to a temporary path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_test_fasta(path, protein_ids):
    """
    Write compact FASTA records for test proteins.
    """
    write_text(
        path,
        "".join(f">{protein_id}\nACDEFGHIK{index}\n"
                for index, protein_id in enumerate(protein_ids)),
    )


def run_prep_cli(*args, check=True):
    """
    Run the dataset-prep CLI in a subprocess.
    """
    command = [sys.executable, str(PREP_SCRIPT), *map(str, args)]
    completed_process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed_process.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"{' '.join(command)}\n"
            f"STDOUT:\n{completed_process.stdout}\n"
            f"STDERR:\n{completed_process.stderr}"
        )

    return completed_process


def pair_frame(rows):
    """
    Return a compact pair dataframe.
    """
    return pd.DataFrame(
        rows,
        columns=["protein_a", "protein_b", "label"],
    )


def test_fasta_read_write_round_trip(tmp_path):
    fasta_path = tmp_path / "input.fasta"
    output_path = tmp_path / "output.fasta"
    write_text(fasta_path, ">B\nBBBB\n>A\nAAAA\n")

    sequences = read_fasta(fasta_path)
    write_fasta(sequences, {"A", "B"}, output_path)

    assert read_fasta(output_path) == {"A": "AAAA", "B": "BBBB"}
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == ">A"


def test_duplicate_fasta_ids_fail(tmp_path):
    fasta_path = tmp_path / "dupes.fasta"
    write_text(fasta_path, ">A\nAAAA\n>A\nCCCC\n")

    with pytest.raises(ValueError, match="Duplicate FASTA sequence ID"):
        read_fasta(fasta_path)


def test_extension_separator_inference(tmp_path):
    assert infer_separator(tmp_path / "pairs.csv") == ","
    assert infer_separator(tmp_path / "pairs.tsv") == "\t"
    assert infer_separator(tmp_path / "pairs.txt") == "\t"

    with pytest.raises(ValueError, match="Cannot infer separator"):
        infer_separator(tmp_path / "pairs.dat")


@pytest.mark.parametrize("negative_ratio", [0.0, -1.0, math.inf, math.nan])
def test_negative_ratio_must_be_finite_positive(negative_ratio):
    with pytest.raises(ValueError, match="negative_ratio"):
        validate_negative_ratio(negative_ratio)


def test_self_pairs_are_removed_and_counted():
    pairs = pair_frame([
        ("A", "A", 1),
        ("A", "B", 1),
    ])

    result = canonicalize_pairs(pairs)

    assert result.metadata["n_self_pairs_removed"] == 1
    assert result.pairs[["protein_a", "protein_b", "label"]].values.tolist() == [
        ["A", "B", 1],
    ]


def test_duplicate_unordered_same_label_pairs_are_deduplicated():
    pairs = pair_frame([
        ("A", "B", 1),
        ("B", "A", 1),
        ("A", "B", 1),
    ])

    result = canonicalize_pairs(pairs)

    assert result.metadata["n_duplicate_pairs_removed"] == 2
    assert len(result.pairs) == 1
    assert result.pairs.iloc[0].to_dict() == {
        "protein_a": "A",
        "protein_b": "B",
        "label": 1,
    }


def test_contradictory_unordered_pair_labels_fail():
    pairs = pair_frame([
        ("A", "B", 1),
        ("B", "A", 0),
    ])

    with pytest.raises(ValueError, match="contradictory labels"):
        canonicalize_pairs(pairs)


def test_missing_sequence_pairs_are_dropped_and_counted():
    pairs = pair_frame([
        ("A", "B", 1),
        ("A", "MISSING", 0),
    ])
    sequences = {"A": "AAAA", "B": "BBBB"}

    result = canonicalize_filter_and_assign(pairs, sequences)

    assert result.metadata["n_pairs_dropped_missing_sequence"] == 1
    assert result.metadata["n_pairs_output"] == 1


def test_sampled_negatives_do_not_overlap_positives():
    positives = pair_frame([
        ("A", "B", 1),
        ("B", "A", 1),
        ("C", "D", 1),
    ])

    negatives, metadata = sample_negative_pairs(
        positive_pairs=positives,
        negative_ratio=1.0,
        seed=7,
    )
    positive_keys = {
        tuple(sorted((row.protein_a, row.protein_b)))
        for row in positives.itertuples()
    }
    negative_keys = {
        tuple(sorted((row.protein_a, row.protein_b)))
        for row in negatives.itertuples()
    }

    assert metadata["target_n_negatives"] == 2
    assert len(negative_keys) == 2
    assert positive_keys.isdisjoint(negative_keys)


def test_negative_sampling_fails_when_not_enough_candidates():
    positives = pair_frame([
        ("A", "B", 1),
        ("A", "C", 1),
        ("B", "C", 1),
    ])

    with pytest.raises(ValueError, match="Not enough possible negative pairs"):
        sample_negative_pairs(
            positive_pairs=positives,
            negative_ratio=1.0,
            seed=0,
        )


def test_canonical_pair_ids_are_generated_after_deduplication():
    pairs = pair_frame([
        ("B", "A", 1),
        ("A", "B", 1),
        ("C", "D", 0),
    ])
    sequences = {"A": "AAAA", "B": "BBBB", "C": "CCCC", "D": "DDDD"}

    result = canonicalize_filter_and_assign(pairs, sequences)

    assert result.pairs["pair_id"].tolist() == ["pair_0", "pair_1"]
    assert result.pairs[["protein_a", "protein_b", "label"]].values.tolist() == [
        ["C", "D", 0],
        ["A", "B", 1],
    ]


def test_generic_edges_positive_negative_cli_writes_canonical_outputs(tmp_path):
    fasta_path = tmp_path / "proteins.fasta"
    positives_path = tmp_path / "positives.tsv"
    negatives_path = tmp_path / "negatives.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["A", "B", "C", "D", "E", "F"])
    write_text(positives_path, "protein_a\tprotein_b\nA\tB\nC\tD\n")
    write_text(negatives_path, "protein_a\tprotein_b\nE\tF\nA\tC\n")

    run_prep_cli(
        "generic_edges",
        "--dataset-name", "toy_real",
        "--positive-pairs", positives_path,
        "--negative-pairs", negatives_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
    )

    dataset_dir = out_dir / "toy_real"
    pairs = pd.read_csv(dataset_dir / "pairs.csv")
    metadata = json.loads(
        (dataset_dir / "dataset_metadata.json").read_text(encoding="utf-8"))

    assert (dataset_dir / "proteins.fasta").exists()
    assert pairs.columns.tolist() == ["pair_id", "protein_a", "protein_b", "label"]
    assert set(pairs["label"]) == {0, 1}
    assert metadata["loader_name"] == "generic_edges"
    assert metadata["n_positive_input"] == 2
    assert metadata["n_negative_input"] == 2


def test_generic_edges_sampled_negatives_are_reproducible(tmp_path):
    fasta_path = tmp_path / "proteins.fasta"
    positives_path = tmp_path / "positives.csv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["A", "B", "C", "D"])
    write_text(positives_path, "protein_a,protein_b\nA,B\nC,D\n")

    common_args = [
        "generic_edges",
        "--positive-pairs", positives_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--sample-negatives",
        "--negative-ratio", "1.0",
        "--seed", "13",
    ]
    run_prep_cli(*common_args, "--dataset-name", "sampled_one")
    run_prep_cli(*common_args, "--dataset-name", "sampled_two")

    first_pairs = pd.read_csv(out_dir / "sampled_one" / "pairs.csv")
    second_pairs = pd.read_csv(out_dir / "sampled_two" / "pairs.csv")

    pd.testing.assert_frame_equal(first_pairs, second_pairs)


def test_generic_edges_output_dir_overwrite_rules(tmp_path):
    fasta_path = tmp_path / "proteins.fasta"
    positives_path = tmp_path / "positives.tsv"
    negatives_path = tmp_path / "negatives.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["A", "B", "C", "D"])
    write_text(positives_path, "protein_a\tprotein_b\nA\tB\n")
    write_text(negatives_path, "protein_a\tprotein_b\nC\tD\n")
    common_args = [
        "generic_edges",
        "--dataset-name", "overwrite_demo",
        "--positive-pairs", positives_path,
        "--negative-pairs", negatives_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
    ]

    run_prep_cli(*common_args)
    sentinel_path = out_dir / "overwrite_demo" / "keep_me.txt"
    write_text(sentinel_path, "do not delete\n")
    failed_process = run_prep_cli(*common_args, check=False)
    run_prep_cli(*common_args, "--overwrite")

    assert failed_process.returncode != 0
    assert "Output directory already exists" in failed_process.stderr
    assert sentinel_path.exists()


def test_biogrid_cli_filters_model_organism_and_writes_metadata(tmp_path):
    fasta_path = tmp_path / "model_organism.fasta"
    interactions_path = tmp_path / "biogrid.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["YAL001C", "YBR002C", "YCR003C", "YDR004C"])
    write_text(
        interactions_path,
        "\t".join([
            "Interactor A",
            "Interactor B",
            "Organism A",
            "Organism B",
            "System Type",
        ])
        + "\n"
        + "YAL001C\tYBR002C\t559292\t559292\tphysical\n"
        + "YCR003C\tYDR004C\t559292\t559292\tphysical\n"
        + "YAL001C\tYCR003C\t559292\t559292\tphysical\n"
        + "YBR002C\tYDR004C\t6239\t6239\tphysical\n",
    )

    run_prep_cli(
        "biogrid",
        "--dataset-name", "biogrid_yeast",
        "--interactions", interactions_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--protein-a-col", "Interactor A",
        "--protein-b-col", "Interactor B",
        "--organism-a-col", "Organism A",
        "--organism-b-col", "Organism B",
        "--organism-id", "559292",
        "--experimental-system-type-col", "System Type",
        "--allowed-system-types", "physical",
        "--sample-negatives",
        "--negative-ratio", "1.0",
        "--seed", "5",
    )

    dataset_dir = out_dir / "biogrid_yeast"
    pairs = pd.read_csv(dataset_dir / "pairs.csv")
    metadata = json.loads(
        (dataset_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    positive_keys = {
        tuple(sorted((row.protein_a, row.protein_b)))
        for row in pairs[pairs["label"] == 1].itertuples()
    }
    negative_keys = {
        tuple(sorted((row.protein_a, row.protein_b)))
        for row in pairs[pairs["label"] == 0].itertuples()
    }

    assert set(pairs["label"]) == {0, 1}
    assert positive_keys.isdisjoint(negative_keys)
    assert metadata["loader_name"] == "biogrid"
    assert metadata["n_positive_input"] == 4
    assert metadata["loader_specific_options"]["organism_id"] == "559292"
    assert (
        metadata["loader_specific_options"][
            "n_positive_after_loader_filters"
        ]
        == 3
    )


def test_biogrid_fails_when_id_columns_cannot_be_inferred(tmp_path):
    fasta_path = tmp_path / "model_organism.fasta"
    interactions_path = tmp_path / "biogrid.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["A", "B", "C", "D"])
    write_text(
        interactions_path,
        "Left\tRight\nA\tB\nC\tD\n",
    )

    completed_process = run_prep_cli(
        "biogrid",
        "--dataset-name", "biogrid_model_organism",
        "--interactions", interactions_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--sample-negatives",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "Pass --protein-a-col explicitly" in completed_process.stderr


def test_biogrid_fails_on_ambiguous_multi_id_values(tmp_path):
    fasta_path = tmp_path / "model_organism.fasta"
    interactions_path = tmp_path / "biogrid.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["A", "B", "C", "D"])
    write_text(
        interactions_path,
        "Interactor A\tInteractor B\nA|ALT\tB\nC\tD\n",
    )

    completed_process = run_prep_cli(
        "biogrid",
        "--dataset-name", "biogrid_model_organism",
        "--interactions", interactions_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--protein-a-col", "Interactor A",
        "--protein-b-col", "Interactor B",
        "--sample-negatives",
        check=False,
    )

    assert completed_process.returncode != 0
    assert "ambiguous multi-ID values" in completed_process.stderr
