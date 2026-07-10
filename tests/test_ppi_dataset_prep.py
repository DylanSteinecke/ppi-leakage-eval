import gzip
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ppi_dataset_utils import (
    apply_id_mapping_to_pairs,
    canonicalize_filter_and_assign,
    canonicalize_pairs,
    infer_separator,
    read_fasta,
    read_table,
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


def test_gzip_uniprot_fasta_ids_are_normalized(tmp_path):
    fasta_path = tmp_path / "proteins.fasta.gz"
    with gzip.open(fasta_path, "wt", encoding="utf-8") as fout:
        fout.write(
            ">sp|P12345|PROT_A description\nAAAA\n"
            ">tr|Q98765|PROT_B description\nCCCC\n"
        )

    sequences = read_fasta(
        fasta_path,
        id_format="uniprot_accession",
    )

    assert sequences == {"P12345": "AAAA", "Q98765": "CCCC"}


def test_extension_separator_inference(tmp_path):
    assert infer_separator(tmp_path / "pairs.csv") == ","
    assert infer_separator(tmp_path / "pairs.tab") == "\t"
    assert infer_separator(tmp_path / "pairs.tsv") == "\t"
    assert infer_separator(tmp_path / "pairs.txt") == "\t"
    assert infer_separator(tmp_path / "pairs.tsv.gz") == "\t"

    with pytest.raises(ValueError, match="Cannot infer separator"):
        infer_separator(tmp_path / "pairs.dat")


def test_read_table_selects_member_from_multi_file_zip(tmp_path):
    archive_path = tmp_path / "tables.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("first.tsv", "protein_a\tprotein_b\n001\t002\n")
        archive.writestr("second.tsv", "protein_a\tprotein_b\nA\tB\n")

    table = read_table(archive_path, archive_member="first.tsv")

    assert table.to_dict(orient="records") == [
        {"protein_a": "001", "protein_b": "002"},
    ]
    with pytest.raises(ValueError, match="contains 2 files"):
        read_table(archive_path)


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


def test_negative_sampling_uses_only_proteins_with_sequences():
    positives = pair_frame([
        ("A", "B", 1),
        ("C", "D", 1),
        ("MISSING_X", "MISSING_Y", 1),
    ])

    negatives, metadata = sample_negative_pairs(
        positive_pairs=positives,
        negative_ratio=1.0,
        seed=3,
        allowed_protein_ids={"A", "B", "C", "D"},
    )

    sampled_proteins = set(negatives["protein_a"]) | set(negatives["protein_b"])
    assert sampled_proteins <= {"A", "B", "C", "D"}
    assert metadata["n_positive_pairs_for_sampling"] == 2
    assert metadata["target_n_negatives"] == 2


def test_negative_sampling_scales_without_materializing_pair_universe():
    n_proteins = 5000
    positives = pair_frame([
        ("P0000", f"P{index:04d}", 1)
        for index in range(1, n_proteins)
    ])

    negatives, metadata = sample_negative_pairs(
        positive_pairs=positives,
        negative_ratio=0.001,
        seed=9,
    )

    assert len(negatives) == 5
    assert metadata["n_proteins_for_sampling"] == n_proteins
    assert not (
        (negatives["protein_a"] == "P0000")
        | (negatives["protein_b"] == "P0000")
    ).any()


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


def test_id_mapping_happens_before_pair_deduplication():
    pairs = pair_frame([
        ("A_alias", "B_alias", 1),
        ("A", "B", 1),
        ("C", "D", 1),
    ])
    mapping_table = pd.DataFrame({
        "raw_id": ["A_alias", "A", "B_alias", "B", "C", "D"],
        "canonical_id": ["UPA", "UPA", "UPB", "UPB", "UPC", "UPD"],
    })

    mapped = apply_id_mapping_to_pairs(
        pairs=pairs,
        mapping_table=mapping_table,
        map_from_col="raw_id",
        map_to_col="canonical_id",
    )
    canonicalized = canonicalize_pairs(mapped.pairs)

    assert mapped.metadata["n_pairs_before_id_mapping"] == 3
    assert mapped.metadata["n_pairs_after_id_mapping"] == 3
    assert canonicalized.metadata["n_duplicate_pairs_removed"] == 1
    assert canonicalized.pairs[["protein_a", "protein_b", "label"]].values.tolist() == [
        ["UPA", "UPB", 1],
        ["UPC", "UPD", 1],
    ]


def test_id_mapping_drops_unmapped_and_ambiguous_pairs():
    pairs = pair_frame([
        ("A", "B", 1),
        ("X", "C", 1),
        ("AMB", "D", 1),
    ])
    mapping_table = pd.DataFrame({
        "raw_id": ["A", "B", "C", "D", "AMB", "AMB"],
        "canonical_id": ["UPA", "UPB", "UPC", "UPD", "UP1", "UP2"],
    })

    result = apply_id_mapping_to_pairs(
        pairs=pairs,
        mapping_table=mapping_table,
        map_from_col="raw_id",
        map_to_col="canonical_id",
    )

    assert result.pairs[["protein_a", "protein_b", "label"]].values.tolist() == [
        ["UPA", "UPB", 1],
    ]
    assert result.metadata["n_pairs_before_id_mapping"] == 3
    assert result.metadata["n_pairs_after_id_mapping"] == 1
    assert result.metadata["n_pairs_dropped_unmapped_id"] == 1
    assert result.metadata["n_pairs_dropped_ambiguous_id"] == 1
    assert result.metadata["n_interactors_unmapped"] == 1
    assert result.metadata["n_interactors_ambiguous"] == 1
    assert result.metadata["unmapped_id_examples"] == ["X"]
    assert result.metadata["ambiguous_id_examples"] == [
        {"raw_id": "AMB", "canonical_ids": ["UP1", "UP2"]},
    ]


def test_id_mapping_table_requires_requested_columns():
    pairs = pair_frame([("A", "B", 1)])
    mapping_table = pd.DataFrame({
        "wrong_raw_col": ["A", "B"],
        "canonical_id": ["UPA", "UPB"],
    })

    with pytest.raises(ValueError, match="ID mapping table is missing"):
        apply_id_mapping_to_pairs(
            pairs=pairs,
            mapping_table=mapping_table,
            map_from_col="raw_id",
            map_to_col="canonical_id",
        )


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


def test_generic_edges_rejects_single_class_output_without_creating_dir(
        tmp_path):
    fasta_path = tmp_path / "proteins.fasta"
    positives_path = tmp_path / "positives.tsv"
    negatives_path = tmp_path / "negatives.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["A", "B"])
    write_text(positives_path, "protein_a\tprotein_b\nA\tB\n")
    write_text(negatives_path, "protein_a\tprotein_b\nX\tY\n")

    completed_process = run_prep_cli(
        "generic_edges",
        "--dataset-name", "invalid_single_class",
        "--positive-pairs", positives_path,
        "--negative-pairs", negatives_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        check=False,
    )

    assert completed_process.returncode != 0
    assert "must contain both labels" in completed_process.stderr
    assert not (out_dir / "invalid_single_class").exists()


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
    assert metadata["id_mapping_used"] is False


def test_biogrid_cli_reads_zip_member_and_compressed_uniprot_fasta(tmp_path):
    fasta_path = tmp_path / "model_organism.fasta.gz"
    interactions_path = tmp_path / "biogrid.tab3.zip"
    archive_member = "BIOGRID-ORGANISM-Yeast.tab3.txt"
    out_dir = tmp_path / "processed"
    with gzip.open(fasta_path, "wt", encoding="utf-8") as fout:
        for protein_id in ("UPA", "UPB", "UPC", "UPD"):
            fout.write(f">sp|{protein_id}|{protein_id}_YEAST\nACDEFGHIK\n")
    with zipfile.ZipFile(interactions_path, "w") as archive:
        archive.writestr("README.txt", "archive notes\n")
        archive.writestr(
            archive_member,
            "Interactor A\tInteractor B\n"
            "UPA\tUPB\n"
            "UPC\tUPD\n"
            "UPA|ALT\tUPD\n",
        )

    run_prep_cli(
        "biogrid",
        "--dataset-name", "biogrid_archive",
        "--interactions", interactions_path,
        "--archive-member", archive_member,
        "--table-chunksize", "2",
        "--fasta", fasta_path,
        "--fasta-id-format", "uniprot_accession",
        "--out-dir", out_dir,
        "--protein-a-col", "Interactor A",
        "--protein-b-col", "Interactor B",
        "--ambiguous-id-policy", "drop",
        "--sample-negatives",
        "--seed", "5",
    )

    dataset_dir = out_dir / "biogrid_archive"
    pairs = pd.read_csv(dataset_dir / "pairs.csv")
    metadata = json.loads(
        (dataset_dir / "dataset_metadata.json").read_text(encoding="utf-8"))

    assert set(pairs["label"]) == {0, 1}
    assert set(pairs["protein_a"]) | set(pairs["protein_b"]) == {
        "UPA", "UPB", "UPC", "UPD",
    }
    assert metadata["n_pairs_dropped_ambiguous_interactor"] == 1
    assert metadata["loader_specific_options"]["archive_member"] == archive_member
    assert (
        metadata["loader_specific_options"]["fasta_id_format"]
        == "uniprot_accession"
    )


def test_biogrid_id_map_maps_before_deduplication(tmp_path):
    fasta_path = tmp_path / "canonical.fasta"
    interactions_path = tmp_path / "biogrid.tsv"
    id_map_path = tmp_path / "id_map.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["UPA", "UPB", "UPC", "UPD"])
    write_text(
        interactions_path,
        "Interactor A\tInteractor B\n"
        "rawA1\trawB1\n"
        "rawA2\trawB2\n"
        "rawC\trawD\n",
    )
    write_text(
        id_map_path,
        "raw_id\tcanonical_id\n"
        "rawA1\tUPA\n"
        "rawA2\tUPA\n"
        "rawB1\tUPB\n"
        "rawB2\tUPB\n"
        "rawC\tUPC\n"
        "rawD\tUPD\n",
    )

    run_prep_cli(
        "biogrid",
        "--dataset-name", "biogrid_mapped",
        "--interactions", interactions_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--protein-a-col", "Interactor A",
        "--protein-b-col", "Interactor B",
        "--id-map", id_map_path,
        "--map-from-col", "raw_id",
        "--map-to-col", "canonical_id",
        "--sample-negatives",
        "--negative-ratio", "1.0",
        "--seed", "7",
    )

    dataset_dir = out_dir / "biogrid_mapped"
    pairs = pd.read_csv(dataset_dir / "pairs.csv")
    metadata = json.loads(
        (dataset_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    positive_keys = {
        tuple(row)
        for row in pairs.loc[
            pairs["label"] == 1, ["protein_a", "protein_b"]
        ].itertuples(index=False, name=None)
    }

    assert positive_keys == {("UPA", "UPB"), ("UPC", "UPD")}
    assert not set(pairs["protein_a"]).union(pairs["protein_b"]) & {
        "rawA1", "rawA2", "rawB1", "rawB2", "rawC", "rawD",
    }
    assert metadata["id_mapping_used"] is True
    assert metadata["id_map_path"] == str(id_map_path)
    assert metadata["map_from_col"] == "raw_id"
    assert metadata["map_to_col"] == "canonical_id"
    assert metadata["n_pairs_before_id_mapping"] == 3
    assert metadata["n_pairs_after_id_mapping"] == 3
    assert metadata["n_duplicate_pairs_removed"] == 1
    assert metadata["n_positive_after_id_mapping"] == 3


def test_biogrid_id_map_drops_and_counts_unmapped_and_ambiguous_ids(tmp_path):
    fasta_path = tmp_path / "canonical.fasta"
    interactions_path = tmp_path / "biogrid.tsv"
    id_map_path = tmp_path / "id_map.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["UPA", "UPB", "UPP", "UPQ"])
    write_text(
        interactions_path,
        "Interactor A\tInteractor B\n"
        "rawA\trawB\n"
        "rawP\trawQ\n"
        "missing\trawB\n"
        "ambiguous\trawQ\n",
    )
    write_text(
        id_map_path,
        "raw_id\tcanonical_id\n"
        "rawA\tUPA\n"
        "rawB\tUPB\n"
        "rawP\tUPP\n"
        "rawQ\tUPQ\n"
        "ambiguous\tUPX\n"
        "ambiguous\tUPY\n",
    )

    run_prep_cli(
        "biogrid",
        "--dataset-name", "biogrid_mapped_drops",
        "--interactions", interactions_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--protein-a-col", "Interactor A",
        "--protein-b-col", "Interactor B",
        "--id-map", id_map_path,
        "--map-from-col", "raw_id",
        "--map-to-col", "canonical_id",
        "--sample-negatives",
        "--negative-ratio", "1.0",
        "--seed", "11",
    )

    metadata = json.loads(
        (
            out_dir / "biogrid_mapped_drops" / "dataset_metadata.json"
        ).read_text(encoding="utf-8"))

    assert metadata["n_pairs_before_id_mapping"] == 4
    assert metadata["n_pairs_after_id_mapping"] == 2
    assert metadata["n_pairs_dropped_id_mapping"] == 2
    assert metadata["n_pairs_dropped_unmapped_id"] == 1
    assert metadata["n_pairs_dropped_ambiguous_id"] == 1
    assert metadata["n_interactors_unmapped"] == 1
    assert metadata["n_interactors_ambiguous"] == 1
    assert metadata["unmapped_id_examples"] == ["missing"]
    assert metadata["ambiguous_id_examples"] == [
        {"raw_id": "ambiguous", "canonical_ids": ["UPX", "UPY"]},
    ]


def test_biogrid_id_map_args_fail_clearly(tmp_path):
    fasta_path = tmp_path / "model_organism.fasta"
    interactions_path = tmp_path / "biogrid.tsv"
    id_map_path = tmp_path / "id_map.tsv"
    out_dir = tmp_path / "processed"
    write_test_fasta(fasta_path, ["A", "B", "C", "D"])
    write_text(interactions_path, "Interactor A\tInteractor B\nA\tB\nC\tD\n")
    write_text(id_map_path, "raw_id\tcanonical_id\nA\tA\nB\tB\n")

    missing_map_col_process = run_prep_cli(
        "biogrid",
        "--dataset-name", "missing_mapping_args",
        "--interactions", interactions_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--protein-a-col", "Interactor A",
        "--protein-b-col", "Interactor B",
        "--id-map", id_map_path,
        "--sample-negatives",
        check=False,
    )
    stray_map_col_process = run_prep_cli(
        "biogrid",
        "--dataset-name", "stray_mapping_args",
        "--interactions", interactions_path,
        "--fasta", fasta_path,
        "--out-dir", out_dir,
        "--protein-a-col", "Interactor A",
        "--protein-b-col", "Interactor B",
        "--map-from-col", "raw_id",
        "--sample-negatives",
        check=False,
    )

    assert missing_map_col_process.returncode != 0
    assert (
        "--id-map requires --map-from-col and --map-to-col"
        in missing_map_col_process.stderr
    )
    assert stray_map_col_process.returncode != 0
    assert (
        "--map-from-col and --map-to-col can only be used with --id-map"
        in stray_map_col_process.stderr
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
