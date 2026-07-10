# PPI Leakage
Testing how robust models and datasets for protein-protein interactions are to leakage.

## Tests
Run the lightweight regression suite with:

```bash
python -m pytest
```

For connected-change checklists, see
[`docs/change_checklists.md`](docs/change_checklists.md).

## Prepare the yeast BioGRID data

The local UniProt FASTA uses headers such as `sp|P04387|GAL80_YEAST`, while
BioGRID stores the matching accession as `P04387`. Prepare the current local
files with:

```bash
python scripts/prepare_ppi_dataset.py biogrid \
  --dataset-name biogrid_yeast_physical \
  --interactions input/BIOGRID-ORGANISM-LATEST.tab3.zip \
  --archive-member BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-5.0.259.tab3.txt \
  --fasta input/UP000002311_559292.fasta \
  --fasta-id-format uniprot_accession \
  --protein-a-col "SWISS-PROT Accessions Interactor A" \
  --protein-b-col "SWISS-PROT Accessions Interactor B" \
  --experimental-system-type-col "Experimental System Type" \
  --allowed-system-types physical \
  --ambiguous-id-policy drop \
  --sample-negatives \
  --negative-ratio 1.0 \
  --out-dir processed
```

The loader reads the large archive in chunks. Sampled negatives are unobserved
protein pairs, not experimentally confirmed non-interactions. The archive
member includes a BioGRID release number and must be updated when the `LATEST`
download changes.
