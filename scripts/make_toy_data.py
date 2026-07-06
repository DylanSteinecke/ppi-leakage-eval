import argparse
import logging
import random

import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
MOTIFS = ["ACDEFG", "KLMNPQ", "RSTVWY", "GHIKLM", "NPQRST"]
LOGGER = logging.getLogger(__name__)


def random_seq(length=120):
    sequence = "".join(random.choice(AA) for _ in range(length))

    return sequence


def insert_motif(seq, motif):
    i = random.randint(0, len(seq) - len(motif))
    sequence = seq[:i] + motif + seq[i + len(motif):]

    return sequence


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=int, default=200)
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    proteins = []
    seqs = {}
    families = {}

    for i in range(args.n_proteins):
        pid = f"P{i:04d}"
        fam = random.randrange(len(MOTIFS))
        seq = insert_motif(random_seq(), MOTIFS[fam])

        proteins.append(pid)
        seqs[pid] = seq
        families[pid] = fam

    rows = []
    seen = set()

    while len(rows) < args.n_pairs:
        a, b = random.sample(proteins, 2)
        key = tuple(sorted([a, b]))

        if key in seen:
            continue

        seen.add(key)

        # Toy rule: proteins from same motif family interact.
        label = int(families[a] == families[b])

        rows.append({
            "protein_a": a,
            "protein_b": b,
            "label": label,
        })

    pd.DataFrame(rows).to_csv("processed/toy_pairs.csv", index=False)

    with open("processed/toy_sequences.fasta", "w") as f:
        for pid, seq in seqs.items():
            f.write(f">{pid}\n{seq}\n")

    LOGGER.info("Wrote toy_pairs.csv and toy_sequences.fasta")


if __name__ == "__main__":
    main()

    # python make_toy_ppi_data.py
