# Split Protocol Catalog

This companion to [`split_semantics.md`](split_semantics.md) defines the
task-specific protocol semantics and grouping-relation catalog. The normative
cross-cutting rules, versioning, audit contract, and extension requirements
remain in the core document.

## Status and conformance

Specification status is reported separately from implementation status:

- **stable** means the protocol semantics are normative for the named version;
- **proposed** means the protocol is a research direction and is not yet a
  public benchmark contract;
- **implemented** means public code and regression tests enforce every required
  invariant for the stated input and grouping scope;
- **partially implemented** means related code exists, but at least one
  applicable behavior, artifact, audit, or conformance test is missing;
- **not implemented** means the protocol has no supported implementation.

A CLI option or similarly named splitter is not, by itself, evidence that a
protocol is implemented. Benchmark metadata must record the protocol ID,
protocol version, implementation revision, and grouping configuration actually
used.

| Protocol ID | Specification | Repository implementation |
|---|---|---|
| `ppi.provided.v1` | stable | partially implemented |
| `ppi.random_pair.v1` | stable | partially implemented |
| `ppi.c1.v1` | stable | partially implemented; the splitter covers core invariants for already-canonical inputs |
| `ppi.c2.v1` | stable | partially implemented; core projection invariants support exact-identity plus supplied or generated MMseqs2 sequence-cluster grouping, while complete drop artifacts and versioned identity are pending |
| `ppi.c3.v1` | stable | partially implemented; core projection invariants support exact-identity plus supplied or generated MMseqs2 sequence-cluster grouping, while complete drop artifacts and versioned identity are pending |
| `ptm.provided.v1` | stable | not implemented |
| `ptm.random_site.v1` | stable | not implemented |
| `ptm.protein_disjoint.v1` | stable | partially implemented at the task-helper level; not yet a supported benchmark protocol |
| `ptm.homology_disjoint.v1` | stable | not implemented |

Update this table when conformance changes. Do not weaken or reinterpret a
stable protocol definition merely to match the current implementation.

## Canonical example reconciliation

Splitting operates on canonical task examples, not raw source rows. Preparation
must occur before split assignment and remain separately auditable.

For an unordered PPI pair, canonical reconciliation must:

1. resolve endpoint identifiers under the declared identity policy;
2. order the resolved endpoints deterministically;
3. remove or separately classify self-pairs according to dataset policy;
4. collapse duplicate and reversed observations by default, retaining evidence
   provenance and multiplicity; and
5. resolve, represent explicitly, or exclude contradictory labels under a
   declared policy before splitting.

A source-compatibility protocol MAY preserve repeated evidence rows when its
declared multiplicity policy keeps the canonical example atomic across
partitions and reports the resulting evaluation weighting.

Evidence records may remain one-to-many beneath a canonical example, but all
records contributing to that example must inherit the same partition.
Deduplication, conflict resolution, and source reconciliation must be declared
before assignment and must not change in response to held-out model
performance.

PTM preparation must analogously reconcile the declared protein identity,
residue coordinate system, residue identity, and PTM type before assigning the
canonical protein-site-PTM-type example.

## Protocol invariants and optimization policy

The projection and forbidden-overlap rules below define protocol membership.
Candidate search objectives such as target split fractions, class balance,
retained-example count, degree-distribution similarity, or group-size balance
are implementation policies. They must be recorded, but may evolve without a
protocol-version change so long as they do not alter protocol membership rules,
required invariants, or scientific claims. Any tie-breaking policy must be
deterministic for a recorded seed and implementation revision.

## Benchmark tiers

Tiers organize reporting; they are not protocol identities or a universal
difficulty ranking.

| Tier | Purpose | Current examples |
|---|---|---|
| A | Continuity with common or source-provided evaluation | PPI/PTM provided and random-example protocols; PPI C1 |
| B | Exact unseen-entity generalization | PPI C2/C3 with exact identity; PTM protein-disjoint |
| C | Similarity-controlled generalization | PPI C2/C3 with declared homology groups; PTM homology-disjoint |
| D | Deployment- or mechanism-oriented research | Temporal, species, source, interface, structure, or context holdouts |

A primary benchmark should normally pair a Tier A continuity result with at
least one task-appropriate Tier B or C result. Seed counts, baseline choices,
and threshold sweeps are reporting recommendations rather than conditions for
protocol validity unless a protocol explicitly says otherwise.

## Core PPI protocols

### `ppi.provided.v1`

**Purpose:** reproduce and independently audit a source benchmark's supplied
assignments.

**Prediction unit:** canonical unordered pair.

**Assignment:** source-defined.

**Required behavior:**

- preserve the supplied assignments exactly after canonical identity
  reconciliation;
- require a `source_protocol_id` or dataset-specific descriptor;
- reject unknown split labels;
- never silently repair, rebalance, or reinterpret source assignments;
- audit exact-pair, protein, homology, source, species, and temporal overlap when
  the required metadata are available;
- report whether C1, C2, C3, or another protocol's invariants happen to hold,
  without relabeling the source split unless every required invariant passes.

**Forbidden:**

- the same canonical pair in multiple partitions;
- source duplicates crossing partitions;
- any invariant explicitly claimed by the source protocol.

**Dropped examples:** none solely because of split semantics. Preparation
exclusions remain separately auditable.

### `ppi.random_pair.v1`

**Scientific question:** how well does the model predict randomly held-out
interactions from the same observed interaction population?

**Prediction and assignment unit:** canonical unordered edge.

**Allowed:**

- proteins, sequence clusters, domains, structures, species, and network
  neighborhoods may overlap across partitions.

**Forbidden:**

- canonical pair overlap;
- reversed-pair duplicates crossing partitions;
- train-fitted preprocessing using held-out examples.

**Dropped examples:** none after canonical deduplication.

**Required audit:** held-out examples must be classified by realized endpoint
composition:

- both endpoint proteins seen in training;
- exactly one endpoint protein seen in training;
- neither endpoint protein seen in training.

A random-pair result is a permissive comparison baseline and must not be
reported as evidence of unseen-protein generalization.

### `ppi.c1.v1`

**Scientific question:** can the model infer new interactions among proteins
already represented in the training network?

**Prediction and assignment unit:** canonical unordered edge.

**Projection:** train, validation, and test edges are disjoint; every protein in
validation or test occurs in at least one retained training edge.

**Allowed:** exact protein identities and weaker similarities may overlap by
design.

**Forbidden:**

- canonical edge overlap;
- a validation or test protein absent from retained training edges;
- reversed-pair duplicates crossing partitions.

**Dropped examples:** none under the canonical protocol.

**Does not claim:** unseen-protein, homology-disjoint, topology-independent,
temporal, source-independent, or pretraining-independent generalization.

### `ppi.c2.v1`

**Scientific question:** can the model predict an interaction between one
familiar endpoint group and one unseen endpoint group?

**Prediction unit:** canonical unordered pair.

**Assignment entity:** protein or declared protein group.

#### Two-way projection

- train: train–train edges;
- test: exactly one train endpoint group and one test endpoint group;
- dropped: test–test edges.

The train-side endpoint group of every test edge must occur in at least one
retained training edge.

#### Three-way projection

- train: train–train edges;
- validation: train–validation edges;
- test: train–test edges;
- dropped: validation–validation, test–test, validation–test, and all other
  endpoint patterns outside the protocol.

The train-side group in every held-out edge must occur in retained training
edges. Validation groups must not appear in test examples, and test groups must
not appear in validation examples.

Reserved projection drop reasons are:

- `c2_two_validation_endpoints` for validation–validation edges;
- `c2_two_test_endpoints` for test–test edges, including the two-way heldout
  pattern;
- `c2_validation_test_edge` for validation–test edges.

These names define the row-level contract for future conforming artifacts.
Legacy PPI outputs currently report aggregate discard counts rather than
materializing all projection drops.

**Forbidden:**

- a held-out validation or test group in training edges;
- validation groups in test examples;
- test groups in validation examples;
- assignment groups allocated to more than one partition;
- held-out examples with zero or two train endpoint groups;
- canonical pair overlap.

**Grouping parameterization:**

The public protocol remains `ppi.c2.v1`. The definition of "endpoint group" is
recorded separately, for example:

```text
grouping=protein_identity
identity_key=canonical_accession
```

or:

```text
grouping=sequence_cluster
tool=mmseqs2
tool_version=<exact version>
identity_threshold=0.30
coverage_threshold=0.80
coverage_mode=<declared mode>
```

### `ppi.c3.v1`

**Scientific question:** can the model predict interactions whose two endpoint
groups are both unseen during training?

**Prediction unit:** canonical unordered pair.

**Assignment entity:** protein or declared protein group.

#### Two-way projection

- train: train–train edges;
- test: test–test edges;
- dropped: train–test edges.

#### Three-way projection

- train: train–train edges;
- validation: validation–validation edges;
- test: test–test edges;
- dropped: every cross-partition edge.

The reserved row-level drop reason for a cross-partition edge is
`c3_cross_partition_edge`. Legacy PPI outputs currently report aggregate
discard counts rather than materializing all projection drops.

**Forbidden:**

- assignment-group overlap across partitions;
- retention of cross-partition edges;
- canonical pair overlap;
- violation of the declared grouping relation.

**Interpretation warning:** C3 with exact protein identity is protein-identity
isolated, not automatically homology-, domain-, structure-, interface-,
species-, source-, temporal-, pretraining-, or topology-isolated.

### Grouped C2/C3 instances

A grouped C2 or C3 instance uses the same projection protocol with a stronger
assignment relation.

Required grouping metadata:

- grouping kind and identity key;
- tool and version or a stable source-artifact identifier;
- applicable threshold, coverage, and coverage-mode parameters;
- normalized input hash;
- source file hash;
- group count and size distribution;
- missing-entity policy;
- composite-grouping policy, if any.

The current implementation accepts two mutually exclusive artifact sources:
a supplied `protein_id,cluster_id` CSV or task-independent generation with
MMseqs2 `easy-cluster`. Either source is valid only for grouped C2/C3; random,
C1, and source-provided split protocols reject grouping inputs rather than
silently ignoring them. Automatic construction receives normalized protein IDs
and sequences plus the complete eligible protein universe. PPI endpoints,
labels, and C2/C3 projection rules are not inputs to clustering.

Automatic MMseqs2 defaults are minimum sequence identity `0.30`, coverage
`0.80`, coverage mode `0`, E-value `0.001`, tool-selected sensitivity and
cluster mode, and one thread. Coverage mode controls how `-c` is interpreted:

- `0`: coverage of query and target;
- `1`: coverage of target;
- `2`: coverage of query;
- `3`: target length is at least the threshold fraction of query length;
- `4`: query length is at least the threshold fraction of target length; and
- `5`: the shorter sequence is at least the threshold fraction of the longer.

These meanings follow the
[MMseqs2 parameter definition](https://github.com/soedinglab/MMseqs2/blob/master/src/commons/Parameters.cpp).
The task-independent mapping is content-addressed by normalized sequences and
eligible membership, exact tool version, workflow, effective parameters, and
cache/parser revisions. Cache resolution uses
`PPI_SEQUENCE_CLUSTER_CACHE_DIR`, then
`$XDG_CACHE_HOME/ppi-leakage/sequence_clusters`, then
`~/.cache/ppi-leakage/sequence_clusters`. Entries are locked and atomically
published only after validation; an existing corrupt entry is an actionable
error, not a cache miss.

One grid invocation has one grouping configuration. Threshold sweeps therefore
use distinct versioned grid configurations and run names. A generated result
must be called **MMseqs2 sequence-cluster-disjoint**. The clustering relation
does not establish that every cross-partition protein pair is below the stated
identity threshold; a strict homology-disjoint claim requires a separate
all-vs-all cross-partition audit.

Missing assignment entities must cause an error by default. An explicit
`exclude_unmapped` policy may be supported, but every excluded example must be
recorded and its effect on class and entity distributions reported.

## Core PTM protocols

### `ptm.provided.v1`

**Scientific question:** reproduce and independently audit a source PTM
benchmark.

**Prediction unit:** canonical protein–site–PTM-type example.

**Assignment:** source-defined.

**Required behavior:**

- preserve source assignments after canonical site reconciliation;
- preserve the source negative-label policy;
- audit protein, exact-sequence, homology-cluster, local-context, species,
  temporal, and evidence-source overlap when metadata are available;
- reject duplicate canonical sites crossing partitions;
- do not describe the split as protein- or homology-disjoint unless all required
  invariants pass.

### `ptm.random_site.v1`

**Scientific question:** can the model predict randomly held-out sites drawn
from the same observed protein and context population?

**Prediction and assignment unit:** canonical site example.

**Allowed:**

- the same protein may occur in every partition;
- overlapping local windows may cross partitions;
- homologous proteins, motifs, structures, species, and sources may overlap.

**Forbidden:**

- the same canonical site example in multiple partitions;
- source duplicates of the same canonical site crossing partitions;
- train-fitted preprocessing using held-out labels or examples.

**Required audit:**

- exact protein overlap;
- exact-sequence overlap;
- homology-cluster overlap when a cluster relation is available;
- exact and near-duplicate window overlap when window-derived inputs are used;
- distance between held-out sites and the nearest training site on the same
  protein when coordinates permit it.

This is a permissive comparison baseline, not evidence of new-protein
performance.

### `ptm.protein_disjoint.v1`

**Scientific question:** can the model predict sites on proteins whose declared
identity units were unseen during training?

**Prediction unit:** canonical site example.

**Assignment entity:** exact protein identity by default.

**Projection:** every site inherits its protein's partition.

**Forbidden:**

- the assignment protein in more than one partition;
- site windows generated and split independently of protein assignment;
- canonical sites crossing partitions;
- duplicate canonical sequences crossing partitions when the protocol claims
  sequence-level identity isolation.

**Identity policy:** the manifest must state whether identity means canonical
accession, canonical sequence, or another explicitly defined key. For most
biological generalization claims, exact-sequence aliases should be made atomic.

**Dropped examples:** normally none after complete assignment. Missing or
conflicting sequence records are preparation exclusions, not anonymous split
drops.

### `ptm.homology_disjoint.v1`

**Scientific question:** can the model predict sites on proteins outside the
declared sequence-similarity neighborhood of training proteins?

**Prediction unit:** canonical site example.

**Assignment entity:** sequence-homology cluster.

**Projection:** every site inherits its protein cluster's partition.

**Forbidden:**

- a declared sequence cluster in more than one partition;
- exact-sequence aliases assigned to different clusters;
- clustering performed independently within preexisting partitions;
- site windows split independently of cluster assignment.

**Required reporting:**

- clustering tool and parameters;
- cluster-size distribution;
- fraction of examples retained;
- nearest cross-partition sequence similarity when it can be computed from the
  declared relation or an available audit artifact.

Publication-quality reports SHOULD include either a sensitivity analysis across
grouping thresholds or a continuous cross-partition similarity diagnostic. If
neither is feasible, the report should state the limitation; omission alone
does not invalidate an otherwise conforming split.

### PTM negative-label requirement

For every PTM protocol, the manifest must state whether a negative is:

- experimentally supported negative evidence;
- an unlabeled eligible residue;
- a source-provided negative;
- a sampled subset of eligible unlabeled residues.

The benchmark must not describe unlabeled residues as confirmed biological
negatives.

## Proposed protocol registry

The following are valid research directions, not supported protocol contracts.
Their candidate IDs communicate intended naming only. They must not be accepted
by a public CLI, emitted as an implemented protocol, or used to make benchmark
claims until each has a dedicated specification, real-data implementation,
required audit artifacts, and invariant regression tests.

| Candidate protocol ID | Task | Assignment entity | Scientific question | Important caveat |
|---|---|---|---|---|
| `ppi.temporal.v1` | PPI | dated evidence record or release | predict interactions discovered later | labels and negatives must be reconstructed as of the cutoff |
| `ptm.temporal.v1` | PTM | dated annotation or release | predict sites discovered later | later positives cannot be treated naively as historical negatives |
| `ppi.species_holdout.v1` | PPI | species/clade policy | transfer to unseen species | interspecies pairs need an explicit projection rule |
| `ptm.species_holdout.v1` | PTM | species/clade | transfer to unseen species | homology may still cross species |
| `ppi.evidence_source_holdout.v1` | PPI | study, assay, or source | transfer across curation or assay bias | duplicate biological pairs across sources need reconciliation |
| `ptm.evidence_source_holdout.v1` | PTM | study, assay, or source | transfer across evidence bias | source and label quality may be confounded |
| `ppi.interface_structure_disjoint.v1` | PPI | pair/interface component | generalize to novel interfaces | pair-specific; cannot be represented only as protein groups |
| `ptm.local_context_disjoint.v1` | PTM | site-context component | generalize beyond memorized motifs/windows | site-specific; same protein may cross unless composed with protein isolation |
| `ptm.site_structure_disjoint.v1` | PTM | site-neighborhood component | generalize to structurally novel site environments | structural coverage may create substantial missingness |
| `ptm.ptm_type_holdout.v1` | multi-PTM | PTM type or ontology group | transfer to unseen modification types | incoherent for a single-task classifier with no PTM-type representation |

## Grouping and relation semantics

### Exact protein identity

A protocol must state the identity key. Canonical accession identity and exact
sequence identity are not interchangeable.

Aliases that resolve to the same canonical sequence should normally be unified
before splitting or grouped atomically. If they are not, the manifest must state
why and the audit must report cross-partition exact-sequence overlap.

### Exact sequence identity

Proteins with identical normalized amino-acid sequences form one atomic group.
The normalization and hashing policy must be versioned.

### Sequence-homology clusters

A cluster artifact must record enough provenance to identify or reproduce it:

- tool and version, or a stable published-artifact identifier;
- applicable identity, coverage, and coverage-mode parameters;
- command or configuration when locally generated;
- sequence-database version when one was used;
- normalized input hash;
- representative and transitivity policy;
- cluster count and size distribution;
- unmapped proteins and handling policy.

Clusters must be computed across the full eligible protein universe, not
separately within already-created partitions. A clustering algorithm's
threshold and coverage semantics must be stated exactly; a cluster label alone
does not establish homology isolation.

### Domain, structure, interface, and local-context relations

These relations are not interchangeable:

- domain architecture may be protein-level;
- protein-fold similarity may be protein-level;
- PPI interface similarity is pair- or complex-level;
- PTM local-context similarity is site-level;
- PTM structural-neighborhood similarity is site-level.

The implementation must use the correct assignment entity rather than coercing
a pair- or site-level relation into a protein map.

### Taxonomy

Taxonomy is usually a holdout policy rather than a default generic grouping
source. A single-species dataset would otherwise collapse into one group, and
interspecies PPI examples require explicit projection semantics.

### Composite hard constraints

Multiple must-link relations may be composed by unioning relation edges and
taking connected components.

Required safeguards:

1. relation IDs are namespaced by source;
2. composition is explicit, never automatic;
3. component statistics are audited;
4. giant components produce warnings or infeasibility errors;
5. retention and group-size effects are reported;
6. each relation remains identifiable in the manifest.

Not every measured similarity should become a hard constraint. A protocol may
enforce exact identity and one homology threshold while auditing domains,
structure, or motifs.

## References

1. Park, Y. and Marcotte, E. M. “Flaws in evaluation schemes for pair-input
   computational predictions.” *Nature Methods* (2012).
2. Bushuiev, A. et al. “Revealing data leakage in protein interaction
   benchmarks.” arXiv:2404.10457 (2024).
3. Kapoor, S. and Narayanan, A. “Leakage and the reproducibility crisis in
   machine-learning-based science.” *Patterns* (2023); arXiv:2207.07048.
4. AlQuraishi, M. “ProteinNet: a standardized data set for machine learning of
   protein structure.” *BMC Bioinformatics* (2019); arXiv:1902.00249.
5. Esmaili, F. et al. “A Review of Machine Learning and Algorithmic Methods for
   Protein Phosphorylation Sites Prediction.” arXiv:2208.04311 (2022).

These works motivate the catalog but do not replace task-specific empirical
validation. Bibliographic metadata should be checked for a release or
publication.
