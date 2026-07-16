# Split Semantics and Benchmark Protocols

> Normative core. Protocol-specific definitions and implementation guidance are
> maintained separately so that scientific meaning is not coupled to one
> implementation. Task-specific definitions are maintained in
> [`split_protocol_catalog.md`](split_protocol_catalog.md), and non-normative
> implementation guidance is maintained in
> [`split_engineering_guide.md`](split_engineering_guide.md). This document is the
> authoritative source for the cross-cutting concepts, compatibility rules, and
> extension contract below.

## Status and scope

This document is the normative specification for dataset-splitting semantics in
this repository. Its requirements apply to any result that claims conformance
to a public, versioned protocol ID.

It defines:

- the concepts every public split protocol must specify;
- how prediction examples, assignment entities, grouping relations, and
  partitions relate;
- task/protocol compatibility rules;
- which changes require a protocol-version increment;
- minimum eligibility, negative-construction, audit, and reporting principles;
- the extension contract for new protein-related tasks and split protocols.

Protocol-specific projection rules, thresholds, drop reasons, and required
diagnostics belong in the protocol catalog. Serialization, CLI migration,
implementation, and test procedures belong in the engineering guide.

The repository is intended to benchmark both:

1. common splits, so results remain comparable with prior work; and
2. stricter, task-appropriate splits, so performance reflects meaningful
   generalization rather than avoidable leakage.

The initial tasks are:

- `ppi`: interaction prediction for an unordered protein pair;
- `ptm`: prediction for a candidate residue site on one protein.

Future tasks may include protein function, localization, binding, variant
effect, or structure-related prediction. A future task must satisfy the
extension contract in this document. It must not inherit PPI or PTM split names
only because its implementation appears similar.

### Conformance boundary

This document does not retroactively assert that existing, unversioned split
artifacts conform to a newly named versioned protocol.

In particular:

- existing CLI values such as `random`, `c1`, `c2`, and `c3` describe legacy
  implemented behavior until a catalog entry explicitly maps them to a public
  protocol version;
- a legacy result must not be relabeled as `ppi.c1.v1`, `ppi.c2.v1`, or another
  versioned protocol merely because its high-level split topology is similar;
- a versioned conformance claim requires the protocol's semantic invariants and
  its required manifest, accounting, audit, and reporting fields;
- changes to legacy PPI behavior require regression coverage and an explicit
  compatibility or migration note; scientific changes require a new protocol
  version before the result claims versioned conformance.

The protocol catalog may designate a legacy implementation as conforming after
an explicit review. That review must show that any artifact-format migration
does not alter the scientific split assignment, or it must issue a new protocol
version when the scientific meaning changes.

### Status model

Specification stability and implementation completeness are separate:

- **specification status:** **stable**, **proposed**, or **retired**;
- **implementation status:** **implemented**, **partially implemented**, or
  **not implemented**.

`stable` means that the scientific semantics of the named version are
normative; it does not imply that code exists. `implemented` means the public
code, artifacts, and regression tests satisfy the protocol's applicable
requirements. The catalog records both axes and the current implementation
scope.

Code may provide a legacy split strategy without that strategy yet being a
conforming implementation of a versioned protocol. Proposed protocols may be
implemented experimentally, but they MUST NOT be presented as stable benchmark
contracts.

## Normative language

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are
normative:

- **MUST / MUST NOT**: required for a valid conformance claim.
- **SHOULD / SHOULD NOT**: the default; deviations require a documented reason.
- **MAY**: optional behavior that must still be explicit, reproducible, and
  tested when used.

Statements using lowercase words such as “can,” “normally,” or “recommended”
are explanatory or advisory unless a protocol catalog entry strengthens them.

## Goals and limits

The framework supports comparable common splits and stricter,
deployment-matched protocols; separates model effects from split design;
measures realized leakage and difficulty; reuses genuinely common machinery;
and preserves task-specific projection semantics.

It does not define one universally best split, require every proposed protocol,
force every relation into `protein_id -> group_id`, treat unlabeled examples as
confirmed negatives, or claim that downstream splitting resolves unknown
pretraining exposure.

## Core conceptual model

A split is a scientific protocol, not merely a row-partitioning algorithm.
Every public protocol must define the following concepts.

### Task

A **task** is a prediction problem with a declared example schema, prediction
unit, label semantics, and evaluation contract.

Examples:

- PPI: a binary score for an unordered protein pair;
- PTM: a score for a protein–residue-site–PTM-type example.

### Prediction example and prediction unit

A **prediction example** is one canonical input row after eligibility
filtering, identity normalization, duplicate resolution, and label
reconciliation.

The **prediction unit** is the object for which a model emits one score.

- PPI: one canonical unordered protein pair.
- PTM: one canonical protein–residue-site–PTM-type example.

The stable identity of a prediction example must not depend on a derived
feature such as a tokenized window or embedding-cache key.

### Assignment entity

An **assignment entity** is the indivisible object assigned to `train`, `val`,
or `test` before task examples are projected.

Examples include:

- one PPI edge;
- one protein;
- one sequence-homology cluster;
- one species or taxonomic clade;
- one PTM local-context cluster;
- one PPI interface-similarity component;
- one source study or evidence family;
- one temporal block.

The assignment entity need not be the prediction unit.

### Grouping relation and atomic assignment group

A **grouping relation** declares that two or more assignment entities must be
kept together. Its connected components form **atomic assignment groups**.

A grouping relation MUST be label-independent unless the protocol explicitly
declares and justifies label dependence. Applicable provenance, construction
tool, parameters, thresholds, and artifact identity MUST be recorded.

For transitive clustering, a threshold describes the relation or clustering
procedure; it does not imply that every pair inside a connected component has
that direct pairwise similarity. Reports MUST state the actual grouping method
and SHOULD include realized nearest-neighbor similarity diagnostics.

### Partition assignment

A **partition assignment** maps each atomic assignment group to `train`, `val`,
or `test`.

Partition assignment is distinct from task-example projection. Two tasks may
reuse an assignment engine but project examples differently.

### Projection policy

A **projection policy** maps prediction examples to `train`, `val`, `test`, or
a specific drop reason after assignment entities have been partitioned.

Examples:

- a PTM site inherits its protein's partition;
- a PPI C2 test edge has one train endpoint group and one test endpoint group;
- a PPI C3 test edge has two test endpoint groups;
- a temporal record is assigned by its evidence date.

Projection is task-specific whenever example topology differs.

### Leakage constraint

A **leakage constraint** is a declared relation whose cross-partition
occurrence is forbidden by the protocol.

The manifest and report MUST distinguish:

- **controlled** relations: enforced as hard constraints;
- **audited** relations: measured but allowed;
- **unmeasured** relations: unavailable or not evaluated.

A benchmark MUST NOT claim to control a relation that it only audits.

### Split protocol and protocol instance

A **split protocol** is a stable scientific definition with a task-qualified,
versioned ID, such as `ppi.c3.v1`.

A **protocol instance** is one realization with concrete parameters, for
example:

```text
task=ppi
protocol=ppi.c3.v1
grouping=sequence_cluster
tool=mmseqs2
identity_threshold=0.30
coverage_threshold=0.80
split_seed=17
```

Display labels such as “C3” MAY be used in prose or figures. A conforming
manifest MUST store the full protocol ID and version.

### Protocol identity and versioning

The following versions are distinct:

- **protocol version**: scientific meaning of a public split;
- **manifest schema version**: serialization format of split artifacts;
- **task schema version**: canonical example schema;
- **grouping artifact version**: relation construction and parameters;
- **implementation revision**: code revision used for the instance.

A change that alters allowed or forbidden overlap, projection membership,
eligibility, negative-label meaning, or the generalization claim MUST increment
the protocol version.

Candidate search, optimization, and tie-breaking MAY evolve under the same
protocol version when they only choose among semantically valid instances. The
manifest MUST identify the selection policy and implementation revision. If a
protocol explicitly freezes a selection policy, or a change alters candidate
validity rather than selection among valid candidates, the protocol version
MUST change.

A serialization-only migration increments the manifest schema version. A
refactor that preserves all observable semantics and deterministic assignments
does not increment the protocol version.

Benchmark tiers and task-specific canonical schemas are defined in the protocol
catalog. Tiers are reporting aids, not protocol identities or a universal
difficulty ordering. Realized difficulty MUST be supported by diagnostics, not
inferred from a protocol name.

Canonical example identity MUST be independent of derived features. Duplicate
or conflicting observations MUST be reconciled, represented explicitly, or
made atomic under the task's declared policy before assignment. Detailed PPI
and PTM identity rules belong in the catalog.

## Task/protocol compatibility

Public protocol identifiers MUST be task-qualified. A shared assignment
primitive does not make public split names interchangeable.

Compatibility MUST be validated before expensive loading, clustering, or
candidate search. An unsupported pairing MUST fail with an actionable error;
it MUST NOT silently fall back to a random or similarly named protocol.

Short CLI values MAY remain as backward-compatible aliases when the task is
unambiguous. An alias is an interface convenience, not a protocol identity.
Until a catalog entry formally registers a mapping, legacy output MUST retain
its legacy identity rather than inventing a versioned one.

## Eligible cohort and duplicate timing

The **eligible cohort** is the complete set of examples and entities that may
be assigned by one protocol instance.

Before construction, a conforming instance MUST identify its source/evidence
snapshot, inclusion rules, canonical identity policy, duplicate/conflict
policy, candidate universe, and label policy. Duplicate reconciliation occurs
before assignment. Cross-source identity MUST be resolved or audited when the
same biological example may occur in multiple sources.

Eligibility MUST NOT change in response to held-out model performance, and
held-out examples MUST NOT be removed merely because they are difficult or
inconvenient.

## Candidate construction and selection

Every conforming candidate MUST have disjoint, nonempty required partitions;
account for each eligible example exactly once as retained or specifically
dropped; preserve hard groups and canonical-example atomicity; satisfy all
protocol invariants and declared sufficiency requirements; and be deterministic
for fixed inputs, parameters, and seeds.

When a construction algorithm targets partition sizes, it MUST declare its
feasibility rule or tolerance before candidate optimization. Construction MUST
NOT weaken the scientific definition to reach a requested fraction.

Candidate search MUST record its objective and deterministic tie-break. It MUST
NOT use model performance on validation or test data to choose a split. Changes
to the search policy normally identify a new implementation revision, not a new
protocol version, unless they change which candidates are semantically valid.

Split, model, negative-sampling, and stochastic-grouping seeds MUST remain
distinct wherever those sources of randomness exist.

## Negative construction

Negative construction is part of the benchmark protocol, not incidental data
plumbing.

When a task uses negative or unlabeled examples, the manifest MUST state their
meaning and record the applicable evidence snapshot, candidate universe,
construction timing, sampling rule, and seed.

Generated negatives MUST obey the same partition and grouping constraints as
positive examples. For entity- or cluster-disjoint protocols, negatives should
normally be generated within the allowed projection space after entity
assignment.

Unobserved PPI pairs and unlabeled PTM sites MUST NOT be described as confirmed
negatives without supporting evidence. Temporal protocols account for
annotations discovered after the cutoff. Task-specific policies belong in the
catalog.

## Preprocessing and derived artifacts

Label-independent operations MAY use the eligible cohort to establish
cross-partition relations; their applicable provenance MUST be recorded.

Feature construction and task adaptation MUST follow the evaluation contract.
Vocabularies, normalization, feature selection, class weights, and learned
task-specific representations are normally fit on training data. Calibration,
classification thresholds, early stopping, and hyperparameter selection MAY
use a designated validation or development partition. Test data MUST NOT
influence any fitted operation or model-selection decision.

A frozen deterministic pretrained encoder MAY encode all partitions when its
checkpoint and tokenizer are fixed, it performs no task-label or cross-example
adaptation, and cache provenance is complete. When pretrained models are used,
known or unknown pretraining exposure MUST be reported separately from
downstream split leakage.

A fine-tuned or task-adapted encoder is split-specific and MUST NOT reuse
learned representations across incompatible training partitions.

## Split-result and accounting contract

A conforming result MUST provide, directly or through fingerprinted references,
the retained partitions; any projection drops and their reasons; partition or
group assignments needed for audit; a manifest; an audit; and applicable
protocol, task, grouping, and manifest-schema versions. Optional partitions,
such as validation, may be explicitly absent.

Every eligible example MUST be accounted for exactly once as retained or
dropped. Every retained or dropped example MUST preserve a stable canonical
example identity; available source-record identities SHOULD also be preserved.
Drop records MUST use the most specific available reason rather than a generic
`dropped` value.

Physical filenames and serialization formats belong to the manifest schema,
not the scientific protocol. Legacy filenames do not, by themselves, establish
versioned conformance.

## Required manifest and audit principles

A conforming manifest MUST identify the task and schema, protocol ID and
version, prediction and assignment units, projection policy, dataset and
eligible cohort, applicable grouping provenance and seeds, manifest-schema
version, and implementation revision. When candidate search is used, it also
records its feasibility rule, trial count, objective, and tie-break.

Counts MUST cover input, eligibility, retained partitions, drops by reason,
actual fractions, task-specific entities and groups, and a label summary
appropriate to the task.

Audits MUST name hard invariants and report whether each passed. Universal
checks include example-identity overlap, canonical-duplicate overlap,
assignment-group overlap, pairwise-disjoint masks, nonempty required
partitions, and complete row accounting. Source-record overlap MUST be checked
when stable source identities are available.

A result with a failed hard invariant MUST NOT be used or reported as a valid
instance of that protocol.

Audited-but-allowed relations SHOULD report counts or distributions. Failed or
nonzero checks SHOULD include a bounded set of representative identities,
normally no more than ten, without exposing sensitive data.

Task-specific required diagnostics belong in the protocol catalog.

## Failure behavior

An infeasible requested split MUST:

- report failed constraints;
- report relevant counts, components, and retention limits;
- suggest scientifically valid parameter changes when possible;
- exit without silently changing protocols, weakening thresholds, relaxing
  invariants, or falling back to random splitting.

## Benchmark reporting

Every result table or model card claiming protocol conformance MUST identify the
complete protocol instance, not only a display label.

Reports SHOULD include:

- model performance and relevant seed variation;
- retained, dropped, label, entity, and assignment-group summaries;
- controlled, audited, and unmeasured leakage dimensions;
- applicable negative construction and realized similarity diagnostics;
- split-construction cost and model provenance.

Nominal labels such as “C3” or “homology-disjoint” are not sufficient evidence
of difficulty. Reports should measure the relations relevant to the scientific
claim, such as nearest sequence, structure, interface, or context similarity;
endpoint novelty; degree shift; species and source overlap; and temporal lag.

A downstream split cannot guarantee that a pretrained model did not see test
sequences or homologs during pretraining. Reports MUST distinguish downstream
split controls from known, audited, or unknown pretraining exposure.

Repeated inspection of test performance turns the test set into a development
resource. Public benchmarks SHOULD use immutable manifests, development
partitions, and a locked confirmatory test workflow.

## Extension contract

### Adding a split protocol

A protocol specification defines its ID, version, specification status,
supported task, scientific question, prediction and assignment units, grouping
and projection, allowed/forbidden/audited overlap, drop rules, applicable
eligibility and negative construction, required diagnostics, and known
limitations. Feasibility and selection policy belong in the protocol only when
they affect its scientific meaning.

### Adding a task

A new task defines its task ID and schema version, stable example identity,
entity roles, label and uncertainty semantics, duplicate policy, candidate and
negative-label policy, and compatible protocols. It MUST NOT reuse a public
split ID unless its prediction unit, entity roles, projection, and scientific
claim are genuinely identical.

The preferred extension pattern is to reuse relation builders and atomic-group
assignment only where their semantics match, while implementing a task-specific
projection adapter and task-specific audits. Only scientifically compatible
task/protocol combinations may be registered.

### Promotion to implemented

An implementation may be marked `implemented` only after:

- its semantics are present in the protocol catalog;
- deterministic construction passes;
- every required hard invariant is emitted and passes;
- every eligible example is retained or receives a reasoned drop code;
- unsupported task combinations fail early;
- an end-to-end smoke test passes;
- compatibility with existing artifacts is tested or an intentional migration
  is documented;
- documentation, CLI help, and emitted identity agree.

Promotion MUST include an explicit conformance review. The review should link
the protocol definition, invariant tests, representative manifest, and any
legacy-alias mapping.

## Scientific rationale

This specification follows broad methodological principles:

- pair-input evaluation must distinguish seen–seen, seen–unseen, and
  unseen–unseen entity regimes;
- biological isolation may require sequence, structure, interface, temporal,
  source, taxonomic, or context relations matched to the task;
- sequence identity is one leakage axis, not a complete proxy for functional or
  structural novelty;
- uncertain negatives and evidence timing are part of benchmark semantics;
- split construction should be immutable, auditable, versioned, and reported
  as a first-class scientific artifact.

The protocol catalog carries the task-specific bibliography. References
motivate the design but do not replace empirical validation.
