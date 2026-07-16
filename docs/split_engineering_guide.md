# Split Engineering Guide

> **Status:** Draft, non-normative target architecture. This guide does not
> describe every currently implemented API or artifact. For legacy behavior,
> the code and tests remain authoritative.

This document provides implementation guidance for the split framework.
Scientific claims and cross-protocol semantics are defined in
[`split_semantics.md`](split_semantics.md). Protocol-specific projection rules,
drop reasons, required invariants, and required diagnostics are defined in
[`split_protocol_catalog.md`](split_protocol_catalog.md). A similarly named
legacy splitter or CLI option does not establish conformance with a cataloged
protocol.

If this guide conflicts with the normative split semantics or protocol catalog,
the normative document takes precedence. Repository-wide agent and coding
instructions remain in `AGENTS.md`.

## Architectural boundary

The implementation should preserve four conceptual layers. These are
responsibility boundaries, not requirements to create one class or module per
layer.

### Relation and grouping construction

Responsibilities:

- normalize and reconcile biological identifiers across sources;
- construct label-independent relations;
- namespace source-local relation and cluster IDs to prevent accidental ID
  collisions, without preventing the same canonical biological entity from
  being linked across sources;
- record tool versions, parameters, input hashes, and relation provenance;
- detect missing and multiply assigned entities;
- compose must-link relations only when explicitly requested;
- expose component-size and realized-similarity diagnostics.

This layer must not decide how PPI edges or PTM sites are projected.

### Partition assignment

Responsibilities:

- assign atomic groups to train, validation, and test;
- generate deterministic candidates for fixed inputs, parameters, and seeds;
- enforce hard partition constraints;
- honor declared split-size feasibility ranges;
- score only valid candidates;
- use a declared deterministic tie-break;
- report infeasibility rather than silently weakening the protocol.

This layer may be shared when tasks genuinely assign the same kind of entity
under the same constraints. Assignment code must not infer task semantics from
dataframe column names.

### Task-specific projection

Responsibilities:

- validate the task example schema;
- extract task-specific entity roles;
- map examples to train, validation, test, or one specific drop reason;
- preserve stable example and source-record identities;
- enforce task-specific invariants;
- generate task-specific diagnostics.

PPI edge projection and PTM site projection must remain separate. Similar
partition topology does not make their prediction units or projection rules
interchangeable.

### Audit and reporting

Responsibilities:

- universal accounting, identity-overlap, and disjointness checks;
- task-specific invariant checks;
- controlled, audited-but-allowed, and unmeasured overlap summaries;
- dropped-example counts and bounded representative examples by reason;
- reproducibility and provenance metadata;
- warnings about uncontrolled dimensions and pretrained-model exposure.

Hard invariant failures must make the result unusable as a conforming protocol
instance. Diagnostic overlap that the protocol permits should be reported
without being mislabeled as a failed hard constraint.

## Minimal registry boundary

A small registry uses this compatibility contract:

```python
@dataclass(frozen=True)
class SplitStrategySpec:
    task_id: str
    strategy_name: str
    protocol_id: str
    protocol_version: int
    prediction_unit: str
    assignment_entity: str
    projection_kind: str
    default_grouping_kind: str | None = None
    allowed_grouping_kinds: tuple[str, ...] = ()
    projection_drop_reasons: tuple[str, ...] = ()


def get_split_strategy(
    task_id: str,
    strategy_name: str,
) -> SplitStrategySpec: ...


def registered_split_strategies(task_id: str) -> tuple[str, ...]: ...
```

The registry maps legacy strategy names to catalog definitions for compatibility
and dispatch. Referencing a catalog ID does not by itself claim that a legacy
result fully conforms to that protocol. Implementation availability and
conformance status remain in the catalog or a conformance report rather than in
the immutable scientific specification. A list of allowed projection drop
reasons is safer than a boolean because every dropped example needs one
specific reason.

Do not add a generic task projection adapter or result wrapper until a second
end-to-end task implementation needs that boundary.

The required conceptual flow is:

```text
canonical eligible examples
        ↓
relation/group construction
        ↓
partition assignment
        ↓
task-specific projection
        ↓
invariant validation and audit
```

Avoid:

- spreading `if task == ...` branches through shared algorithms;
- making every splitter accept arbitrary dataframes;
- treating all relation types as `protein_to_group` mappings;
- storing scientific semantics only in executable registry code;
- introducing a broad plugin system before two concrete implementations need
  it.

## Engineering test matrix

This section describes how to test implementations. The normative semantics and
protocol catalog define which invariants a protocol requires. If an invariant
is repeated here, the normative definition remains authoritative.

Tests should independently verify observable behavior rather than reproduce the
production algorithm. Prefer small deterministic fixtures, explicit expected
identities, and direct assertions on emitted artifacts and audits.

### Universal tests

Every implemented protocol needs tests for:

- deterministic output under fixed inputs, parameters, and seeds;
- correct use and recording of randomness, including a fixture that produces
  seed-dependent candidates where the algorithm is stochastic, without
  requiring every pair of seeds to produce different final assignments;
- pairwise-disjoint retained masks;
- complete accounting of every eligible example as retained or dropped once;
- stable canonical example identities and available source-record identities;
- atomicity of every declared hard grouping relation;
- early failure for incompatible task/protocol combinations;
- early failure for invalid schemas;
- declared behavior for missing assignments;
- required semantic metadata and artifact fingerprints;
- declared negative-label meaning, candidate universe, construction timing,
  and sampling seed where applicable;
- generated negatives, when used, obeying the same grouping and projection
  constraints as positives;
- training-only fitting for learned feature transforms and task-adapted
  encoders, with calibration, thresholds, early stopping, and hyperparameters
  restricted to the declared training/development partitions;
- complete checkpoint, tokenizer, and cache provenance when frozen encoders are
  used;
- no silent fallback or constraint weakening;
- informative infeasibility errors;
- two-way and three-way behavior where the protocol supports them.

Every behavior-changing bug fix needs a focused regression test.

### Protocol-specific tests

Do not duplicate the catalog's invariant lists here. For each implemented
protocol, tests should derive their assertions from its catalog entry and
cover:

- every allowed retained projection pattern;
- every forbidden overlap or endpoint pattern;
- every protocol-specific drop reason;
- each required audit when its prerequisite metadata are present;
- the declared canonicalization and negative-label policy;
- both a valid instance and a representative invalid or infeasible instance.

Grouped protocols additionally need tests for full-universe grouping,
transitive components, missing mappings, and fingerprint changes when grouping
provenance changes. Advanced protocols need relation-appropriate fixtures, such
as cutoff censoring for temporal splits and atomic site components for context
or structure splits.

### Adversarial fixtures

Small synthetic fixtures should cover:

- isolated proteins;
- hubs and severe degree imbalance;
- duplicate and reversed PPI edges;
- conflicting duplicate labels;
- giant homology components;
- one protein with many PTM sites;
- overlapping PTM windows;
- rare PTM types and class imbalance;
- insufficient groups for a three-way split;
- impossible class-presence or size constraints;
- mixed-species interactions;
- source-local group-ID collisions;
- canonical entities repeated across sources;
- missing timestamps and assignments.

Tests for adversarial cases should assert the documented result or a specific,
actionable failure. They should not accept silent dropping, fallback to a weaker
protocol, or nondeterministic behavior.

## Recommended migration sequence

This sequence records implementation dependencies, not current completion
status or a release commitment. Verify current state in code and tests before
starting a step; active scheduling belongs in the issue tracker. Reorder only
when doing so preserves the same compatibility and conformance boundaries.

1. Preserve legacy PPI random/C1/C2/C3 behavior with golden regression tests.
2. Define the protocol catalog entries before emitting task-qualified protocol
   identities or claiming conformance.
3. Add task-qualified protocol metadata and early compatibility validation.
4. Keep existing PPI projection behavior intact behind a PPI-specific adapter.
5. Implement and validate one real PTM dataset schema and loader.
6. Implement PTM provided, random-site, and protein-disjoint protocols.
7. Add a minimal frozen-encoder PTM vertical slice and CPU smoke test.
8. Implement PTM homology-disjoint splitting with versioned clustering
   provenance.
9. Extract shared group-assignment code only after PPI and PTM demonstrate real
   semantic and implementation duplication.
10. Add advanced leakage dimensions one at a time with dedicated data,
    provenance, invariant tests, and benchmark experiments.

Repository commands, coding style, development-agent instructions, and generic
definitions of done belong in `AGENTS.md` or focused change checklists. This
guide should remain concerned with split-framework architecture, implementation
testing, and migration boundaries.
