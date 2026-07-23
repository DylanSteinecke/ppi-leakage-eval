# Change Checklists

This project has several connected layers: Python CLI arguments, shell runner
environment variables, run directories, split metadata, `invocations.jsonl`,
diagnostics, metrics summaries and plots, benchmark aggregation, and pytest
tests. When one layer changes, related layers may need updates too.

Before making a change, scan the section matching the thing being changed.
After implementation, rescan it before running tests.

`resolved_args` is the complete machine-readable record of parsed CLI state.
Only promote fields to top-level `split_metadata.json` fields,
`benchmark_manifest.csv` columns, or `benchmark_summary.csv` context columns
when they are useful for auditability, grouping, or cross-run comparison.

## When Adding or Changing a CLI Flag

- Update `argument_parser()` in `src/ppi_benchmark/cli/train.py`.
- Decide whether the flag should also be exposed through the relevant example
  runner under `scripts/` as an environment variable.
- Confirm the flag appears correctly in `resolved_args` inside
  `split_metadata.json`.
- Add or update top-level `split_metadata.json` fields only if the flag is
  important for auditability or benchmark comparison.
- Confirm `invocations.jsonl` captures the invocation correctly.
- Update `src/ppi_benchmark/cli/aggregate.py` if the flag should appear in
  `benchmark_manifest.csv` or `benchmark_summary.csv`.
- Update `ppi_benchmark.reporting.summaries.SUMMARY_GROUP_COLUMNS` only if metrics
  summaries should be grouped by the new flag.
- Add or update pytest coverage for the flag's behavior.
- Update README or runner examples if users need to know about the flag.

Do not manually duplicate every CLI flag everywhere. `resolved_args` is the
complete record. Top-level metadata, manifest columns, and summary columns
should remain curated.

## When Adding a Runner Environment Variable

- Update the relevant example runner under `scripts/`.
- Keep env vars documented near the top of the script.
- Forward the value to the Python CLI if it changes model, split, output, or
  aggregation behavior.
- Decide whether the resulting CLI arg should appear in top-level metadata,
  manifest columns, or summary columns.
- Ensure `invocations.jsonl` captures the resolved command.
- Add or update tests if behavior changes.

Current runner env vars include:

- `PAIRS`
- `FASTA`
- `PROTEIN_METADATA`
- `OUT_DIR`
- `K`
- `TRAIN_SIZE`
- `VAL_SIZE`
- `NUM_RERUNS`
- `MAX_ITER`
- `EXECUTION_ID`
- `RUN_STAMP`
- `AGGREGATE_RESULTS`
- `GENERATE_TOY_DATA`
- `YEAST_DATASET_NAME`
- `PREPARE_YEAST_DATA`
- `BIOGRID_ARCHIVE`
- `BIOGRID_ARCHIVE_MEMBER`
- `YEAST_FASTA`
- `NEGATIVE_RATIO`
- `NEGATIVE_SAMPLING_POLICY`
- `NEGATIVE_SAMPLING_SEED`

## When Adding or Changing Negative Construction

- Read the negative-construction rules in `docs/split_semantics.md` and the
  task policy in `docs/split_protocol_catalog.md`.
- Declare the negative-label meaning, evidence snapshot, candidate universe,
  construction timing, sampling policy, ratio, and independent seed.
- Update the runtime sampling specification and relevant dataset loaders; do
  not select a policy implicitly from optional metadata or silently fall back
  when required metadata are missing.
- Emit structured prepared-dataset metadata and propagate its identity into
  split metadata and aggregation fields used for cross-run comparison.
- Verify generated examples obey the selected split's grouping and projection
  constraints, and keep source-provided negatives distinct from generated
  policies.
- Add deterministic, infeasible-space, missing-metadata, positive-overlap, and
  end-to-end artifact tests for every supported policy.
- Update the README, example-runner variables, and task catalog entry. Update
  `split_semantics.md` only if the cross-protocol contract changes.

## When Adding or Changing a Split Strategy or Protocol

- Decide whether the change is a legacy implementation option, a new public
  protocol, or an implementation of an existing cataloged protocol. A CLI name
  alone does not establish protocol conformance.
- For a public protocol, add or update its entry and implementation status in
  `docs/split_protocol_catalog.md` before claiming the protocol ID in outputs.
- Determine whether the change affects scientific meaning. Update the protocol
  version for changes to allowed or forbidden overlap, projection membership,
  eligibility, negative-label meaning, or the generalization claim.
- Update `docs/split_semantics.md` only when a cross-protocol concept,
  conformance rule, or extension requirement changes.
- Update `docs/split_engineering_guide.md` only when architecture, artifact,
  migration, or reusable conformance-test guidance changes.
- Add the strategy specification in
  `src/ppi_benchmark/splitting/protocols.py` when it is exposed through the
  current CLI, and update dispatch only when routing changes.
- Implement split creation in the task-appropriate adapter or splitter; do not
  infer task compatibility from a similar strategy name.
- Validate task/protocol compatibility before expensive loading or grouping.
- Preserve provided and legacy behavior, or document and test the intentional
  compatibility migration.
- Emit the protocol ID, implementation revision, grouping configuration,
  accounting, and required audits when claiming versioned conformance.
- Add tests for canonicalization, projection, hard invariants, drop reasons,
  two- or three-way behavior as applicable, and actionable failure cases.
- Update diagnostics when leakage assumptions change and aggregation only for
  fields needed in cross-run comparisons.
- Add the strategy to `scripts/_run_ppi_benchmark_grid.sh` only if it should be
  part of the default benchmark grid.

## When Adding a Split Diagnostic

- Update `src/ppi_benchmark/splitting/diagnostics.py`.
- Store the diagnostic under `split_metadata["diagnostics"]`.
- Do not create a separate diagnostics artifact unless there is a strong
  reason.
- Ensure all diagnostic values are JSON-serializable native Python types.
- Add capped examples when useful for debugging.
- Update `ppi_benchmark.cli.aggregate.DIAGNOSTIC_COLUMNS` if the diagnostic
  should appear in benchmark-level CSVs.
- Add tests in `tests/test_split_diagnostics.py`.
- Update plots or tables only if they display the diagnostic.

Diagnostics should be non-fatal by default. They report leakage risk; they
should not fail runs unless a future explicit validation mode is added.

## When Renaming or Removing Metadata or Diagnostic Fields

- Search the repo for the old field name.
- Update tests, aggregation, plots, docs, and examples.
- Decide whether backward compatibility is needed.
- If backward compatibility is not supported, make failures clear.
- Prefer renaming fields early before they become stable.
- If a field is part of `DIAGNOSTIC_COLUMNS`, `MANIFEST_COLUMNS`, or
  `SUMMARY_CONTEXT_COLUMNS`, update those lists.
- If the field appears in published outputs or downstream notebooks, consider
  adding a short migration note.

## When Adding a New Output Artifact

- Add the path to `OutputPaths` in `src/ppi_benchmark/cli/train.py`.
- Define the canonical filename near the other output filename constants.
- Update `prepare_outputs()`.
- Decide whether the artifact exists always, only with validation, only with
  test evaluation, or only when plots are enabled.
- Add the artifact to immutable-run integrity and role classification.
- Decide whether historical aggregation should discover or normalize it.
- Add the path to `split_metadata.json` if useful for auditability.
- Update benchmark aggregation only if the artifact contributes to cross-run
  summaries.
- Add tests asserting the artifact is present or absent in the right
  conditions.

## When Adding a Model or Classifier

- Update model choices in `src/ppi_benchmark/models.py`.
- Update CLI choices if needed.
- Add fast unit tests or CLI smoke tests.
- Ensure model metadata appears in metrics rows.
- Ensure mixed model selections work within one immutable invocation.
- Avoid heavy tests for expensive models; use small smoke tests.

## When Adding a Feature Type

- Update `src/ppi_benchmark/features.py`.
- Update `FEATURE_CHOICES`.
- Ensure feature extractors fit only on train data and only transform val/test.
- Add or update leakage tests for train-only feature fitting.
- Ensure feature name creation remains stable.
- Update runner feature grids only if the feature should be included by default.
- Add tests using small data.

## When Changing Split Metadata

- Update `src/ppi_benchmark/splitting/artifacts.py`.
- Keep `split_assignments.csv` minimal.
- Keep `dropped_pairs.csv` minimal.
- Keep `split_metadata.json` as the main audit artifact.
- Preserve `source_row_index`.
- Ensure metadata remains JSON-serializable.
- Update run-fingerprint identity when the field affects cross-run comparison.
- Update tests that read `split_metadata.json`.
- Update benchmark aggregation if new metadata should become a manifest or
  summary column.

## When Changing Benchmark Aggregation

- Update `src/ppi_benchmark/cli/aggregate.py`.
- Decide whether the new field belongs in:
  - `MANIFEST_COLUMNS`
  - `SUMMARY_CONTEXT_COLUMNS`
  - `DIAGNOSTIC_COLUMNS`
- Ensure aggregation reads from `split_metadata.json` or `invocations.jsonl`,
  not from a second source of truth.
- Add tests for aggregation output columns.
- Confirm `scripts/_run_ppi_benchmark_grid.sh` still calls aggregation
  correctly when `AGGREGATE_RESULTS=1`.

## When Changing the Shell Runner

- Update `scripts/_run_ppi_benchmark_grid.sh`.
- Keep environment variables documented near the top of the script.
- Ensure extra user args still pass through to the Python CLI.
- Ensure run directories remain stable and timestamped.
- Ensure every generated configuration uses a distinct immutable run directory.
- Ensure benchmark aggregation still runs when `AGGREGATE_RESULTS=1`.

## When Preparing Future PTM Support

- Keep shared infrastructure task-neutral where practical:
  - run directories
  - source row tracking
  - metadata
  - invocations
  - generic diagnostics helpers
  - benchmark aggregation
- Add PTM-specific code as task-specific wrappers rather than forcing PPI
  abstractions onto PTM.
- Future PTM diagnostics will likely need:
  - exact site overlap
  - same protein overlap
  - same residue overlap
  - PTM type distribution
  - residue type distribution
  - local window overlap
  - protein-family or sequence-identity overlap later
- Do not implement PTM diagnostics in this checklist doc.

## Testing Checklist

Before merging behavior changes, run tests using the project test environment:

```bash
conda run -n ppi python -m pytest
```

If already inside the `ppi` environment, this is also fine:

```bash
python -m pytest
```

For CLI-affecting changes, add tests that cover:

- required/removed flags
- run-dir outputs
- validation vs test behavior
- atomic run claims and immutable-directory failures
- completion fingerprint integrity
- metadata fields
- benchmark aggregation if relevant

For split-affecting changes, add tests that cover:

- generated splits
- provided split columns
- label balance / binary label validation
- leakage diagnostics
- split metadata

Keep this document short enough to stay useful.
