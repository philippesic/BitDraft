# Results and closeout

This file defines Binary Eagle's result presentation and retention rules. The reusable
procedure is [close-results](../skills/close-results/SKILL.md). Read it when closing an
investigation; it grants no additional run, publication or deletion authority.

## Document roles

- `AGENTS.md`: rules every agent needs; `README.md`: introduction, setup and navigation.
- `docs/`: how the project works now and how to change it. Replace obsolete instructions.
- `tasks/`: unfinished multi-session work, following the [task convention](../tasks/README.md).
- `results/`: durable conclusions, comparisons and decisions, including negative and
  inconclusive investigations. Routine repairs do not automatically need a result.
- [FINDINGS](../FINDINGS.md): short conclusions and links, grouped by topic when useful.
  Preserve existing finding IDs and anchors; keep detailed explanations in result pages.
- Git: implementation history. Do not maintain a running commit diary.

## Result layout

Use `results/descriptive-result-name/result.md` for one coherent question or investigation,
not one folder per run, commit or session. Basic sections are:

1. A title stating the conclusion and an opening explanation of why it matters.
2. `Outcome`: useful measurements, a chart or a small table when they help understanding.
3. `Scope and limitations`: conditions tested and what the evidence cannot establish.
4. `Decision`: retain, revise, discard or unresolved, with rationale and whether proposed
   or already decided. A decision in a document does not itself authorize execution.
5. `Evidence and reproduction`: measurements, executed source, dependencies and artifact
   references needed to assess the claim or repeat the work; state missing information.

Use `result.json` for structured facts and figure inputs only when useful. Small tables
and optional `chart.svg` assets stay beside the result. Retain numerical data and an
explicit regeneration recipe for every figure. Regenerating a figure is different from
repeating an experiment. Link archived supporting reports under `results/sources/` only
when necessary; create no empty archive or placeholder folders.

For measurements, identify model, target, hardware, data roles, recipe/budget, timing
definition and repetitions as applicable. Figures state units, cohort and aggregation.
Do not mix clocks or workload eras. Display uncertainty only when supported by data;
rounded summary values do not supply per-bar error bars.

## Evidence and artifacts

Small immutable provenance and execution receipts already in `evidence/` retain their
identities and links. New compact result measurements belong beside their result;
archived supporting reports belong in `results/sources/`. Neither is a raw-run store.
Large logs, datasets, weights and frozen execution environments belong in an external
artifact location identified precisely by the result. The ignored `artifacts/` directory
is for disposable local previews and caches, not the sole home of valuable evidence.

Check claims against source records. Label imported, recomputed, rerun and unverified
claims accurately. Link the exact executed source, including dirty changes and runtime
identities where needed. A report's repository revision is not proof of executed source.
Legacy imports with missing provenance remain usable within their stated limits, but
are not evidence that supporting artifacts can safely be removed.

## Lifecycle and retention

Open task → investigate → write result if warranted → update current docs → prune obsolete
code within scope → update FINDINGS → close task. Use the shared procedure for closeout.
Delete completed task files; rewrite current state while work remains instead of appending
a diary. Retain tools and tests only while they still serve the project.

Before removing implementation or artifacts, identify paths, ownership, recoverable source,
dependent inputs, evidence, backup location and verification, and regeneration limits.
Preserve what the result needs before deleting anything. Existing authorization governs
cleanup; a local commit alone is not a remote backup. No historical artifact deletion is
authorized by adoption of this convention. Do not rewrite Git history during closeout.

Raw execution records remain immutable. Correct a result explicitly with a dated note;
a materially changed question or body of evidence gets a new result linked to the earlier
one. FINDINGS points to the current conclusion while conflicting evidence stays discoverable.

## Examples

[Fixed-data continuation](../results/he4-fixed-data-continuation/result.md) illustrates a
negative result with compact numerical data; [Csmallbroad continuation](../results/csmallbroad-continuation/result.md)
is inconclusive; [drafter tradeoffs](../results/retained-drafter-tradeoffs/result.md)
retains useful tradeoffs. These are imported historical findings, not new experiments.
