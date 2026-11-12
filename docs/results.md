# Results and closeout

This file defines BitDraft's result presentation and retention rules. The reusable
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
6. `Artifact disposition`: retained inputs/evidence, discarded outputs, and any lost exact
   replay/resume. This section is required by the repository check.

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
cleanup; a local commit alone is not a remote backup. Artifact deletion still needs an
authorized scope; a retention plan is a record, not permission. Git history changes require a separate explicit instruction.

### Compact evidence is the default

At artifact creation, identify its purpose and disposal point in the task or run recipe.
At closeout, extract the facts the result actually needs: all comparison arms, counts and
failure denominators, per-request or per-group values needed to recompute statistics,
small figure inputs, relevant diagnostics, and exact source/recipe/model/data identities.
Preserve negative and incomplete outcomes. Do not select favorable runs or treat a hash
as a replacement for missing measurements. Verify the summary against the source before
deleting it. Retained measurements must support every reported aggregation and figure.

Decide explicitly for each artifact class: retain a current input, selected/recovery state,
or reusable reference with its dependency closure; or discard it after compact evidence
verification. A historical selected state is not automatically needed forever. If its
weights are discarded, record that exact replay/resume is unavailable. Unknown ownership,
active jobs, unresolved dependencies and unpreserved unique source block that disposal;
record the specific exception rather than quietly keeping an entire run forever.

Keep retained raw inputs outside source. No investigation is closed while its raw-output
disposition is undecided. A successful run alone does not complete closeout. Existing
imported findings describe historical observations; they do not promise all raw bytes.

### Retirement command

Training checkpoint stores use their owner's retention and selected/recovery protections,
described in [execution](execution.md#state-resume-and-branch). The generic command below
is for other closed raw artifacts, including standalone legacy archives; do not bypass
Store retention by unlinking checkpoint payloads through a generic cleanup.

Use `uv run --locked --no-sync --offline python -m tools.retention PLAN RECEIPT`
to verify a proposed cleanup without deleting. The explicit apply form adds
`--apply PLAN_SHA256`. Plans contain schema1, owner, the actual authorization, result path,
limitations, an absolute physical root, protected path roots, authenticated retained files
and an exact authenticated `remove` file list. Retained evidence and the plan/receipt must
live outside that root. Review the exact plan and dependency closure before applying it.

The command verifies preserved evidence, file identities, ownership, writable retired
parents, hardlink closure, protected roots and observed process dependencies. It refuses symlinks and Git metadata,
journals each intent with fsync before removal and resumes only journaled missing files.
Do not use a recursive deletion as a shortcut. Git/worktree retirement is separate because
it must preserve unpublished commits, staged/unstaged edits and submodule objects first.
The tool cannot infer scientific value or grant authorization, detect every remote reader,
or enforce behavior of arbitrary shell commands. Agents must follow this closeout route.

Raw execution records remain immutable while retained. Correct a result explicitly with
a dated note;
a materially changed question or body of evidence gets a new result linked to the earlier
one. FINDINGS points to the current conclusion while conflicting evidence stays discoverable.

## Examples

[Fixed-data continuation](../results/he4-fixed-data-continuation/result.md) illustrates a
negative result with compact numerical data; [Csmallbroad continuation](../results/csmallbroad-continuation/result.md)
is inconclusive; [drafter tradeoffs](../results/retained-drafter-tradeoffs/result.md)
retains useful tradeoffs. These are imported historical findings, not new experiments.
