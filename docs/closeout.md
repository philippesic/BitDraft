# Investigation closeout

This is the tracked presentation and retention specification for Binary Eagle. A shared
closeout skill should read this file; the reusable skill itself is deferred until this
trial format has been reviewed. Instructions here grant no new run or deletion authority.

## Knowledge locations

FINDINGS.md is a short question/conclusion/result map. Keep stable finding IDs and anchors.
`docs/findings/` holds readable results. `evidence/` holds immutable execution or provenance
records. Compact plotting data belongs beside its result. Large captures, weights and logs
stay outside Git, at explicit locations recorded behind the result. Current product docs
explain maintained behavior; open tasks contain only unfinished work.

## Result presentation

Use a finding-led title, then a short explanation of what happened and why it matters.
Include a chart when measurements benefit from one, or a small table for functional checks.
State the tested model, target, hardware, data roles, recipe/budget, timing definition and
repetitions as applicable. State limitations next to the claim, including failures and
inconclusive outcomes. Mark the decision: retain, revise, discard or unresolved, explaining
whether it is an existing decision or a proposed disposition awaiting review.

Link supporting measurements and the exact executed source, including dirty changes and
runtime/dependency identities where required. A repository report revision alone is not
proof of the executed source. If that identity cannot be recovered, mark it missing.

Figures name their units, cohort and aggregation. Do not mix clocks or workload eras.
Retain compact numerical data and a regeneration recipe. Display uncertainty only when
supported by source measurements; do not invent error bars from rounded summaries.

## Closeout procedure

Check claims against evidence; distinguish rerun, recomputed, imported and unverified.
Produce the result, update the finding map, and assess whether implementation/tests/tools
still serve the project. Record retained evidence, reproduction limits and any proposed
retirements. Perform only authorized cleanup, validate remaining source and links, then
close the investigation task. A completed run alone is not a closed investigation.

## Retention and correction

Preserve immutable execution records. Result summaries may be corrected with a short dated
explanation; a materially new investigation receives a new result. Link earlier conflicting
results from the current one rather than erasing them from discoverable knowledge.

Before retiring implementation or artifacts, identify the exact paths, ownership, source
snapshot, dependent states and evidence, backup location and verification, and what can be
regenerated. Distinguish regenerating a figure from repeating an experiment. Keep required
inputs, numerical measurements and executed source, or explicitly document reduced
reproducibility when their retirement is authorized. A local commit is not a remote backup.
Do not rewrite Git history during ordinary closeout. No purge is authorized by this trial.

## Trial examples

[F003](findings/f003.md) is a negative result; [F006](findings/f006.md) is inconclusive;
[F002](findings/f002.md) retains useful tradeoffs without declaring a deployment winner.
Other migrated findings retain their prior text and are not fully closed-out result pages.
The preview renders F003 from its compact data; no historical experiment was rerun.
