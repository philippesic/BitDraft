---
name: close-results
description: Close an investigation by verifying durable conclusions, preserving useful evidence, updating current documentation and the findings index, and retiring completed tasks. Use for investigation closeout, not every routine repair.
---

# Close results

Read the project's `AGENTS.md`, `docs/results.md`, relevant active task and existing
`FINDINGS.md` before acting. Project instructions define presentation, artifact locations,
validation commands and authorization. This procedure adds no execution or deletion rights.
If the project specification is missing, establish it within the assigned scope before
claiming closeout; do not invent scientific or retention policy.

1. Decide whether the work produced a durable lesson, meaningful comparison or decision.
   Routine repairs normally update code, tests and current docs without a result page.
2. Verify claims against evidence, including unfavorable outcomes and failed attempts.
   Distinguish new measurements, recomputation, historical imports and unverified claims.
   Locate executed source and dependencies; record gaps rather than treating report
   revision identifiers as execution provenance.
3. Create or update `results/descriptive-result-name/result.md`: conclusion first, then
   Outcome, Scope and limitations, Decision, Evidence and reproduction. Follow the
   project's presentation rules. Add charts, JSON and supporting tables only when useful;
   retain compact figure data and regeneration inputs. Archive reports under
   `results/sources/` only when necessary. Reference required large inputs at their external home. Extract compact measurements
   and figure inputs from outputs while their source is available; preserve all arms,
   unfavorable outcomes and failure denominators. Verify the extracted facts against raw
   records before deciding their disposal.
4. Update current `docs/` for maintained behavior. Assess obsolete implementation, tests,
   tools and artifacts. Before authorized removal, verify ownership, recoverability and
   the evidence needed to support the result. Preserve active/shared work. Give every raw artifact class an explicit retain/discard decision and reason. Retain
   only needed inputs, source, selected/recovery state, reusable references and supporting
   measurements. Record lost exact replay/resume when disposing of weights or captures.
   Use the project retirement entry point and preserve its receipt; do not recursively
   erase an unreviewed tree. When cleanup needs user input, leave that portion pending and
   finish independent closeout work. Unresolved artifact disposition prevents closeout.
5. Update `FINDINGS.md` with a short conclusion and result link. Preserve earlier conflicting
   evidence. Add a dated correction for a small correction; materially new questions or
   evidence receive a new result linked to the earlier work. Never edit raw execution records.
6. Run project validation and check links and figure regeneration where applicable.
   Remove the task only when its completion criteria are met. Otherwise rewrite its state
   and next action, including unresolved evidence or cleanup. Commit and synchronize as
   project instructions require; report local-only work and verification limits.

A finished run alone is not a closed investigation. A discarded implementation may leave
active source while its result and sufficient supporting evidence remain discoverable.
