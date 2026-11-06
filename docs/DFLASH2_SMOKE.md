# DFlash2 smoke comparison

Completed November 6: [results and full qualification](../experiments/dflash2-smoke-20261106/report.md). Both4B DFlash2 cells are measured with a declared F32 down fallback;9B is unsupported without a matched release. Exact candidate continuation and GPU release passed. Parent review/integration and external message delivery remain pending; the implementation worker sent no message. The following records the authorized scope.

User authorized this follow-up on November 5, after the current component
campaign finishes. This is a quick compatibility/performance screen, followed
by one additional 30-minute training continuation of the selected current
candidate. It does not replace or interrupt the active campaign.

Execution order:

1. Finish and review the component precision/recovery campaign and select its
   best performing qualified candidate using that campaign's native results.
2. Qualify genuine DFlash2 checkpoints/runtime and smoke-test the frozen four
   target cells: Qwen3.5 4B Q4_K_M, 4B NVFP4, 9B Q4_K_M and 9B NVFP4 on RTX 5080.
   Use the existing [baseline registry](../experiments/w1a8-standing-20261102/baseline.json).
   Compare DFlash2 against ordinary DFlash Q4_K_M with the target/verifier fixed
   within each cell. Verify the actual DFlash2 selector/model path executes;
   runtime support or a renamed DFlash export is not a DFlash2 test. Establish
   checkpoint/architecture compatibility before downloading/converting/training.
3. Continue the chosen current candidate for **1,800 seconds of actual training
   update wall time**, preserving parameter eligibility, optimizer/schedule,
   RNG, cursor and data ancestry. Save/evaluate before and after; record actual
   updates, unique/repeated exposure, full elapsed time and setup/I/O separately.
   This is one extra training run, not 30 minutes per matrix cell. Benchmark this
   candidate in its compatible target cells alongside the smoke baselines.
   A 4B checkpoint does not establish a 9B candidate result; unsupported cells
   remain explicit. Do not silently change the candidate into DFlash2 or add
   architecture/trainable capacity merely to fill the comparison table.
4. Publish the compact comparison and charts, release owned GPU jobs, then
   send the user's exact queued shader benchmark message. The message must wait
   for this follow-up, not merely the original campaign's completion.

Use existing native runners and consumed DEV prompts; preserve the current
campaign's sealed/final evaluation provenance. Start with the registry's per-cell
depths (2 for 4B Q4_K_M, 3 for the others) if the real DFlash2 contract supports
them; report any necessary model-specific difference. Keep context, output cap,
seed, sampler, target precision and verifier semantics matched. Reuse the
qualified runtime where possible; runtime changes affecting comparators require
fresh matched comparator measurements rather than stale historical timings.

Report accepted/proposed tokens, emitted tokens per round, clean GPU forward
where supported, complete host draft ms/round, full request tokens/s and actual
memory. Agent-proposed smoke budget: two warmups and two alternating measured
repetitions on the same small consumed DEV prompt cohort, with additional repeats
only for obvious instability. Label the result a smoke screen rather than a
deployment promotion or broad quality conclusion. Record invalid, unsupported,
blocked and actually measured outcomes separately. Keep diagnostics outside
timed regions and verify target-output correctness.

No additional feature search, separate 9B training campaign or large depth sweep
is granted by this smoke request. Fix routine implementation/compatibility issues
within scope. If a genuine DFlash2 release or compatible execution path cannot
be qualified, preserve the exact blocker and do not fabricate a result; complete
the useful independent candidate continuation/comparison before handoff.

Parent owns this queued follow-up after the current worker finishes. Preserve one
GPU owner and the current campaign's artifacts. Follow the ordinary retention
procedure at the completed milestone. The exact external message and deduplicated
delivery key remain in
[pending handoff](../runs/component-campaign-20261105-01/pending-thread-message.json).


## Selected continuation after component review

Csmallbroad has the highest measured request throughput among the two retained
binary frontier states on both final consumed DEV and heldout. Use its actual
4,096-update full state for the single extra 1,800-second continuation. HE4
remains the unmodified lower-latency reference. The small throughput difference
is not a statistically established ranking; selection here uses the requested
request-throughput criterion among available retained recipes. H0broad remains
a matched-control result, not a newly retained resumable checkpoint.
