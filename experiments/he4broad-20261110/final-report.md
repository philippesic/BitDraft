# HE4broad: completed eight-update-hour run

**Complete and validated:** 220,095 updates, 28,800.105 synchronized update
seconds. The 0.105-second overshoot is within the last complete update
(0.183 seconds). There were no discarded/rejected updates or uncheckpointed
completed updates at closeout. The existing 47.9 minutes before localization
count toward this total; there is no additional eight-hour allocation.

## Native results and decision boundary

| Checkpoint/panel | Draft acceptance | HE4broad tokens/s | External DFlash Q4 | Target only |
| --- | ---: | ---: | ---: | ---: |
| Best observed, seven update hours | 48.8487% | 222.065 | 248.677 | 184.351 |
| Final, eight update hours | 46.9355% | 199.505 | 228.353 | 174.403 |

These are output tokens divided by complete-request wall time, including prompt
processing, on three consumed DEV prompts. Five measured process repetitions
per prompt/arm, two warmups, 256-token reserve; fixed Qwen3.5-4B Q4_K_M target,
external DFlash Q4 comparator, depth two and F16 KV. All eight requested new
hourly panels, all six historical formal panels and the extra qualification
smoke are preserved. **All 1,059 requests are correct**, including warmups;
all 213 server processes were reaped. No heldout prompts were opened.

The final point is weaker. Code loses five accepted draft tokens per three-prompt
pass, prose loses one, reasoning is unchanged: 297→291 accepted tokens, with
304→310 rounds, across the same 605 output tokens. This quality regression is
real on these fixed prompts; it does not establish broad overfitting.

The final candidate's raw throughput falls 10.16%, while unchanged Q4 falls
8.17% and target-only 5.40%. Thus the full raw decrease cannot be assigned to
changed candidate weights alone. Request-window GPU samples show stable median
SM clocks around 2,985 MHz and lower temperatures, not evidence of sustained
thermal clock reduction. Utilization is lower for all arms. The physical cause
of the shared timing slowdown remains unresolved without a controlled rerun.
All repetitions and the original final result remain included; no opportunistic
rerun or imputation was used.

The seven-hour state is the best **observed** DEV3 point, still about 10.7% below
its Q4 control. The final state is about 12.6% below its contemporaneous Q4
control. Both remain full-state resumable and retained. External Q4 remains the
deployment incumbent; this run demonstrates no promotion or architectural ceiling.
A broader already-consumed development comparison with TRAIN exclusion checks
would be needed before another training allocation or model/objective decision.
No further GPU work is assigned by this completed campaign.

## State, budget and artifact verification

The actual pinned workflow restored the final full checkpoint on the GPU and
verified exact parameters, all Adam moments/group settings/steps and RNG. Its
native re-export matched the retained final GGUF bytes. The duplicate recovery
GGUF was then removed intentionally; the canonical final export remains.

An independent CPU read verified final cursor/step count 220,095, the exact
1,568,435-window seed-8101 permutation, finite parameters/Adam fields, unchanged
constant rates/config identity and the original cumulative clocks. The complete
220,095-update log follows that permutation without repeated windows (14.03% of
the window pool); overlapping windows are not independent examples. All fourteen
formal panels' target signatures, native counters, requests and source/runtime
identities pass the independent metadata audit: 1,050 formal responses across
210 cells, plus nine smoke requests across three cells.

The snapshot inventory retains **16 full states and 16 native exports**, current
teacher captures, all three frozen source closures, recipe/index/initialization
and ownership/recovery evidence, raw native requests/results and required
base/target/comparator/dictionary parents. Checkpoint/native SHA values were
verified by the final GPU recovery verifier; file identities/sizes were rechecked
at closeout. Local models/data are outside Git. The earlier remote backup covers
only the pre-localization portion; no independent backup of the final/new states
or the teacher tree is claimed by this closeout.

All owned workflow/trainer/server processes have exited with successful receipts.
The owned tmux session is gone. Actual GPU compute-client and utilization checks,
plus acquire/release of all three GPU locks, confirm release. Teacher/archive
names are plain local ext4 directories; there are no run SSHFS mounts to unmount.
Recovery/export/native-cache scratch is gone. Useful source, evidence, compiled
cache and retained model states remain. Monitoring is closed after final
publication/push, using the exact stored event receipt.

## Time accounting

Training iteration wall totaled **49,625.032 seconds** (13 h 47 m), including
**20,817.785 seconds** of input materialization. This includes the superseded
remote-data phase; the remaining local phase completed from the same saved state.
Only synchronized update time consumes the eight-hour allocation. Setup/restore,
checkpoints, export, native serving and final verification remain outside that
clock. Separate train-worker and supervised elapsed scopes are in the machine
summary; they must not be added indiscriminately to overlapping iteration clocks.
The separately authorized matched loader smoke used scratch states and did not
replace the production state or its accounting.

The earlier [storage repair](storage-repair-20261111.json) reclaimed 4.47 GiB
without restarting training or deleting model states. At final release there was
approximately 52.6 GiB free locally. No campaign state was retired at closeout.

[Complete throughput curve](throughput-progress.html) ·
[Seven-to-eight-hour comparison](final-7-to-8-comparison.html) ·
[Machine summary](final-summary.json) · [Final validation](final-audit.json) ·
[Independent integrity audit](final-integrity-audit.json) ·
[Retention and exact identities](final-retention.json).
