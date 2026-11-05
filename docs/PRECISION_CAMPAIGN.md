# Precision campaign: execution checkpoint

## Authorization — November 4, 2026, approximately 22:07 local

User: “go ahead and run this until you produce the full comparison chart, should
be quite extensive.” After completing that comparison, if the user has not
returned (roughly 11am Pacific), choose further work within scope: additional
combinations, more overall or targeted training, or new features. Extra work is
conditional on completing the first comparison early. A user return/check-in is
steering, not automatic cancellation; follow explicit pause/stop instructions.

This authorizes executing [PRECISION_SEARCH.md](PRECISION_SEARCH.md), including
necessary pipeline/native implementation, qualification, pilot-selected training
budgets, matched controls, component trials and charts. It supersedes that plan's
pre-execution “no new allocation” status for this campaign. Preserve the original
vanilla allocation and references; do not shorten/relabel their results. Do not
start a new goal. Follow the existing stage tiers for any later feature additions.

## Reconciled starting evidence

Actual local inspection November 4 at 22:07 PST found the baseline supervisor
finished with exit 0. Its result records 14,231 updates, 1,800.021702 measured
update seconds, 213,465 supervised tokens and checkpoints at zero/300/900/1800.
Evidence: runs/dflash-modern-train-20261104-14/{state.json,artifacts/result.json}.
The prior baseline supervising agent is interrupted; no live DFlash trainer,
remote_job or llama-server was observed. Native evaluation is still outstanding.
RTX5080 showed 1,352 MiB used and 0% utilization; that alone is not a GPU lease.
The new sole campaign worker must freshly check pause/process/ownership state
before taking GPU ownership. Do not stop unrelated processes.

Current vanilla manifest/STATUS contain older launch-only fields. Reconcile those
with completed training and actual evaluation evidence during execution; do not
invent native results from training completion.

## Deliverable and execution order

1. Recover the retained prototype work, verify actual CLI/adaptation/export, and
   finish native admission/operations/build for the essential package and A8/A4/
   A2/A1 precision contracts. Reference-only export is not native completion.
2. Export/evaluate preserved vanilla checkpoints and the matched Q4 comparator.
3. Qualify/train the essential foundation, including midpoint/gain removal
   controls. Use a short pilot to fix matched exposure, acceptance envelopes and
   resource budgets before selecting precision candidates.
4. Execute four anchors, down-through-A1, bounded component reductions and
   interaction/recovery trials per PRECISION_SEARCH.md. Keep unsupported, failed,
   unadapted and trained outcomes visible. Prioritize broad actionable coverage.
5. Produce reproducible machine-readable results, full comparison tables,
   acceptance-versus-latency plots, throughput and size/memory plots, recovery
   curves, native correctness and matching evidence. Identify frontier and
   deployment incumbent separately. Render the chart in the parent conversation.
6. Only after the comparison is complete, if the user is still absent, select
   further work by evidence and record the rationale, new bounded budget and
   ancestry. Prefer resolving uncertain promising candidates before speculative
   additions; any features follow the existing tiers and matched controls.

Commit coherent source/report progress, keep large artifacts out of Git, preserve
recoverable checkpoints, clean owned disposable work and processes. No pushes.
Do not report completion from scaffolding, CPU tests, a detached job launch or
acceptance alone. Persist through routine failures. The parent remains available
for discussion; delegated completion wakes it for integration and chart display.

## Owner checkpoint — November 4, 22:24 Pacific

Sole implementation/training/native owner: prepared branch
`t3/precision-frontier-campaign`. All four prototype commits recovered without
duplicating equivalent earlier picks; retained baseline benchmark also recovered.
Fresh local pause control clear; RTX5080 idle before adopted lane. Vanilla
checkpoint evaluation is running, supervised in local tmux
`binary-eagle-runtime` / `precision-vanilla-eval-20261104-01`. Canonical run path
`runs/precision-vanilla-eval-20261104-01`; ownership.json contains boot/process
birth/source pins. Authenticate those before TERM of supervisor; allow120seconds
and verify terminal state/group absence. No other campaign GPU job active.

Completed isolated CPU/CUDA compilation of candidate runtime under
`runs/precision-native-20261104-01`; source/base objects remain untouched.
New loader/graph supports affine FFN gains and midpoints, per-projection
A1/A2/A4/A8 and fixed/learned clipping metadata. Still UNQUALIFIED pending
actual GPU operation and native model checks. A1 uses signed +/-absmax per token
for dynamic mode, threshold0 (nonnegative positive), no zero code; all-zero
rows scale0. Fixed/learned mode uses +/-bound, with zero input positive.
Dynamic input STE is identity; fixed/learned uses clipped input STE plus step
gradient. A2 retains its separate three-level {-scale,0,+scale} grid.

Nine inherited CPU tests pass after A1 contract addition. Runner CUDA support,
baseline-parent conversion, matched cursor continuation and signal checkpoints
implemented but full CLI/GPU verification remains next. Do not mistake this
checkpoint for campaign completion. Vanilla registry now reflects actual
14,231 updates / 1,800.021702 update seconds and pending native evaluation.

## Qualified pipeline and pilot — November 4, 22:33 Pacific

Real CUDA qualification completed:36 operation cases (A1/A2/A4/A8,
dynamic/fixed/learned, aligned edge and actual gate/up/down geometries),12
forward/gradient cases, full-model2→4 exact parameter AND optimizer resume,
explicit A8→A1 adaptation with inherited cursor/reset optimizer, native GGUF
round trips and native target-only output equality. Receipts under
`runs/precision-campaign-20261104-01/`; native qualification panel contains
target/Q4/vanilla-zero/essential4/A1-adapt2, all responses correct. Native
runtime now frozen at `runs/precision-native-20261104-03` for the campaign;
its CUDA-only affine operations reject CPU fallback. Isolated build receipts
pin sources/libraries; no submodule gitlink changed. CUDA supports dynamic,
fixed and learned exported bounds through actual INT8-code DP4A execution.
A1/A2 do not imply packed-activation speedups or smaller physical buffers.

Vanilla2warmup+5alternating repetitions completed,3 previously consumed DEV
prompts. All outputs correct, but native W1 variants remain much slower than
Q4. These are retained under the original runtime as a distinct series; final
controls will be remeasured under the common campaign runtime.128-step essential
pilot showed healthy updates, steady ~0.126s/update after first-use overhead,
but little native acceptance recovery on the narrow pilot cohort. No winner
is selected from these smokes.

Frozen pre-selection exposure policy: essential foundation2048 additional
updates from preserved vanilla1800 (pilot128 included by exact resume).
Matched vanilla/no-midpoint/no-gain controls each get2048 updates from the same
vanilla parent, same TRAIN order/cursor, Adam reset and constant rates. Standard
precision window1024 updates /15360 supervised positions; points0/256/1024.
This is approximately129s warm update compute on the pilot, plus separately
accounted setup/I/O/evaluation. All four anchors, including extreme A1/A2,
receive this window regardless of zero-shot damage. Per-job ceiling3600 update
seconds guards failures; it is not an allocation for preserved vanilla.

Initial quality screens are diagnostic: G8/G4 useful-progress envelope at least
0.10 accepted draft tokens/round and0.05 accepted/proposed; exploratory G2/G1
at least0.02 accepted/round or an observed positive recovery trend. Failing
these labels an arm below this short-run envelope, not permanently inferior.
No elimination before the qualified A1/A2 recovery window. Deployment needs
full matched repeated throughput; smokes cannot promote. Preserve near-ties.

Finite format queue: FFN W1 with G,D in{8,4,2,1}; protected-weight trials
limited to Q8_0/Q4_0 for fusion, attention Q/K, attention V/output, and a
drafter-private Q4_0 head. Target remains immutable. Embedding reduction is
blocked pending qualified private-copy support; no shared target modification.
Head starts at actual target-shared GGUF format (record inventory), so a private
head's bytes and lost sharing must be charged. Gate/up widths/gains stay shared.
No adaptive head/fusion/attention master weights are unlocked. Component training
uses actual block-quantized/dequantized frozen weights, not ignored flags.

Campaign resource reservation: before precision selection reserve four anchors,
up to two component mutations/family,2 global interactions and2 one-window
recovery extensions with matched controls. Reserve four locked finalist
confirmations from the common frozen foundation before exploring mutations.
Standard window1024; confirmation2048 with matched optimizer-reset history.
A bounded first-tranche ceiling60000 physical updates and20000 measured GPU
update seconds includes controls, pilot, confirmations and extensions; it is
a maximum, not an instruction to fill every combination. Storage/actual recovery
will choose the finite adapted queue after zero-shot cost/sensitivity probes.
No feature tiers2+ before the initial comparison is complete.

Active local supervised run: `precision-foundation-20261104-01`, socket
`binary-eagle-runtime`; canonical run state/ownership receipts under runs.
It trains foundation +3 removal controls serially and then a consumed natural12
DEV panel (4 code,4 reasoning,4 prose). This does not consume sealed evidence.
Foundation checkpoint freeze precedes anchor selection. Source changes remain
in the prepared checkout; no other worker or GPU job is authorized concurrently.

Storage: redundant qualification copies were removed only after verifying exact
parameters AND optimizer against retained full four-update recovery. Exports,
logs/configs/proof receipts retained. Actual free-byte accounting is recorded
in `qualification-storage.json`; concurrent pilot output affects observed delta.
Research/vanilla checkpoints untouched. New archive helper verifies SHA256
through lossless compression/decompression before removing an uncompressed
owned checkpoint; archived checkpoints restore through mmap-backed scratch.

### Direct matched precision attribution

The first precision tranche will instantiate all locked arms directly from the
same frozen2048-update essential foundation, with the same1024-update window,
TRAIN order/cursor, optimizer reset, objective and parameter eligibility. Thus
G8/D8 is the immediate unchanged continuation for that common parent; each
family anchor also supplies the matched down/component comparator. This avoids
precision-curriculum differences and the need to retrain selected exploratory
recipes merely to repair attribution. Zero/intermediate/final native points
remain explicit. Administrative bounds remain four anchors/up to two component
mutations per family/two interactions/two recovery extensions. No Cartesian
training sweep. Native repeated finalist confirmation and held-out evaluation
remain reserved; the earlier2048 confirmation allowance is a ceiling, not a
requirement to change an already matched training history.

Full draft costs will use the retained host interposer in disjoint generation,
feature-catchup and acceptance spans. Narrow common-spec generation time is
retained separately. Ordinary throughput is uninstrumented; paired traces must
preserve outputs AND accepted/proposed/round/prefix counts, and report overhead.
No asynchronous host component span is claimed to be an isolated GPU latency.

Checkpoint byte-shuffling plus Zstandard improved a real5.10GB qualification
checkpoint to4.00GB, with exact raw SHA256 reconstruction and successful load.
This retains full optimizer/RNG recovery while limiting the large tranche.
Both old standard.zst and new.bsz archive formats remain supported.

## Frozen tranche and heldout preparation code — November4,22:55Pacific

Source checkpoint a8f8a655 freezes `prepare_precision.py` and `run_tranche.py`.
`runs/precision-campaign-20261104-01/tranche.json` locks28 zero-shot native
configurations before screening:16 G/D ladder points,4 non-down candidate
mutations,8 fixed protected-format probes. Twelve receive matched adaptation:
four anchors; G8/D4 and G8/D8+attention V/output Q4_0; G4/D4 and G4/D8+learned
bound; G2/D2 and G2/D8+fusion Q4_0; G1/D1 and G1/D8+private Q4_0 head. All
start directly from the same frozen essential foundation for1024 updates.
This uses at most two component/quantizer mutations per family and reserves
interactions/recovery for evidence-led allocation. Unsupported embedding/binary
activation-kernel claims remain blocked, not measured as zero.

Actual source inventory reveals the target head is tied to its Q6_K embedding
(no separate output.weight), despite the aggregate target Q4_K_M label. Private
head exports therefore derive from that actual tied Q6_K tensor; target/head
sharing remains unchanged for all ordinary arms. A private head adds its full
Q4_0/Q8_0 bytes and changes only the drafter's output projection. First CPU zero
export attempt stopped at the previously assumed output tensor; fallback to
actual tied embedding fixed, and supervised CPU export resumed existing useful
outputs. No GPU training was interrupted.

Foundation, vanilla-matched and no-midpoint2048-update allocations completed;
no-gain is finishing, followed by natural12 DEV evaluation. Active GPU owner
remains `precision-foundation-20261104-01`. Parallel CPU-only export is owned
by local supervisor `precision-zero-export-20261104-01` with birth/source/stop
receipt. No second GPU job. Zero GPU panel starts only after release and final
operation requalification under the common runtime.

Fresh heldout preparation code is implemented but NOT executed. It requires a
validated locked finalist shortlist, keeps historical512 TRAIN/current60/old
DEV/pilot/matrix/sealed opaque exclusions, adds192-group TRAIN exposure and all
previous natural consumption, then selects24 new singleton natural source
clusters (8/domain). It consumes those inputs before requests and never opens
old sealed prompt payloads. No heldout inputs have been generated or inspected
for this campaign. Candidate selection still uses consumed DEV only.

Storage projected for12 full recoverable adapted states plus final/intermediate
exports. After zero screening, unselected zero-update GGUFs are eligible as
reproducible cache (common parent/config/export ancestry and native evidence
retained); retain four anchor zero files and every trained recovery state.
Preserve raw metadata compatibility information and verify a byte-identical
re-export before dropping any such cache. Do not remove training states to make
room. Windows/WSL free-space readings remain separate.

## All zero arms measured; matched anchors active — November4,23:08Pacific

Four2048-update foundation/control trajectories and natural12 DEV smoke
completed. Native acceptance: foundation756/3931; matched vanilla763/3915;
no-midpoint697/4047; no-gain810/3819. Full-request rates142.95/143.08/139.52/
146.29tok/s versus matched Q4 248.81. All outputs exact. These are one-repeat
DEV diagnostics; no noisy scope change or deployment promotion. Common
foundation remains frozen with the requested full package.

All28 locked zero-update configurations were actually served under the final
common runtime, including A1/A2/A4, learned bound, protected Q8_0/Q4_0 and
private tied-source head variants.180 native responses (plain/traced), all exact
target signatures; accepted/proposed/round/prefix counters invariant under
tracing.36 CUDA operation cases requalified against final common libraries.
Evidence: `runs/precision-campaign-20261104-01/{final-runtime-ops.jsonl,zero-panel/result.json}`.

On capture3 smoke, Q4 accepts341/522 (~65.3%); G8/D8 accepts122/957 (~12.7%);
G4/D8 ~4.74%,G2/D8 ~4.94%,G1/D8 ~1.81%,G1/D1 zero. Complete traced draft
cost ~2.77–2.82ms/round for these binary-FFN arms versus Q4~2.14ms. No low-bit
speedup in current INT8-code DP4A implementation. This is zero-swap damage,
not a trained precision conclusion; A1/A2 bounded recovery remains mandatory.

Active GPU owner: local supervised `precision-anchors-20261104-01`, socket
`binary-eagle-runtime`; state/ownership in canonical runs. It trains all four
anchors for1024 matched updates, retaining256/final native points and a full
lossless final optimizer/RNG checkpoint, then measures zero/256/1024 on the
same natural12 cohort with paired plain/traced requests. Source/config pins
and stop identities are in ownership.json. Authenticate before TERM, allow
120seconds for a boundary save, verify terminal/group absence. No competing GPU
job. Zero export/screen and foundation supervisors exited0 and released.

Raw-response auditor and standalone chart generator now exist. Current chart
artifacts remain checkpoint-only until repeated DEV confirmation, frozen
shortlist/heldout and full audit finish. Do not hand back the current partial
chart as campaign completion. Preserved vanilla original-runtime summary is
committed separately under experiments/precision-campaign-20261104.

## Recovery and measurement corrections — November4,23:35Pacific

G8/D8 natural12 recovery:756/3931 accepted/proposed at0;796/3849 at256;
760/3923 at1024. G4/D8:352/4734→452/4533→412/4617. G2/D8:
387/4666→331/4775→341/4756. Small one-repeat consumed DEV observations,
not long-run precision rankings; unchanged A8 also shows no sustained gain.
G1/D8 is now training its full1024 updates, preserving extreme-family exposure.
No extension, interaction, finalist selection or heldout has occurred yet.

Zero protected-format probes: Q4_0 fusion/attention reduce actual GGUF bytes
with little additional zero-shot acceptance damage on capture3. A private head
adds340+MiB or more and slows full draft execution on this implementation.
Mutation adaptation order therefore starts attention V/output, an independent
head probe, learned clipping and fusion before the down-input queue. All remain
common-parent/matched-history; scheduling does not change exposure.

Auditing recovered vanilla benchmark revealed two excluded repetitions used
fresh servers, so process-local graphs were not warm for measured cells.
Preserve those historical measured values as diagnostic, with that limitation.
Formal campaign confirmation now uses two excluded warmup REQUESTS inside
each same-process server cell and five alternating measured repetitions.
Zero/recovery smokes used no warmups and remain labeled diagnostics.

Review also found learned gate/up clipping must share a Parameter just like
their input gains; otherwise adaptation could produce incompatible bounds for
the native shared pack. This is corrected before any learned-bound full-model
adaptation. A fixture verifies shared gradient/update/checkpoint identity;
export resolves the shared clip parameter. Dynamic/fixed reference histories
and existing native zero artifacts are unchanged. Native learned adaptation
remains a required measured mutation, not inferred from the fixture.

All four pilot states were retained via lossless byte-shuffled archives, not
dropped. A byte-identical re-export of a zero GGUF passed before16 unselected
zero models were removed as reproducible cache; configs, exact writer metadata
compatibility, parent, hashes and native results remain. Twelve adapted-arm
zero models remain available for same-cohort recovery. Storage supervisor
`precision-storage-20261104-01` exited0; receipts under campaign root. CPU
maintenance overlapped early cells of the G4 recovery smoke; that panel has an
explicit resource note, and formal timing must run without maintenance.

The retained decoded-token correctness audit now reopens raw responses and
checks full output/stop signatures, accepted/proposed/prefix counts, complete
cell coverage, matched exposure and unchanged trace counters. Checkpoint audit
passed all completed panels. Heldout preparation remains unexecuted.

## Runtime gained-stream correction — November4,23:50Pacific

All four anchors finished1024 updates and their natural12 zero/256/1024 panels.
G1/D8 recovered96/5244→482/4474→597/4244 accepted/proposed, warranting continued
attention; A2 did not recover in its current absmax/ternary recipe. No candidate
has been promoted. All full final optimizer/RNG states are archived losslessly.

A stronger actual gain+pack test found the generic CUDA gain op flushes tiny
signed inputs under inherited fast-math. The prior bare-pack36 cases passed
but this gained-stream test failed; preserve that failed receipt under
`generic-gain-failure.json`. Runtime03 is now superseded for campaign claims.
This is a concrete arithmetic risk, not a reason to chase unrelated HF parity.

Runtime04 fuses positive gains into the custom pack kernel using explicit
RNE F32 multiplication with FTZ disabled, before both quantization and input
sum. Target/verifier objects and arithmetic remain unchanged. All36 gained
CUDA cases now pass byte-exactly, including zero/subnormal inputs, A1/A2/A4/A8,
dynamic/fixed/learned bounds, row midpoints and actual gate/up/down shapes.
Build receipts and source patch live under runs/precision-native-20261104-04
and experiments/precision-campaign-20261104/runtime. Do not pool runtime03
and04 latency effects as precision effects.

Active GPU owner: `precision-runtime-requal-20261104-01`. It recreated deleted
zero caches with their exact writer compatibility and verified all restored
hashes; re-served28 zero configurations under04; now remeasures all zero/256/
1024 anchors plus removal controls and external Q4 on natural12. Metadata,
ownership/birth/source/stop receipts remain in the canonical run. Final promotion
will remeasure every relevant control under04 with proper per-process warmups.

To preserve space, two terminal control archives were copied to the existing
project Windows artifact store, hash-verified, and canonical paths atomically
replaced by verified symlinks. No checkpoint bytes lost; source/parent paths
still resolve. Separate WSL/Windows free-byte measurements are recorded in
control-archive-relocation.json. No VHDX compaction or Windows reclaim claim.

Future jobs now execute from small immutable owned SOURCE snapshots (not new
Git/device checkouts) in the campaign artifact root, preserving exact source
through unrelated edits. Weights/data/runtime stay canonical. The launcher
records original commit/diff, snapshot hashes, actual supervisor source, boot
identity and process birth, and children run from their frozen source root.

TRAIN-only input-range calibration support and per-projection learned/fixed
initial bounds are prepared for a possible reserved interaction, not executed
or selected yet. Gate/up bounds remain tied. Bounds pin their TRAIN calibration
ancestry; frozen observation verifies no parameter mutation. No new Stage2+
features or heldout generation have begun. After runtime requalification,
continue the8 locked component/down mutations before choosing interactions,
extensions, repeated confirmation and heldout. Full chart remains unfinished.

## Mutation checkpoint — November5,01:45Pacific

Runtime04 requalification finished:28 zero-arm panel180 responses and
anchor/control panel408 responses have exact full target output/stop signatures
and unchanged plain/traced acceptance/prefix counters. Runtime03 remains a
separate diagnostic series. Current sole GPU owner is
`runs/precision-mutations-20261105-02`; ownership.json authenticates source
snapshot, process birth/boot, group and TERM/grace120s stop procedure. Preserve
this healthy serial job. First mutation launcher failed before training because
its frozen snapshot omitted a top-level exporter dependency; corrected complete
snapshot imports pass. No paid updates were lost or silently restarted.

Six mutations have completed training; G2D2 native recovery panel is underway,
then explicit G1D1 remains. All use the frozen common foundation, same1024
updates/15360 supervised positions/order/cursor and reset Adam. Attention V/output
Q4 reduces actual file size to354.6MiB but does not beat external Q4 throughput.
Private Q4 head costs803.5MiB total and hurts latency; target tied Q6_K embedding
is unchanged. G4/D8 learned clipping reaches1070/3301 acceptance and about165.8
full-request tokens/s on consumed natural12, versus substantially poorer dynamic
G4. This does not isolate precision from clipping policy or prove learned bounds
beat fixed bounds. Diagnostic timing is not final confirmation.

The remaining initial allocation is frozen in reserved-allocation.json before
execution: two common-parent1024-update interactions, learned G8/D8 control and
learned G4/D8 with Q4_0 fusion+attentionQK+VO; two exact one-window1024-update
continuations, G1/D8 and unchanged G8/D8. Reserve decisions respond to clipping
confounding, potential size frontier and G1's positive recovery slope. They use
all initial interaction/recovery slots, not extra per-family mutations. Total
extra4096 physical updates; paired control is charged. Calibrated A1/A2 and
fixed-vs-learned ablations remain untested variants, eligible only after full
initial chart if user is absent. No feature tier2+ or heldout has begun.

After standard mutations finish: safely release only reproducible zero caches,
relocate retained owned pilot archives if needed with verified canonical aliases
and both-filesystem free-space receipts; run reserved allocation; freeze search;
repeat DEV with two same-process excluded warmups and five alternating measured
repetitions; lock max4 finalists before fresh24-case heldout; audit and produce
full charts. Proper-warmup GPU protocol and protected-component independent
code audits remain required. Full campaign is unfinished; continue in this thread.

## Reserved trials and storage — November5,02:05Pacific

All12 standard arms and their natural12 native panels completed. Reserved GPU
owner is precision-reserved-20261105-01, immutable snapshot/ownership/stop
receipts in its canonical run. Same-process two-warmup native smoke passed.
Learned A8 reaches1075/3292 and166.97request tokens/s at1024, nearly identical
to learned A4; dynamic-vs-learned range policy is a major confound in attributing
the earlier G4 improvement. Combined small G4 interaction and paired dynamic
G1/A8 continuations remain underway. These are still diagnostic panels.

Storage supervisor precision-reserved-storage-20261105-01 exited0. Eight
reproducible zero caches released; three paid pilot checkpoints relocated
losslessly to the existing Windows artifact store with verified canonical
symlinks. Receipts zero-cache-release-final.json and reserved-storage-relocation.json
record exact paths/hashes and both filesystem free bytes. Checkpoints unchanged.

The pre-selection record also reserved four2048-update finalist confirmations.
Honor that exposure: exact1024→2048 continuations have the same history as
uninterrupted common-parent2048 training under the constant schedule. Reuse
matching completed continuations, charge all additional confirmation updates,
lock their configs before confirmation and distinguish mixed recipes from pure
precision effects. Final four-family dynamic1024 attribution stays matched.

To make space without dropping any paid bytes, lossless dictionary archival of
GGUF exports is CPU-fixture qualified. A frozen full native GGUF is the pinned
recovery dictionary; full raw SHA256 is verified before removing a redundant
uncompressed copy and again on hydration. Training checkpoint archives untouched.
Native execution always receives the exact original GGUF, with hydration outside
request timing. Actual full-model archival/admission qualification remains pending
after the reserved GPU job. Sixteen CPU correctness tests pass; receipt saved.

## Search complete; finalist confirmation — November5,02:43Pacific

All12 standard arms, both reserved interactions and both matched recovery
continuations completed native zero/intermediate/final panels. G1's extension
plateaus597/4244→611/4221; unchanged G8 similarly760/3923→770/3899. Learned
A8/A4 recipes are near-tied at1024, while the compact learned G4 interaction
uses174.94MiB and reaches1052/3338 acceptance/~173.6request tokens/s in the
same diagnostic panel where Q4 reaches~256. No throughput winner or W1
deployment promotion. These are short-run single-seed recipe findings.

Four finalist configurations are frozen in finalist-training-plan.json:
G8D8-learned-clip, G4D8-learned-small, G2D8 and G1D8. Each gets common-parent
2048-update exposure with the same TRAIN order/cursor and initial reset policy.
Reuse exact completed G1 continuation; charge three additional1024-update
confirmation windows. Different clipping/protected formats are recipes, not
sole-precision attribution; the four dynamic1024 anchors provide that matched
comparison. Holdout payloads remain unopened.

Lossless native archival completed under precision-archive-20261105-01, exit0:
77 exact GGUF byte streams (34.90GiB raw) retained in4.17GiB containers plus
the fixed dictionary. Twenty protected-component exports/169 tensor code checks
match authenticated frozen sources, including private heads. Historical deleted
zero caches re-created to their exact old hashes before archival. Actual archived
GGUF hydration/admission passed40 native responses with same-process warmups,
exact target signatures and unchanged trace counters. Model-size charts use RAW
GGUF bytes, never these compressed sizes. Paid optimizer checkpoints unchanged.
WSL free increased10,000,355,328→30,655,393,792 bytes during this pass; Windows
free10,958,622,720→10,954,080,256 (concurrent activity). No host reclaim claim.
Receipts: native-export-archives.json, component-code-audit.json and
archive-native-qualification.

Active sole GPU owner is precision-finalists-20261105-01, executing its immutable
source snapshot. Its actual state/ownership records boot, birth, source pins and
TERM/grace120s recovery procedure. Learned A8 has completed2048 exposure; compact
learned A4 and dynamic A2 are the remaining serial confirmation windows. Then lock the
complete repeated DEV roster (including256 points and recovery states), two
excluded requests inside each server and five alternating measured repetitions;
lock four final model pins before selecting fresh24 heldout inputs; audit and
produce complete charts. No feature-tier2+ work before that artifact exists.

Training exposure audit: the captured TRAIN corpus has1426 eligible anchor
windows across24 sequences. A standard1024-update window covers1750 unique
native label positions;15360 supervised positions count repeated overlapping
window exposure, not unique source tokens. The audit now itemizes updates,
windows, unique positions and captures. This limits generalization claims and
helps distinguish useful future data/range-policy work from more unchanged
training. Fresh heldout still unopened; no followup training allocated early.

Checkpoint charts now build181 measured panel/arm points from native receipts
and retain explicit incomplete status. This verifies plotting/table generation,
not campaign completion. Old panels without request-boundary timestamps retain
memory peaks but have no invented request-time clock/power summaries. Formal
confirmation samples actual clock/power/temperature alongside memory.

## Repeated confirmation active

All four predeclared recipes completed matched2048-update exposure. Final
pre-confirmation audit passes four finalist histories and all completed raw
native panels:27,658 campaign updates /3,634.355100807792 measured update
seconds /414,870 supervised positions, including qualification; frozen vanilla
14,231/1800.021702/213,465 is separate. Receipt pre-confirmation-audit.json.

Active sole GPU owner: runs/precision-confirmation-20261105-01. It executes
from its immutable source snapshot under binary-eagle-runtime tmux; actual
state/ownership.json records boot/birth/source/group and TERM/grace120s stop
procedure. Preserve it across context rotation. No tests/builds or other GPU
work during this timing block. Its53-arm roster is frozen in confirmation-arms.json;
result output is confirmation-dev12. Two excluded requests inside every server
cell, five alternating repetitions, paired plain/traced execution. Early rolling
monitor:534 target-identical responses,266 available trace pairs with equal
accepted/proposed/round/prefix counters. This is progress, not final results.

Fresh heldout remains unopened. Pending source edits prepare native vocabulary-only
context admission: validate on all12 already-consumed DEV geometries after the
formal timing block, then freeze native4096-context/256-output eligibility before
ranked24-case selection. Whole singleton groups remain intact, no prompt
truncation or post-selection replacement. These helper edits are not built or
qualified yet and must not run during the quiet window; prepare_holdout fails
closed without their qualification. Pending files: native_tokenize.py, tokenize.cpp,
prepare_holdout.py, optional eligibility in modern_natural_confirmation.py and
its synthetic-only fixture. Test/build only after the formal job exits0.

Next: finish all53×5 paired cells; build/qualify native tokenizer and synthetic
selection fixture; freeze four final model pins before any fresh source payload
selection; run24 heldout prompts with the same2warmup/5alternating protocol;
final raw/source/exposure audit; complete JSON/CSV/readable tables, static plots
and standalone interactive HTML. Do not write the completion marker before
those measured endpoints exist. Only then allocate optional followup if user
is absent. Existing checkpoint charts remain explicitly incomplete.

Pending post-confirmation code is prepared but unexecuted: qualify_slots.py
will repeat gained-stream cases at the actual16-slot width, linked against
unchanged runtime04 (five-minute diagnostic bound). Current N4 cases and full
native model panels already pass; this adds direct shape coverage. Then
native_tokenize.py prepare builds vocabulary-only admission and checks all12
consumed DEV counts against measured HTTP; run the synthetic natural-selection
fixture, including default-callback equivalence and whole-component exclusion.
lock_heldout.py freezes the four predeclared2048 model pins and native context
policy before source-pool access. post_confirmation.py serializes these checks,
selection and the24-case2warmup/5paired heldout panel. These sources remain
unqualified working edits until the quiet block ends; do not execute early.

If optional followup is later allocated, add this campaign's own newly consumed
heldout identities/groups/topics to the next selection's opaque exclusions.
The current helper covers all historical/global192TRAIN/consumed cohorts; the
first new campaign heldout does not exist yet. Never reuse it as fresh data.

Rolling formal audit found a live-log sampling race in repetition2, learned A8
2048 traced: reasoning03 sampled the preceding cumulative line; prose00 then
included both requests. All output signatures remain exact. The final raw log
contains14 sequential cumulative boundaries and14 separate per-request prefix
lines; corresponding HTTP responses independently report accepted/proposed
counts. Do not interrupt or restart the healthy full timing job. Its unchanged
source will fail its final pair check after gathering all530 cells. Pending
native_counters.py / repair_native_counters.py will reconcile only from complete,
reaped logs, require one-to-one begin counters, validate every response count and
integer prefix reconstruction, preserve original progress/raw files, and save a
correction receipt plus complete result. Unit fixtures cover delayed sampling,
missing boundaries, nonsequential boundaries and response disagreement; run after
the quiet block. Future heldout runner reconciles every final cell before
acceptance. No request timings, output tokens, models or runtime change.

November5,06:32Pacific: fifth/final DEV repetition has started (428/530 cells).
All independently checked HTTP accepted/proposed pairs and output signatures
remain equal; actual response cache_n is0 throughout. Pending after quiet:
syntax + natural-selection unittest + new native-counter adversarial fixtures;
run repair_native_counters.py confirmation-dev12 --expected-cells530 only after
all530 cells/reaped servers and expected original final pair-check exit1. It
preserves original progress and writes recovery receipt/result; post_confirmation
requires that exact failure/recovery path. Future benchmark reconciles every
reaped log before accepting a cell. Native checker/tokenizer/new selection still
unexecuted. Full pipeline completion remains pending; do not end from this state.
Chart output now carries model/panel/source provenance and corrected memory
accounting. Host-span-subtraction calculations are explicitly neither a
zero-cost-drafter prediction nor a throughput upper bound due asynchronous work.

November5,07:30Pacific: all530 DEV cells completed (7420 responses,6360 measured), expected final live-log pair-check exit1. Qualified final-log reconciliation changed exactly2 rows; all final pairs/counters agree and original progress/raw/timings remain untouched. Four parser fixtures +13 natural-selection fixtures pass; changed Python syntax/diff clean. Charts-preheld-20261105 builds234 native points,53 formal DEV arms; all5 static plots inspected and standalone controller compiles in V8. Status remains explicitly incomplete. post-dev-validation.json records this checkpoint. Native16-slot/tokenizer qualifications and freshheldout remain; next supervised post_confirmation.py performs those and the lock before any fresh payload. No new training allocation.

## Locked fresh heldout active — November5,07:31Pacific

Sole GPU owner is runs/precision-heldout-20261105-03, immutable snapshot/source245f8c10, supervised local binary-eagle-runtime tmux. ownership.json records boot/process birth/source/group and TERM/grace120s stop procedure; preserve healthy work across context rotation. Output heldout-panel:6 arms (target, Q4, four predeclared2048 recipes),24 fresh singleton source components (8/domain),2 excluded same-process warmups,5 alternating paired repetitions. Native target plain/traced first cells completed. Do not test/build or run another GPU job during this timing window.

Additional actual native qualification passed36 gained-stream cases atN16 with byte-exact codes/scales/products and negative admission checks, linked unchanged runtime04. Vocabulary-only native helper passed all12 consumed DEV token-count comparisons. Four model pins/context policy locked before new source selection.7807 eligible singleton geometries checked; selected24 prompts span32–323 native tokens, no truncation/replacement. Consumption recorded before first inference; no training/tuning after opening. Receipts: slot16-qualification/result.json, tokenizer/qualification.json, heldout-selection.json, heldout24/{selection-freeze,eligibility-geometry,consumption,audit}.

Two earlier post wrappers failed before GPU work/fresh payload:01 lacked snapshot-root import setup;02 compiled the checker but its receipt pointed to a nonexistent overlay public header. Fixed import resolution and pinned actual shared public header; failed snapshots/metadata retained.03 successfully passed both native qualifications and opened heldout only after lock. No runtime/model/measurement policy changed; no DEV repeat required.

Next: finish all60 heldout cells; independent component-code audit of compact G4 1536/2048 exports; final raw/exposure/source audit; publish_precision.py data; chart_precision.py to new charts-initial output; inspect PNGs and controller; publish_precision.py publish only after all artifacts exist. Initial full comparison still incomplete. Only after publication consider bounded further work if user remains absent.

## Initial comparison published; conditional control allocated

Full initial publication completed November5 at16:24:27UTC (08:24Pacific).
Receipt campaign-completion.json pins16 actual artifacts under charts-initial,
including240-point JSON/CSV/full table, domain CSV, standalone interactiveHTML,
five PNG/SVG figures and the exact generator source. All5 final figures inspected;
controller compiled in V8; parent owns browser display. Final audit authenticates26
raw native panels, all training ancestry/exposure and22 component-code exports.
Formal DEV53 + heldout6 arms yield8,980 target-identical responses; no paired
counter changes. Original reference and fixed artifact hashes preserved.

Fresh heldout: Q4=241.98 request/281.97 decode tok/s; compact learned G4=163.13/
180.54, acceptance10765/35935 (29.96%), request/Q4=.6742 [95% .6567–.6915].
Target-only174.69. Compact drafter174.94 versusQ4's363.78MiB; combined deployment
bytes6.16% lower, sampled device6031 versus6209MiB. This is a size frontier,
not a speed promotion. Full G8 learned=156.24 request tok/s; dynamic G1=127.68,
G2=115.42. Small TRAIN exposure and unresolved full-model float parity remain
explicit limitations; logical A1/A2 still execute INT8-code DP4A.

User remains absent, so the conditional followup is now used: exactly one
fixed-bound compact G4/D8 control, bound1 throughout, versus the existing learned
compact G4/D8 initialized at1. Common frozen foundation/data/order, reset Adam,
1024 updates/15360 supervised positions; protected formats, loss, target and
runtime unchanged. This isolates learning the bounds from their initialization
for this recipe only. No training extension or new feature tier. Plan is frozen
at clip-control-followup/plan.json only AFTER initial publication. Zero native
admission must match learned-unit initialization, then actual training, full
streamed-state recovery/native-export equality, intermediate DEV smoke, final
4-arm2warmup/5alternating paired DEV and a NEW24-case heldout. Initial heldout
identities/groups/topics are frozen exclusions; cohorts stay separate.

A streamed Zstd checkpoint option avoids raw-plus-archive disk duplication;
2 exact optimizer/RNG resume fixtures and2 existing pipeline fixtures pass.
No existing paid checkpoint moved/deleted; no additional archive moved onto Windows. Full trained
checkpoint decoding/export qualification uses an owned RAM temporary, admitted
against actual RAM/tmpfs headroom; it must release before native timing. Preserve
an8GiB WSL reserve. Current code/plan preparation is complete; launch one supervised
precision-clip-control-20261105-01 lane after fresh hardware/pause/process check,
then keep ownership/checkpoints current through native results and followup charts.

## Followup GPU owner — November5,08:57Pacific

Sole owner precision-clip-control-20261105-01 is live from immutable source6221205b
under local binary-eagle-runtime tmux. Canonical run ownership.json records
boot/birth/source/group and TERM/grace120s stop procedure; preserve healthy job.
Output clip-control-followup. Zero native admission completed40 responses and
fixed/learned unit initialization has exactly equal accepted/proposed/round/prefix
counters. Actual matched fixed training is active (64/1024 at this checkpoint).
No other GPU job. Driver waits for child recovery on signal and stops before a
new stage. Streamed Zstd removes raw/archive duplication; small exact Adam/RNG
resume2tests, existing pipeline2tests and natural-selection13tests pass.

Next within the same serial job: finish1024; full CPU checkpoint optimizer/RNG
coverage/finite checks plus byte-identical export against resident1024 native
point, using owned RAM temp; archive exports; frozen-component audit; intermediate
DEV smoke; proper2warmup/5alternating4-arm DEV; two model pins locked before new
source access, initial24 consumed group/topic/hash exclusions; new24 heldout with
cohort label clip-heldout24. Native completion is not followup chart completion.
After that audit new panels/exposure, build a NEW chart directory and retain all
initial publication hashes. No additional training extension is allocated.

Followup continuation: fixed training completed1024 updates/146.605614 measured
seconds, cursor17303 and15360 supervised positions, matching learned1024 history.
First recovery gate incorrectly counted18 frozen clip constants as optimizer
parameters; no training failure or re-run. Corrected guard validated against actual
Autograd requires_grad flags (3 streamed-state fixtures pass). Replacement owner
precision-clip-control-20261105-02 reuses the completed state, restored all92
trainable Adam states/CUDA RNG, and exported byte-identical GGUF to the resident
1024 point. RAM temp directory released; checkpoint-recovery.json and
ram-recovery-storage.json retain full proof. Keep this healthy sole serial owner
through native intermediate/DEV/new-heldout measurements; no extra updates.

Storage-steward pass removed28 completed redundant packed.pt reference containers,
keeping qual-four's reference example, every paid recovery checkpoint and native
export. Receipt packed-reference-cache-cleanup.json records ownership/no live file
consumers/recovery sources and hashes. Logical removal1,527,527,166bytes; measured
WSL free9,193,504,768→10,721,099,776 (+1,527,595,008), Windows10,931,777,536 unchanged.
No Windows/VHDX reclaim claim. New streamed checkpoint remains4,700,294,541bytes;
no archive relocation or prior checkpoint mutation. Initial charts remain frozen.

Followup formal DEV is active under precision-clip-control-20261105-02; third
repetition complete (24/40 cells). Intermediate DEV smoke completed; all observed
outputs exact. No tests/builds or other GPU work until the full serial DEV/heldout
window ends. Pending unqualified source-only publication helpers:
publish_clip_control.py audits only new raw panels, exact1024 example order,
full recovery, native bound metadata, and all frozen initial artifact hashes;
chart_precision.py adds separate clip-heldout24 facets and paired-effect receipts.
After native-measurements.json exists and owner exits0, syntax/import-check these,
run data audit, build charts-clip-control, inspect all figures/controller, then
publish followup marker. Do not rerun initial publish/data after additional
training or overwrite charts-initial. Final report and storage/process handback
remain. The original initial full chart is already complete and frozen.

Followup DEV completed40 cells/560 responses, all exact and paired counters equal.
Both predeclared1024 model pins locked before new source payload selection.
Fresh24-case roster and source-disjointness pass; heldout-panel is now active
under the same sole precision-clip-control-20261105-02 owner. Its frozen cohort
label is clip-heldout24, distinct from initial heldout24. Prior initial identities,
groups and topics were excluded before whole-component selection; no replacement
or truncation. Continue all40 new heldout cells, then raw audit/new charts. Initial
publication remains byte-frozen; no more training is allocated.

## Campaign handback complete — 2026-11-05T17:53:26.151138+00:00

Initial plus bounded control publication is complete. clip-control-followup/
completion.json pins16 final artifacts under charts-clip-control:258 panel/arm
records, with67 formal confirmed rows and10,580 exact formal responses (9240
measured after warmups). All30 raw panel audits pass. Initial16 artifact hashes
are unchanged. Five combined PNGs inspected, repeated-panel table/bar labels
clarified, controller compiles in V8; parent owns actual browser preview/display.
No browser interaction claim. Reports: experiments/precision-campaign-20261104/
report.md and clip-control-report.md; compact JSON summaries accompany them.

Matched control: DEV fixed/learned request ratio .99993 [95% .98166–1.01978];
new clip-heldout24 ratio1.03268 [1.01798–1.04842], acceptance difference+.02623
[+.01449–+.04009]. Fixed167.514 versus learned162.213 request tok/s; Q4245.606,
target172.792. Both W1 formats174.94MiB, sampled peak6027MiB/increment1394.
Fixed bounds all1; learned native bounds.995856–1.009629. This is one matched
1024-update history, not independent-training variance or a general bound-policy
claim. No candidate beats Q4; retain the tradeoff frontier without deployment
promotion. Different initial/new heldout cohorts remain separate.

All owned recorded process identities and groups are absent; GPU observation
RTX5080/driver617.42/1352MiB/0%. final-storage-process-inventory.json accounts for
18 source-snapshot jobs,13 additional terminal helper jobs,1074 recorded process
identities, retained checkpoints/native archives/basis/source and cleanup.
29 checkpoint archive paths retain118,317,833,959 logical bytes (including
Windows-backed aliases), with the frozen vanilla states separately protected.
28 redundant completed reference containers removed; qual-four example retained.
Final scratch removal42,529bytes; paid checkpoint/native/capture/source data are
unchanged. Exact WSL/Windows before/after values are linked in the inventory and
packed-reference-cache-cleanup.json; no host reclaim/VHDX compaction claim.

No live GPU/trainer/endpoint owner remains. No additional allocation or push.
Parent should review/integrate the coherent tested branch and display the
standalone HTML. Keep the current worktree/branch until integration; broader
active goal remains open for future user direction. Do not restart experiments
from historical launch metadata. This handback contains actual charts/results,
not a scaffold, CPU-only endpoint or job launch.


## Parent review and integration — November 5, 2026

Reviewed the native benchmark/counter recovery, quantizer, training/resume and
GGUF export code plus both experiment reports. Verified sizes and SHA256 hashes
for every artifact in both completion receipts. Independently passed 16 function
fixtures and 20 unittest cases with PYTHONPATH=.:src:tests; no new GPU experiments
were needed. The first local test invocation lacked tests on PYTHONPATH; corrected
invocation passed. Runtime patch whitespace diagnostics refer to retained patch
context lines; the reproduction patch is preserved verbatim. No submodule gitlink
change. Integrated the coherent campaign branch into main without touching
unrelated dirty files.

Previewed the interactive chart successfully with no console errors. The inline
version uses theme styling, short panel labels, a held-out default and metrics
first in its table; the frozen original artifacts and their hashes remain intact.
All original/follow-up cohorts and recipe-versus-precision labels are preserved.
Actual chart display follows this checkpoint in the parent conversation.

Fresh local observation confirms RTX5080 at 1,352 MiB / 0% and no native server
or trainer jobs. Retained evidence and paid recovery archives remain protected;
final-storage-process-inventory.json supplies the bounded retention inventory and
measured WSL/Windows cleanup deltas. No additional training allocation or push
was made in the review. This campaign is complete, including the early-finish
clipping follow-up; the broader research goal remains open.
