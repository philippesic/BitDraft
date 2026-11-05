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
