# Remaining precision component campaign

## Execution authority — November 5, 2026

User: “ok go do it”, following the incremental head/embedding/KV/scalar plan and
an estimate of9–10minutes of update time per approximately4,000-update arm.
This authorizes implementation, qualification, pilot, matched recovery training,
component screens, combinations and measured comparisons in
[REMAINING_PRECISION.md](REMAINING_PRECISION.md). Earlier planning-only statements
are superseded. The estimate excludes setup/export/evaluation and assumes the
existing trainable groups; additional head/embedding-master QAT needs a pilot.
Continue the existing research goal. No pushes or target/verifier changes.

## Subject, sequence and completion

Freeze B0, the learned-small2048 compact G4/D8 model with the retained optimized
execution runtime. Start with shared Q6_K, exact private Q6_K and private Q4_K
head arms. Continue private/tied embeddings, drafter K/V ladders and scalar/norm
storage, then useful combinations and a small confirmed frontier.

The first primary recovery arms get the planned cumulative0/256/1024/4096
readouts; do not silently treat1,024updates as completion of the4,096-update
comparison. Calibrate resource cost and total budget before launching the broad
queue. Give viable damaged quantizations matched recovery with an unchanged
continuation, retaining at least two slow-starter opportunities. Additional
format ladders and targeted-QAT rescue use the finite evidence-led plan, not a
Cartesian product. Distinguish frozen-head upstream compensation from actual
head-master QAT. Missing trainable/native support is blocked, not failed quality.

Completion requires real native exports/admission/evaluation, matched recovery
curves and latency/memory/throughput charts covering the four component groups,
plus combined candidates or concrete evidence why an arm cannot be implemented.
A prepared adapter, CLI, passing CPU test or healthy detached launch is progress,
not completion. Do not reject an arm merely from initial acceptance damage.

## Operational boundary and launch observations

One delegated worker owns GPU work; parent remains available for discussion.
Parent at13:07Pacific observed5080 idle1352MiB/0%, pauseclear, no project trainer/
server/supervisor. Fresh owner checks still required. Long jobs use remote_job.py
and unique local tmux/run directories with process birth/boot/source pins and
stop/recovery instructions. No SSH or2080Ti. User pause overrides experiments.

Resource observations: WSL free7,968,272,384bytes; Windows C free11,541,807,104bytes;
WSL totalRAM approximately20GB, available approximately18GB. Native/CPU state and
private vocabulary matrices must be budgeted explicitly. Preserve all earlier
paid checkpoints/archives/raw model dictionary basis and immutable source data.
Before broad training, choose a safe bounded NEW-campaign checkpoint retention
and compression policy, keeping active state recoverable, frontier states resumable,
and useful native exports/curves/source/data ancestry. Do not promise that a
large master-head+optimizer fits or discard valuable states to force the sweep.
If space truly blocks an arm, record the exact prerequisite and continue useful
independent arms. No disk expansion or personal configuration edits implied.

## Reusable evidence

- docs/REMAINING_PRECISION.md: exact ladders, controls, recovery and tying rules.
- experiments/execution-campaign-20261105/{report.md,runtime/optimized.patch}:
  retained exact DP4A/gate-up/tiled-quantizer/output-pruning implementation.
- runs/execution-campaign-20261105-01/runtime-combined-last02 and
  runtime-reproduction01: actual qualified optimized runtime binaries.
- runs/precision-campaign-20261104-01/finalists/G4D8-learned-small/model-final.json:
  frozen candidate and archived native export/checkpoint ancestry.
- scripts/dflash_baseline: current training, native archive and export components.
- experiments/w1a8-standing-20261102/baseline.json: frozen external Q4 comparator.

The operational worker owns subsequent checkpoints here, manifest/report and
STATUS/active-goal updates. Keep large outputs/raw logs out of Git, commit coherent
source/evidence progress, clean safe owned scratch/processes at handback and
preserve healthy active training if the user asks a question during work.

## Implementation owner checkpoint — November5,13:20Pacific

Implementation runs in the supplied T3 worktree, without delegates or pushes.
Fresh RTX5080 check: idle1352MiB/0%, no project process; pause clear.
Canonical artifacts: `/home/philip/binary-eagle-decoding/runs/component-campaign-20261105-01`.
Full B0 archive authenticated against the inference sidecar: exact learned-small2048
parameters/Adam/RNG/order,5,100,543,059raw bytes and4,207,188,154archive bytes.
All prior artifacts remain protected.

The new model-only runtime overlay reuses the immutable optimized CUDA library,
adds private/tied embedding admission and F16 metadata conversion to F32 graph
operands. Q4_K uses the pinned GGML native quantizer from targetQ6_K, notQ4_0.
Preparation completed; native admission and full-model arithmetic remain to test.
Draft-cache source audit finds implicit Hadamard rotations:K128,V64, followed by
32-value blocks, native scale-half rounding and read dequantization. Simulation
implements these semantics, pending native fixture/history qualification.

Live pilot: local tmux socket `binary-eagle-runtime`, session
`component-pilot-20261105-01`; supervisor record and stdout under the CURRENT
WORKTREE `runs/component-pilot-20261105-01/`, not the canonical artifacts directory.
Exact supervisor/group birth identities are in `state.json` and arm ownership.
Stop the authenticated supervisor withSIGTERM, allow90seconds for recovery save,
then verify its group and GPU release; never stop unrelated processes.
Pilot is32updates, NOT the primary4096allocation; full checkpoint and exact
resume/native gates precede the broad queue.

Resource policy: recover B0 into owned RAM-backed scratch, stream NEW full
checkpoints without a raw disk duplicate, keep bounded rolling NEW recovery
states and selected frontier resumability, archive native exports against shared
component bases. Windows campaign checkpoint destination is
`/mnt/c/binary-eagle-component-20261105-01`. Report real device/tensor/cache bytes
separately from archival compression. Training exposure repeats the24original
TRAIN sequences unless compatible broader captures are explicitly admitted;
overlapping windows never count as fresh labels. Budget remains pilot-defined.

## Native qualification and resource pilot —13:34Pacific

Full pilot:32actual updates in4.882seconds, median0.143224seconds/update;
4096updates project to586.65seconds of update time, excluding setup/export/DEV/
checkpoint I/O. Dense consumed-DEV CE6.89435→6.62556; no acceptance claim from CE.
Actual full-head training fits the5080; exact allocator peak is in
`arms/H0-pilot/result.json`. Full F32 parameter/Adam/CUDA-RNG recovery and identical
native re-export passed. CUDA cache Q8_0/Q4_0/Q5_0 fixtures reproduce native
write/read rounding exactly on2432values each and have finite STE gradients;
K128/V64 Hadamard involution and scalar-gradient checks pass.

Native short64-token captureDEV smoke completed for H0/H6/H4/E6/E4/HE4, all six
K/V pairs, four projection-scalar groups, norms and combined scalars. All final
target responses are correct. H0/H6/E6 acceptance counts agree55/263. Q4_K
private head55/263, E458/257, corrected tied HE457/259. These are unwarmed
smokes, not formal latency/quality conclusions. Real cache logs confirm requested
formats, K128/V64 rotations and buffer allocations. Tied allocation initially
followed the CPU input path; corrected runtime03 stores one output-placedCUDA
Q4_K tensor and aliases the embedding to it. Runtime01's unsupported bool-template
link and runtime02's slow tied placement remain superseded diagnostics; qualified
old runtime sources/binaries are unchanged.

NEW native states now use authenticated tensor-slice archives, referencing exact
frozen tensor bytes and storing only new bytes. Full reconstructed file hashes
are verified before retiring raw files. File compression is not model memory.
CPU archive conversion was initially concurrent with a native reader and hit a
file-lifecycle race; final native panel completed correctly, conversion was
rerun serially and passed. All future archive conversions/native consumers serialize.
The storage report is `consolidation.json`; no pre-campaign artifact was touched.

Windows checkpoint conversion reached its512MiB free-space floor and preserved
the original full state. The incomplete NEW partial was authenticated as disposable
and removed, restoring measured Windows free space to6,299,725,824bytes.
A bounded streamed byte-shuffled retry now precedes the H0/H4 recovery stage.
Live supervisor: current worktree `runs/component-head-recovery-20261105-01`,
local tmux socket `binary-eagle-runtime`, matching session. Birth/group identity
is in its state.json. Stop authenticated supervisor SIGTERM with90s grace,
then verify group/GPU release. The stage includes full4096updates,256/1024
readouts, recoverable rolling checkpoints and actual native readouts, not only
the pilot. The budget is `budget.json`: maximum114688provisional updates
(including controls/reserves/combinations/confirmation), approximately4.56hours
pure update time at pilot speed; actual cost is tracked separately.

Remaining: full head recovery, finite Q3 probe/conditional bridge/R2, embeddings,
KV long-context/history qualification and recovery, scalar recovery, interactions,
locked frontier confirmations and charts. Entire component campaign remains OPEN.

## H0 full window —13:53Pacific

The unchanged-precision continuation completed4096cumulative NEW updates, with
0/256/1024/4096native exports and full rolling optimizer/RNG checkpoints.
Consumed-DEV CE6.89435→6.67338→6.82900→7.05519; agreement
11.11%→14.07%→14.81%→13.33%. More training did not monotonically improve DEV;
this control, not only B0, is the comparator for precision recovery. No threshold
shortening occurred. Actual-native short readouts are now running serially after
trainer GPU release, followed by H4's full matched window.

A source-freezing wrapper now copies each future arm's small Python execution
closure before importing training code, runs that immutable copy, and archives
the closure after terminal result. A CPU import gate passed. This prevents
subsequent implementation edits from changing live long-arm imports or pins.
The current H0 process was already loaded and remained healthy/unmodified.

Prepared, not yet native/CUDA-qualified: R2 native vocabulary block-scale rescue.
One existing half scale per256weights is trainable with identical capacity in
privateQ6 and Q4/Q3 controls; codes/subscales stay frozen and native export writes
those actual half scales. This avoids the extra7.63GB gradient/moment allocation
that dense vocabulary-master Adam would add to the10.60GB R1 CUDA peak.
Its independent CPU analytic-gradient fixture passed exactly; full-head CUDA,
format export/resume and native readouts remain required before use.

Tensor-slice archival now also supports lossless XOR payloads against matching
frozen tensor slices. An actual trained native export roundtrip passed with
complete original hash equality. This preserves changed block-scale bytes without
storing another full vocabulary matrix. The fixture was safely removed after
keeping `qualification/xor-roundtrip.json`.

## H4 full window / automatic serial handoff —14:14Pacific

H4 completed4096updates; dense DEV CE7.32178 versus equally trained H0
7.05519, a+0.26658gap. Its readouts are0/256/1024/4096; no early elimination.
Native readouts and final full recovery save are finishing before lane release.
H0 short native counts are55/263,49/273,51/269,55/263 at the four readouts;
these three64-token DEV requests are noisy diagnostics, not convergence or
formal throughput claims. Native tensors changed persistently: H0 has16.12M,
30.10M and55.41Msign-bit differences from B0 at256/1024/4096.

Bounded successor supervisor is waiting for the owned head supervisor's terminal
success and process-group disappearance before taking the GPU lane. Current
worktree `runs/component-main-recovery-20261105-01`, tmux session same name,
socket `binary-eagle-runtime`; exact process/birth/boot records are in its state
and canonical `successor-owner.json`. It performs a real full-checkpoint storage
qualification, then Q6 private-copy controls, finite Q3 probe, short/medium/
near4096cache diagnostics, and all embedding/KV/scalar primary4096windows.
The waiter launches no competing GPU process. On user pause, stop both owned
supervisors as appropriate; successor has180sgrace, head supervisor90s.

Storage pilot now includes an optional lossless tensor-aware delta container
against the immutable paid B0 checkpoint. Its small exact nested-state/memfd
recovery fixture passed, including matrices/scalars, missing-basis tensors,
optimizer state, Torch/NumPy RNG and data cursor/order. The successor will require
actual full CUDA-produced state recovery, complete tensor/metadata equality and
byte-identical native re-export before writing `checkpoint-policy.json` or
retiring the old NEW container. No adoption is inferred from this CPU fixture.

After the first pair, future primary arms keep native0/256/1024/4096readouts,
with full rolling states at1024and4096. The first pair already kept256states.
This adjusts expensive I/O cadence without changing updates, examples, optimizer,
RNG, objective or precision. Detailed DEV/export/archive/checkpoint costs are
recorded separately. Final frontier retention remains two resumable states,
with original paid B0 and all useful native evidence protected.

## Copy controls, context qualification and finite Q3 —14:40Pacific

H0/H4 full4096windows and native0/256/1024/4096smokes completed. H4's final
short native count is49/273 versus H055/263; this is a small consumed diagnostic,
not a formal rejection or irrecoverability result. Native head-memory/latency and
broader warm repeated comparison still decide usefulness and any bounded R2.

Exact private Q6 head/embedding controls reuse H0's identical frozen-Q6 training
forward/order/exposure/optimizer policy, with zero extra updates claimed. Their
private native storage/execution are measured separately at all four readouts.
Q3_K was selected only after H4 recovery, encoded from the same pinned Q6 source,
admitted natively and is completing its full4096R1window.

Short/medium/near-limit synthetic cache panels use128/1536/3840input positions
with256output reserve and ignore_eos in all arms. All6requested K/V types are
actually allocated; source logs confirm rotations and cache formats. These
repeated consumed-DEV contexts are cache stress tests, not new natural heldout.
No new sealed payload has been opened.

Full tensor-delta checkpoint recovery passed all real F32 parameters, Adam, RNG,
cursor/order and native full-file bytes. It measured4,278,862,226bytes versus
4,251,488,319for streamed byte-shuffle: no size win. Future arms retain the simpler
byte-shuffled format; the current exact CPT state stays recoverable until its
next verified replacement. See `checkpoint-policy.json`.

A further concrete cache audit finds native SM120 N3/N16 attention uses MMA_F16,
converting quantized stored K/V to F16 read operands. Simulation now includes that
read rounding. The first quantized-cache trainer must pass the actual CUDA
cache-to-F16 fixture before training. Existing query/softmax surrogate precision
is unchanged from the matched control; actual native requests include tail paths.

Future F16 positive scalar exports capture GPU exp-then-half values directly,
record CPU/GPU half-rounding differences and save the small derived deployment
values alongside full masters/optimizer/RNG. This prevents an export CPU exp
rounding tie from changing the simulated deployed value.

A prepared runtime04 adds real IEEE-F16 clip bits in UINT16 GGUF scalar fields,
converted once into the existing F32 kernel parameter. It has NOT yet passed GPU
admission; runtime03 remains live. Existing Sclip is a rounding diagnostic with
F32 metadata until the physical-storage variant is separately qualified. Clip
storage can save only36payload bytes and never shrinks F32 runtime op_params.
No old qualified runtime/source/chart was modified.

No compatible full-vocabulary higher-precision target initializer was found in
the pinned model directory. EAGLE's BF16 output has only50,000rows versus248,320
required; using it would change vocabulary/capacity and is ineligible. DFlash/
DSpark BF16 drafts contain no head/embedding table. See
`higher-precision-initializer-inventory.json`.


## Primary recovery checkpoint —15:29Pacific

H0, exact private H6/E6, H4, finite H3, E4 and tied HE4 have all four native
0/256/1024/4096readouts. Physical R1 H3/E4/HE4 each completed the full4096
updates. Final consumed dense DEV CE is7.09648/7.07413/7.12048 respectively,
versus equally trained H07.05519. These are selection diagnostics, not final
heldout or throughput findings. Full-vocabulary CPU/CUDA embedding lookup passed
for selected mask/special/common/rare/unseen/high IDs before HE4.

The first cache read gate initially failed because its fixture used generic
GGML CPY, which does not support quantized-to-F16. Native flash attention instead
calls convert.cu directly. The corrected fixture calls that same CUDA conversion;
all2432values for each Q8_0/Q4_0/Q5_0 match simulation exactly, with finite
upstream gradients. No cache training was performed before this gate passed.
The failure/log/source closure remain retained; this was a fixture failure,
not evidence against cache quality or native cache support.

Current sole owner: worktree `runs/component-rest-recovery-20261105-01/state.json`,
matching local tmux session on socket `binary-eagle-runtime`. It serially runs
K8F/KF8/K88/K48/K84/K44 and Sscale/Smidpoint/Sgain/Sclip/Snorm/Sall at4096updates
with native0/256/1024/4096readouts. Exact boot/birth/group identity is in each
arm ownership and supervisor state. Stop: authenticate that identity, SIGTERM the
supervisor, allow180seconds for recovery/group cleanup. User pause wins. The
prior main supervisor ended with fixture exit1 and its group is absent; healthy
completed training and all paid sources were preserved. Reusable main-job CUDA
cache is reused explicitly, avoiding another redundant PTX cache.

Storage consolidation authenticated bytes and terminal ownership before removing
1,547,805,163bytes of duplicate completed-job CUDA cache files. Seven distinct
cache files and the active reusable main cache were retained. Evidence:
`cuda-cache-consolidation.json`. No learned states, prior paid data, host settings
or shared source/basis were removed. WSL free was3.5GiB and WindowsC5.4GiB after
HE4; checkpoint writes remain guarded and old NEW states retire only after a
verified replacement.


## Serialized measurement successor —15:38Pacific

`runs/component-post-primary-20261105-01/state.json` is a passive supervised
successor, matching tmux session/socket. It waits for the rest-recovery supervisor
terminal success and absence of that group before any GPU use. It then qualifies
physical F16 clip metadata against the identical F16-rounded F32 precision
reference (required logits/cache and native counters), runs a one-repetition
natural12selection panel and clean native replay, and audits actual head/embedding
CUDA tensor/data pointers. One-repetition selection is explicitly not final
formal timing. The successor ends after selection; combined recovery, reserved
rescue decisions and final two-warmup/five-repetition comparisons remain required.
On pause, authenticate and stop both active recovery and passive successor
supervisors, allowing180seconds recovery grace. No competing GPU job is launched.

A bounded R2 candidate uses one existing native half block-scale per256vocabulary
weights, with frozen packed codes/subscales and signed block-scale sign. Q6/Q3
can have negative scales; a CPU exact native-grid fixture preserves those signs
and reproduces initial packed bytes/dequantized values exactly. The analytic
backward fixture also passes. These CPU results do not qualify the full-head
CUDA gradient, optimizer/memory, export or resumability; an evidence-led R2 must
pass those actual gates with a matched private-Q6 control before use. Dense
full-vocabulary Adam still does not fit the measured VRAM envelope.


Bounded source-order numeric diagnostic: native FWHT normalizes input before
butterflies while the mathematically identical training surrogate normalizes
output. On16,384seeded F32 values, H64 is exact; H128 maximum difference is
7.15e-7 and analytic gradients are identical. After actual format grouping/read
rounding, Q4_0/Q5_0 readouts are unchanged and Q8_0 differs in27values, max0.001953,
RMS3.25e-5. This is a CPU source-order diagnostic, not a CUDA/full-attention
bit-exactness claim. `qualification/hadamard-normalization-order.json` records
the bound. Quantized cache write and native read conversion are independently
CUDA-qualified; actual native recovery/context/request trajectories remain the
practical deployment gate. Existing common F32 query/softmax surrogate likewise
does not claim native MMA bit-exactness. Any changed decisions/conclusions require
further bounded investigation rather than declaring quantization unsupported.


## First cache window complete —15:52Pacific

K8F (Q8_0K/F16V) completed4096updates and all four native readouts. Native
accepted/proposed is56/261,58/255,53/267,54/265 at0/256/1024/4096, compared with
H055/263,49/273,51/269,55/263. Dense DEV final CE7.18649 versus H07.05519.
These are three64-token consumed diagnostics; formal warm repeated natural
request/latency and long-context comparison remain pending. Value-only KF8 is
now training; the remaining queue and passive successor are healthy.

The successor already loaded its launch-time post_primary module before later
publication-helper edits. It will produce qualification/selection/allocation
summaries as launched, but does not automatically include the later publication
index/write call; the active root must write that index after selection. Runtime
helper scripts invoked later use their individually recorded current source pins.
No source changes affect already frozen active training closures.

Prepared final helpers now include explicit common-parent component composition,
actual exercised Nsight format profiles, locked-before-open heldout24 confirmation,
three separate warm/five-repetition cache context panels and authenticated two-state
frontier retention. They are preparation, not completed evaluations. Two-frontier
retention protects selected NEW full states and skips intermediate checkpoints in
the second same-history regeneration to avoid a third full state in flight.


## Value-cache control and budget —16:07Pacific

KF8(F16K/Q8_0V) also completed4096and all native readouts; K88 now starts the
both-Q8control. No unhealthy process or storage failure occurred. KF8 final dense
DEV CE7.47662 versus H07.05519, with the same18/135target-token agreements. Loss
alone is not native acceptance and is not a permanent recovery rejection.

Budget accounting explicitly distinguishes18initial logical arms from16physical
training arms: exact H6/E6 reuse H0's identical4096history. Their8192saved physical
updates cover matched controls for up to two evidence-led R2rescue opportunities;
the overall114688physical-update scheduling ceiling is unchanged. All initial
qualified variants still receive4096. Native exports/controls/checkpoint saves,
selection/formal panels, two final retained reproductions and reserved interaction
work are counted separately. `budget.json` and `accounting.json` retain evidence.

Readout publication now derives actual unique exposed label positions from
sequence/anchor update logs, including the H0pilot prefix and exactQ6reuse. It
separates that count from the1762-position admitted pool and repeated supervised
positions. Future resumes record authenticated archive/source-arm ancestry; B0
archive authentication is checked against the frozen receipt before hydration.
Repeated identical hashing was removed without changing configuration, optimizer,
examples, RNG or any precision semantics. Active frozen trainers are unchanged.


## Both-Q8 cache completed —16:28Pacific

K88 completed4096updates and all native readouts: 56/261, 51/271, 51/269, 61/249.
Final dense DEV CE7.18675, agreement16/135. K48(Q4_0K/Q8_0V) is now training,
then K84/K44, then six scalar windows. All target responses still match target-only.
The recovery owner/passive successor remain unchanged; there is no second GPU job.

Conditional broader-TRAIN preparation found16already-admitted source clusters,
native tokenized without a context/GPU generation. Planned retained8-tap feature
bytes267,468,800 plus one128MiB full-logit scratch. No new capture or training has
run. If selected for a separately labeled R2recipe, native DFlash8-tap capture
will verify each hard label against full native logits before retiring only NEW
unused logit scratch. Original paid captures stay untouched. Native EOG IDs are
read from the actual producer log, and new supervision stops at first EOG.
The explicit new manifest/order must be identical across candidate and matched
control. It cannot be treated as precision-only attribution against old R1data.


## Graceful-stop correction —16:33Pacific

A concrete supervisor audit found stop_group waited only for its direct queue
leader, then killed remaining descendants immediately if that leader exited.
The180second flag did not guarantee trainer checkpoint grace in that case.
No active trainer was stopped during discovery. New remote_job supervisors now
wait for all live group descendants until the real group deadline; a CPU fixture
proves a child saves after its leader dies. Evidence:
`qualification/graceful-group-stop/result.json`. Pre-fix source is retained under
`supervisor-sources/remote_job-pre-group-grace.py`.

**Current running rest-recovery and passive post-primary supervisors still have
old code loaded.** For a requested pause while training, authenticate the live
worker against the arm ownership boot/start identity and supervisor group, then
SIGTERM that trainer worker first, leaving its queue parent alive while it saves.
Wait for its recovery receipt/terminal result and group cleanup before stopping
any remaining parent/waiter. Do not SIGTERM an old queue supervisor first while
its trainer is saving. If only native evaluation is active, no unsaved optimizer
state is at risk; terminate/reap the owned native cell and stop queued work.
User pause still takes priority. Never report release until owned groups are gone.
Earlier direct-supervisor stop shorthand is superseded by this correction.


## All cache windows completed —17:25Pacific

All6draft-cache primary arms completed4096, with all four actual native readouts:

|Arm|Final consumed DEV CE|Native accepted/proposed at0 ·256 ·1024 ·4096|
|---|---:|---|
|K8F|7.18649|56/261 · 58/255 · 53/267 · 54/265|
|KF8|7.47662|57/259 · 52/267 · 52/267 · 55/262|
|K88|7.18675|56/261 · 51/271 · 51/269 · 61/249|
|K48|7.22955|57/259 · 54/263 · 49/274 · 60/251|
|K84|7.22531|57/259 · 51/269 · 50/273 · 57/258|
|K44|7.14318|56/261 · 54/265 · 53/265 · 56/261|

No new heldout payload has been opened. These are short3-prompt64-token native
diagnostics, not formal comparisons or new training-seed samples. Final log/
response reconciliation validates accepted/proposed and reconstructs exact prefix
counts within its rounding bound. Sscale is starting, followed by the other five
scalar windows. Current GPU owner and passive successor are unchanged; stopping
old supervisors remains worker-first as described above.


## First scalar storage window complete —17:45Pacific

Sscale completed4096updates and native0/256/1024/4096readouts:
57/259,57/257,53/267,61/250. Final dense consumed DEV CE7.37839 versus
H07.05519, with19/135versus18/135target-token agreements. The different dense
loss and short native outcomes reinforce that neither alone proves throughput.
Stored tensor payload is172,236,800bytes, saving251,904bytes versus B0. No new
scale underflows were found; maximum same-master F16 rounding error5.0932e-5.
Actual native conversion cost, buffer allocation and request benefit remain to be
measured. Smidpoint is now training; five scalar windows remain.

The standalone HTML now supports recovery curves and latest-readout latency/
acceptance/model+cache payload scatter views, with separate fields for short DEV,
selection and final data. V8 syntax and DOM-stub mode/filter checks pass after
fixing tooltip newline escaping. This validates interaction code, not browser
appearance or any unmeasured performance. Source/data/runtime hashes and actual
unique/repeated exposure are in the full CSV/JSON. No external JS dependency.


## Midpoint window complete —18:11Pacific

Smidpoint completed4096and native0/256/1024/4096:57/259,55/263,52/267,50/272.
Final consumed DEV CE7.45003; tensor payload172,236,800bytes, saving251,904bytes.
No new midpoint underflow-to-zero was found. Sgain is now past halfway; Sclip,
Snorm and Sall remain after it. Thirteen physical windows are complete.

The native scalar importer audit now records actual F16 metadata source bytes
and consumer types in diagnostic graph callbacks. Its C++ build passes; it has
not run on GPU yet. After clip-storage/runtime04qualification, run
`python -m scripts.component_campaign.scalar_import_audit` in the released GPU
lane. This must verify all Sall half fields against actual exported bytes;
clip fields have a separate required-logit/cache equivalence proof. Audit timing
is excluded. It is preparation, not an admission/completion claim.

Final retention verification supports the selected actual update count and saved
GPU exp-then-half deployment values, authenticates full masters/Adam/RNG and
requires exact recovered native bytes. `verify_frontier_recovery` remains pending.
Optional extensions now preserve4096readouts rather than skipping directly to
an extended endpoint; no extension or new training recipe has run.


## Gain/clip recovery and continuity —18:44Pacific

Sgain and Sclip completed4096updates plus all four native readouts. Sgain native
counts0/256/1024/4096are56/261,52/267,51/271,55/261; final dense DEV CE7.25814.
It saves141,312tensor bytes with zero new underflows. Actual GPU-versus-CPU
exp-then-half rounding differs in1/2/3/6gain values across the four points; the
export and full state preserve the GPU values used by training.

Sclip counts are58/257,54/263,48/277,56/259; final dense DEV CE7.25606.
Its current native file stores the F16-rounded value in F32metadata. Physical
half-bit field admission remains pending under runtime04; no runtime memory
saving is claimed for the clip constant. Snorm is at1024and continues to4096;
Sall is the final primary window. Fifteen physical windows are complete.

Continuation checklist after the old passive successor runs its launched work:

1. Inspect `post-primary-result.json` and selection/native/replay results; fix
   routine qualification failures without replacing old qualified artifacts.
2. Run `scalar_import_audit` in the free GPU lane. The launched successor does
   not include this later-added call. It also does not write the later-added
   publication index: link `selection-primary-summary.json` manually and publish.
3. Obtain a one-repetition all-arm consumed-DEV host trace if needed to fill
   selection host cost; label it diagnostic. Formal finalists still require
   two same-process warmups and five alternating repetitions. Profiling is
   separately flagged and excluded from plain request performance.
4. Record actual evidence-led rescue/bridge/combination decisions. Run common-B0
   matched4096combinations, including actual head/cache/scalar interaction. No
   global nonrecoverability claim from frozen-head R1 or unrun R2. Broader TRAIN
   and native block-scale R2 are prepared options, not completed work.
5. Lock useful configurations, regenerate/pin two exact full frontier states,
   verify full recovery including saved GPU exp-half values, and lock actual
   native weights before opening new heldout. If a same-history reproduction
   differs, investigate/re-measure the actual retained state; never label an
   unmatched optimizer checkpoint as the measured original native snapshot.
6. Complete formal native/clean-GPU/host/context and post-lock heldout24 panels,
   actual format profiles, JSON/CSV/PNG/SVG/standalone HTML, final report/state,
   scoped storage cleanup and verified owned-process/GPU release. No pushes.

Current supervisors remain `component-rest-recovery-20261105-01` and passive
`component-post-primary-20261105-01`, local tmux socket `binary-eagle-runtime`.
They still have pre-fix group-stop code loaded: use the authenticated worker-first
stop procedure above if a user pause arrives. Healthy jobs stay alive through
source edits/context rotation. The broader nine-model goal remains open.

## Primary recovery complete —19:13Pacific

All17 physical4096-update windows and19 logical arms (the two exact private-Q6
controls reuse H0's identical training history) have completed all four native
readouts. Actual physical exposure is69,632updates and1,044,480supervised
positions, repeating the same1,762unique TRAIN label positions per recipe;
aggregate update compute is10,489.14seconds, excluding setup/export/evaluation.
Sall final consumed DEV CE is7.23740; native counts at0/256/1024/4096are
57/259,52/269,47/279,55/261. No fresh heldout has been opened.

The rest supervisor exited normally and released its process group. The existing
post-primary supervisor now owns the GPU, starting physical half-clip admission,
then all-arm natural12 selection, clean GPU replay and physical-tie allocation.
Its launched code predates later publication/scalar-audit additions; run those
manually after it releases. The campaign is still in progress: rescue decisions,
combinations, full-state frontier retention and formal final panels remain.

## Measured selection and next allocation —19:29Pacific

The one-repetition natural12 selection is complete: Q4 is246.91request tok/s,
B0179.06, H0175.76, H4172.69, HE4178.76 and K44174.25. Acceptance is
1631/2183for Q4,1124/3196B0,1098/3246H0,1050/3342H4,1100/3240HE4,
1098/3245K44. These are consumed-DEV selection observations, not formal promotion.
Clean one-repetition GPU replay is1.1762msH0,1.0168H4,1.0057HE4,1.2005H3,
1.2238K44. Scalar row-scale/midpoint storage costs1.2577/1.2703ms; Sall1.2934.
Actual physical-tie audit passes, including54exact Q6-control logit/cache checks.
Actual graph readers consumed all74F16 scalar tensor fields; clip conversion has
its separate81-row byte-equivalence proof. Raw proof/decision packet is
`runs/component-campaign-20261105-01/selection-evidence.json` (canonical root).

Use the two reserved recovery opportunities for head-only and tied Q4 native
block-scale QAT, each with a separately matched private-Q6 control. These learn
only the existing2,483,200half native block scales, preserving packed integer
codes/subscales, native scale signs and tying. Full dense vocabulary Adam does
not fit the GPU resource estimate. The R2 CUDA gradient/Adam/export/resume pilot
must pass and publish actual cost/peak before its4096continuation. This is a
separate trainability recipe, not relabeled R1 compensation or extra architecture.

Three common-B0/original-history R1 combinations follow: Csmall=K44+gain/clip/norm
half storage; Chead=H4+Csmall; Ctied=HE4+Csmall. This actually tests head/cache/
scalar/embedding interactions. Row scales/midpoints stay F32 in useful combinations
because their quarter-MiB saving has measured conversion cost. The all-scalar
isolated recovery remains retained. Q3 head is slower here; private E4 adds memory
without a measured lookup/request advantage. No Q3 embedding/Q5 bridge/Q2 extreme
is currently selected; none is declared irrecoverable.

Then capture16already-admitted TRAIN inputs with the qualified native DFlash taps
and hard labels, and run an explicitly separate matched H0broad/Csmallbroad recipe.
Do not reuse old EAGLE features or modify primary data/history. Current primary
continuation worsens dense DEV loss after its early best point, and1,762unique
labels are a concrete coverage limitation. Keep new heldout sealed until final
recipes/native states are locked. Planned total is28physical4096windows:17done,
fourR2(two variants/two controls),three combinations,two broader-data windows and
two authenticated frontier regenerations. Pilot32updates resume within4096.
This is planning/accounting, not proof that the later stages have completed.

`component-post-primary-20261105-01` finished normally. Current GPU owner is
`component-selection-followup-20261105-01` (tmux same socket; state under workspace
`runs/`, supervisor566178/group566179, actual birth/boot identity in its receipts).
It finishes all-arm one-repetition captureDEV host traces, summary and charts.
Passive successor `component-recovery-combinations-20261105-01` has supervisor
569261/group569262 and waits for both terminal success and prior group release.
It qualifies exact tied Q6, runs the four R2 windows/three combinations, then the
new capture/two broader windows, all serially. Both new supervisors use the fixed
group grace implementation and300seconds; worker-first graceful checkpointing is
still preferred. Fresh authenticate identities before any stop. No other GPU
owner. The old post-primary owner receipt is preserved as
`post-primary-initial-owner.json`; an old loaded wait helper may rewrite its old
owner filename when the followup starts, so use distinct successor owner receipts
and supervisor states as current authority.

### Selection publication repair and R2 start —19:34Pacific

All selection/scalar/host measurements succeeded, but the first followup exited1
while publishing: the archive reference helper expected an archive receipt for
the existing raw external-Q4 file. Fixed raw-file pinning; failed attempt retained.
Its passive recovery successor also exited safely without GPU work. The `-02`
followup reused all measurements, published78actual readouts and exited0.
Current sole GPU group is workspace run
`component-recovery-combinations-20261105-02`, supervisor569889/group569892.
Its exact state/birth/boot/source records are under that run and canonical
`recovery-combinations-owner.json`; the `-01` groups are finished, not live.

Exact tied-Q6 native requests, single-allocation and required-logit/cache checks
passed (`qualification/tied-q6/verified.json`). H4R2's32-update CUDA pilot is now
loading/running; it will save full state, native-evaluate0/32, then resume4096
before the matched control. Do not interrupt healthy training. Both filesystems
remain constrained; .NET drive inventory finds only Windows C (about5.27GBfree).
No host configuration, disk compaction, old paid artifacts or target changes.

## First targeted recovery and actual head-forward gate —20:02Pacific

H4R2 completed4096, with final consumed DEV CE7.39061 and native readouts
55/263,50/271,54/265,53/265at0/256/1024/4096. This is not a dense-loss recovery
success; the equally trained R2 Q6 control has not completed yet. The32-update
pilot actually resumed full masters/Adam/RNG/cursor. Its all2,483,200native
block-scale gradients are finite/nonzero; Adam step32and native frozen-code/
packed-grid checks pass. Pilot compute5.69965seconds, median.171798s/update,
projecting729.56seconds/4096pure updates. Peak Torch allocated11,987,964,416B,
reserved12,108,955,648B, host peak11,872,620,544B. Setup/checkpoint/native costs
are separate in `r2-pilot-budget.json`; do not call this full dense-master QAT.

Before H6R2 training, a bounded actual native full-head audit passed: q4/H0/H4/
H3/HE4/H4R2, three captured DEV domains, two required columns each, all248,320
vocabulary logits. Every36argmax matches the exact packed-weight F32 QAT head;
worst absolute error.0242939, RMS.00429259, target CE delta.00463534. All upstream
gradients are finite/nonzero. This bounds native activation/reduction drift;
it is not full-graph bit equivalence. Native target labels remain immutable.
Proof: `qualification/head-forward-initial/verified.json`, with actual normalized
inputs/layouts, native output hashes and source/model/runtime pins. Only NEW full
output scratch was retired after passing; no model/checkpoint/capture was removed.
The audit is strictly serial between jobs and its timing is excluded. `train.py`
has this explicit preflight before H6R2; it is a no-op once verified. H6R2 is now
loading in the same sole GPU group; HE4R2/HE6R2, three combinations and two broader
windows remain queued at their full4096allocations.

Passive post-recovery supervisor is workspace
`component-post-recovery-20261105-01`, supervisor571807/group571809. It waits for
successful recovery-combinations `-02` terminal state and absent prior group,
then audits actual exports and measures consumed natural12/clean GPU/host traces.
Its loaded code predates the new head-forward helper: **run the default final
`head_forward_qualify` manually after it releases**, to include all four final R2
heads, before final lock/formal panels. It does not select finalists or finish
the campaign. No fresh heldout is opened. WindowsC stillabout4.9GiB free outside
rolling writes; WSLabout2.2GiB. Future source changes must preserve healthy jobs.

## Matched R2 pair and explicit combination ladder —20:32Pacific

H6R2 completed4096and native0/256/1024/4096:55/263,48/275,53/267,52/268.
Its final dense DEV CE7.27905 leaves H4R2's equally trained residual gap0.11156,
versus0.26658under the frozen-head R1 recipe. Short native endpoints53/265Q4
versus52/268Q6are near each other; larger consumed DEV and request cost still
decide usefulness. Do not infer final throughput from this smoke. Both R2 recipes
start B0 with matched examples/order/exposure and native block-scale capacity.
HE4R2 is now training (past1,200), followed by HE6R2. Nineteen physical4096windows
are complete,77,824actual updates and11,843.14seconds of update compute; setup/
checkpoint/native costs remain separate. No new heldout has been opened.

The explicit ladder amendment isolates the requested KV-before-scalars stage:
H4 -> HE4 -> CtieKV(HE4+K44) -> Ctied(HE4+K44+half gain/clip/norm).
The two extra interaction/restoration arms are Csmall(K44+half gain/clip/norm)
and Chead(H4+K44, all scalar storage F32), preserving the no-embedding alternative.
Chead's **unlaunched** prior plan was backed up before this revision; no paid or
active state changed. `selection-ladder-amendment.json` and `queue-policy.json`
pin the rationale/old plan. The future queue expands Ctied's CtieKV prerequisite;
the existing active source closure and optimizer are untouched. Total accounting
is now29physical4096windows (17primary+4R2+4combinations+2broader+2frontier
regenerations),118,784updates. The added existing-parameter window projects586.65
pure compute seconds, with setup/checkpoint/native overhead separate. This is a
necessary ladder isolation, not an externally imposed budget or a shortened arm.

The old passive post-recovery `-01` was intentionally SIGTERMed after matching
its command/cwd/start time/group and verifying it owned no GPU work; its terminal
status is correctly **interrupted**, and the group is absent. Active GPU group
569892 was preserved. Replacement `component-post-recovery-20261105-02` has
supervisor575695/group575700; its initial full state and birth/boot identity are
in `post-recovery-watcher-replacement.json`. It now includes CtieKV selection and
the default final full-head audit before its measurements. The earlier manual
head-audit reminder is superseded if this new default audit passes. Both runtime
owners still use serial release checks; do not start competing GPU work.

Legacy B0 hydration now streams through the same authenticated model+256MiB
headroom policy as new slices, without modifying the old archive/receipt/basis.
A real B0 decode into NEW scratch matched every byte; an existing-partial fixture
proved exclusive-open failure cannot remove another partial. Verified scratch
was released; evidence `qualification/hydration-safety/verified.json`. No host
config/compaction or prior paid deletion. Per-readout table memory now comes from
reaped native cells, with exact model tensor bytes, rounded allocator/cache/
rotation logs, compute history and sampled device peak separate. Full-state
status names the actual retained update, rather than labeling every readout as
resumable. Interim charts currently contain85readouts; final selection remains open.

Existing zero-cache stress offers no strong long-context-specific K44 failure:
short128B0250/1023vsK44256/1011; medium1536B0290/944vs285/954; near3840B0269/984
vs268/986. These synthetic repeated consumed inputs are cache/history stress,
not natural generalization. No additional long TRAIN capture is motivated by
these observations yet; final selected recipes still require separate5-repetition
short/medium/near-limit panels. The planned broader16native TRAIN captures remain
a distinct matched short-context recipe; never claim they validate long training.

## Restart reconciliation —21:48Pacific

Unexpected WSL restart terminated the old recovery and passive post-recovery
supervisors. Their old `running` state receipts remain unchanged as historical
evidence; no process was adopted or signalled. Fresh boot/GPU checks show no
compute owner and5080pause clear;2080Ti pause remains untouched.
`restart-reconciliation.json` authenticates every archived native byte stream
and repack original identity. All21physical4096windows (17primary+4R2) survived.
Csmall had setup/native-zero files but no update log, readout or recovery state;
its entire setup directory is preserved under the timestamped reconciliation.
The new attempt starts from B0 with the original full4096allocation.

Resume supervisor: workspace `runs/component-restart-recovery-20261105-01`, local
tmux socket `binary-eagle-runtime`, session `component-restart-recovery-20261105-01`.
It first completes the authenticated NEW archive repack, then reuses completed
R2 receipts and executes Csmall/Chead/CtieKV/Ctied,16native TRAIN captures and
H0broad/Csmallbroad. No stale predecessor wait is used: its predecessor is the
new successful restart-audit supervisor. Source and birth/boot identities are
recorded in the new run and campaign owner receipts. Stop only the authenticated
new supervisor with SIGTERM (300-second checkpoint grace); preserve other jobs.
Final audits, consumed-DEV selection, retained-state qualification, locked formal
panels, fresh heldout, charts/report and release remain required. This is a
restart checkpoint, not completion. Parent owns integration and delayed message.

Post-recovery successor is now the NEW workspace supervisor
`component-restart-selection-20261105-01`, on the same local tmux socket/session
name. `restart-selection-owner.json` pins its birth/boot and predecessor. It waits
only for the new authenticated recovery supervisor's successful terminal state
and group release, then runs the final full-head audit (all four R2 arms), native
tensor audits and consumed natural12/GPU/host selection. It cannot lock finalists
or open heldout. Both new supervisors must be terminal at completed handoff.

Checkpoint creation/conversion now removes a failed partial only after this
writer successfully opened it exclusively. Three isolated exclusive-open race
fixtures preserve the other writer's sentinel; a real streamed CPU checkpoint
roundtrip remains exact. Proof: `qualification/checkpoint-partial-ownership.json`.
No paid checkpoint was changed. Future frozen trainer closures include the fix.

## Storage reclaimed and serial recovery resumed —23:04Pacific

The parent retired six explicitly authorized older exploratory optimizer states;
the canonical retirement report records26,038,592,683bytes reclaimed. These
states are intentionally retired, not missing active-campaign dependencies.
Both filesystems now have about14GiB free. No host configuration changed.

The affected Windows HE6R2 checkpoint now passes its full archive hash,
decoded raw-stream hash and actual in-RAM masters/Adam/RNG/cursor checks.
Evidence: canonical `qualification/HE6R2-post-reclaim-structure.json` and
`post-reclaim-owner.json`. This restores the read gate, not a claim of actual
GPU continuation or a completed final comparison. The prior I/O failure remains
preserved. All21 completed physical windows are reused; no paid arm is rerun.

Current sole recovery supervisor: workspace
`runs/component-post-reclaim-recovery-20261105-01`, tmux socket
`binary-eagle-runtime`, session matching the run. Canonical
`post-reclaim-launch.json` records exact source/birth/boot identities and the
stop procedure (authenticated supervisor SIGTERM,300seconds checkpoint grace,
then verify group/GPU release). Earlier-boot supervisors are stale, not live
predecessors. New CPU structure supervisor finished successfully and released.

The serial queue is Csmall/Chead/CtieKV/Ctied, each4096updates from B0, then
16native TRAIN captures and matched H0broad/Csmallbroad4096windows. Csmall's
old setup remains preserved under the second-reboot audit; it had no durable
optimizer cursor. Native audits, recovered DEV selection, retained-state actual
GPU resume/export, final lock, formal panels, fresh heldout and full publication
remain required. Parent owns canonical STATUS, integration and delayed message.

### First post-reclaim combination complete —23:24Pacific

Csmall completed the full4096updates,720.85seconds of update compute and
1093.65seconds of actual trainer wall including setup/export/DEV/checkpoint I/O.
Its four short native readouts passed target output semantics and observed
Q4_0/Q4_0 draft caches, with accepted/proposed55/263,51/269,50/271,55/262.
This is consumed captureDEV diagnostic evidence, not final timing or heldout.
The final full state was hash/decoded-stream verified before the rolling1024
predecessor retired. All native exports and source closures remain recoverable.

Chead is now loading/training in the same authenticated recovery supervisor.
Accounting records22completed physical windows and90,112updates; current-arm
in-flight updates are not counted as completed. CtieKV/Ctied, broader capture/
matched windows, final audits/selection, retained-state proof and formal panels
remain. The passive selection supervisor is
`component-post-reclaim-selection-20261105-01`; its birth/boot and release
dependency are in canonical `post-reclaim-selection-launch.json`. It cannot open
heldout or tune finalists. Both supervisors must be terminal before handoff.

### All four combinations and16native TRAIN captures —November6,00:24Pacific

Csmall/Chead/CtieKV/Ctied each completed4096updates and all four actual native
readouts. All requests preserve target tokens/content/stopping semantics under
runtime04 with observed draft cache formats. The endpoint short accepted/proposed
counts are55/262,53/265,61/249,59/254respectively. These consumed captureDEV
diagnostics do not establish final request speed or unseen-prompt quality.
Accounting now records25physical windows,102,400updates and16,099.12seconds
of update compute. The complete trainer walls for these four windows are
1093.65/1079.92/1078.46/1105.12seconds; native readouts are additional.

All16broader native TRAIN captures completed and their full-vocabulary hard
labels passed verification. Manifest publication initially rejected the
unchanged frozen reference because it had only path/hash while `pin` also returns
bytes. The comparison now authenticates every field actually frozen; original
manifest hash is unchanged. No data was changed or recaptured. Failure and
repair proof: `qualification/broader-manifest-reference-repair.json`.

The first post-reclaim recovery/selection supervisors are terminal with exit1,
not successful. Repair supervisor `component-broader-repair-20261106-01`
finished0 and released before the new continuation. Sole recovery supervisor is
now workspace `runs/component-broader-recovery-20261106-01`, with passive
successor `component-recovered-selection-20261106-01`. The same named local
tmux sessions use socket `binary-eagle-runtime`. Exact birth/boot/source and stop
records are in canonical `broader-recovery-launch.json`; authenticate before
SIGTERM and allow300seconds checkpoint grace. No stale predecessor wait or paid
training rerun. H0broad/Csmallbroad each still receive4096updates from B0 under
the separate matched broader manifest. Audits, selection/lock, retained actual
GPU resume/export, formal comparisons, heldout and publication remain required.
