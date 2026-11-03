# Modern packed W1A8 execution qualification

November 3, 2026. Local RTX 5080 / SM120, CUDA13.1.115, F16 target/draft KV,
Qwen3.5-4B Q4_K_M target and correctness mode32. The original serving qualification
uses the zero-update initializer; the follow-up below also qualifies two diagnostic
training updates. No generalization, answer-quality or throughput win is claimed.
The primary comparator remains frozen DFlash Q4_K_M; historical F16 work is separate.

## What is now measured

The retained modern EAGLE checkpoint can be serialized and executed with all nine
linear projections stored as packed one-bit signs and F32 row scales, using
per-token A8 codes and real CUDA DP4A. It borrows the quantized target's native
F32 embeddings. No expanded Q8 weight publication or private embedding is used.

The corrected exporter derives Q/K head counts and rotary dimensions from GGUF
metadata. It permutes only rotary40of160, including scales and affine offsets;
the unrotated tail is preserved. Original metadata-free historical fixtures keep
their full-head Q32/KV8 convention. F16 and BF16 base files are accepted with
logical tensor-shape checks. Twenty-six focused export/provider CPU tests pass,
including independent wrong-full-head rejection. The real initializer's nine
projections /76,880 rows pass the independent source audit. Its GGUF is40,802,176
bytes. The reproducible zero-update NPZ scratch was retired only after that audit;
retained BF16 source, recipe, manifest and output preserve reconstruction.

New fixed-A8 operations pass CPU and real-CUDA reference tests at K2560/5120/7680,
N1/4/7/33, including the50,000-row head, zero vectors, RNE ties, subnormal inputs,
and rejected metadata/layout variants. Codes, scales and final products are
byte-identical to the independent reference on those fixtures. Supported K must
be a positive multiple of32 with bounded integer sums; other modes fail closed.

The development server test completed six arms over three reused prompts, with
128-token limits and two excluded warmups per arm: frozen target-only, candidate
target-only, frozen DFlash Q4_K_M d2, candidate DFlash Q4_K_M d2, candidate ordinary
EAGLE Q4_0 d2, and candidate W1A8 initialization d2. Root independently audited all
18 measured raw responses: token IDs, text, stopping and truncation match both
current frozen target-only and the saved pilot references. All16 distinct loaded
runtime images match the appropriate frozen/candidate pins. All nine packed
projections load, and the drafter occupies28.47MiB of CUDA model buffer. The
untrained W1A8 arm has67 accepted /408 proposed draft tokens on the measured
requests. This is positive native drafting, not a training result or native
actor/parent-prefix qualification. One repetition on three reused prompts is
insufficient for a performance or untouched-quality claim.

## Data and actor gates

The CPU-only admission gate authenticates the baseline, exact target/runtime,
native capture/build/tokenizer, serving request and response, raw arrays and
history. The two known helper/server label disagreements are excluded from
supervision and numeric continuation-protection eligibility. Seventeen CPU tests
pass; real packets admit16/18 teacher rows, and altered eligibility flags cannot
bypass a label mismatch. Actual parent-success protection remains unproven.

The actor shift is explicit: feature[t] + token[t+1] predicts token[t+2] at
decoder/cache position t. Both adjacent teacher rows must pass the numeric gate;
the final teacher row has no following actor label. The real cases admit13/16
such transitions. A CPU zero-update actor with exact native scales/norms and
validated offset-to-absolute vocabulary maps predicts2/13 eligible labels
correctly; unsupported labels0. These are teacher-forced CPU estimates, with a
different denominator from native aggregate acceptance. Current Torch A8 dots
dequantized values before row scaling, while the native operation scales after
the integer dot. Full actor/native decision comparison is still required before
QAT. Do not change historical arithmetic or infer protection sets from aggregates.

## Source and reproducibility

[Packed source patch](packed-w1a8.patch) applies cleanly to the retained
quant4group correctness source commit
`aff0cbd567ef3ab4a3bf2f35c9b9c0e3e2fdc265`. It appends two GGML op IDs, adds fixed
row-A8 CPU/CUDA execution, and adds strict EAGLE packed loading and graph wrappers.
Ordinary paths remain available and were checked under both runtimes above.
The frozen runtime and parent gitlink are unchanged. This candidate backend has
not been broadly qualified on all targets or performance-equivalent workloads.

The source snapshot and build recipes are under
`runs/w1a8-modern-packed-runtime-20261103-01/`. Build03 incrementally fixes a missing
string-array template instantiation after Build02 compiled but failed at final
link. Build01 failed before source validation because it omitted the existing
private CUDA/glibc compatibility include. The retry reuses the exact baseline
include, verified to differ from system math headers only in the two recorded
noexcept declarations. No system headers were edited. The first op wrapper
failed an overstrict unused-library expectation after every numerical case
passed; the corrected wrapper also passes and exits zero. Failed evidence is kept.

## Evidence and next gates

- `runs/w1a8-modern-packed-init-20261103-01/{freeze,export,independent-audit,result}.json`
- `runs/w1a8-modern-admission-20261103-01/{qualification,actor-alignment-qualification,source-recovery}.json`
- `runs/w1a8-modern-actor-readout-20261103-01/{report,independent-readout-audit}.json`
- `runs/w1a8-modern-packed-runtime-20261103-01/{build03-result,op02-result,release-and-storage05}.json`
- `runs/w1a8-modern-native-admission-20261103-01/{freeze,result,root-independent-audit,root-library-audit}.json`

All owned GPU servers, supervisors and builders are terminal and their recorded
processes absent. Useful source/build caches and original checkpoints are kept.
WSL has approximately3.6GiB free and C:3.5GiB. The pending storage request still
applies to the two historical full-state retention arms and future full-sign
training. The next scientific work is bounded native/actor trajectory comparison,
native successful-continuation references, and broader authenticated training
data. Untouched correctness/quality and five repeated full-request timings remain
required before validating an early/long screening predictor or screening backlog.
No push.

## Follow-up: actual serving states change the training-data conclusion

The initial conditional-history replay matches all16teacher and18recurrent CPU
proposals. Exact borrowed embeddings and metadata/shift/map checks pass, with
bounded logit drift but no decision changes. This does not establish that its
captured inputs are identical to the actual speculative server's inputs.

Actual native debug logs now provide response-index/round/depth preservation
masks. Three reused requests retain38/178,20/110and9/120accepted/proposed tokens,
all final responses equal frozen target-only controls. The parser checks raw
token IDs and committed-versus-tentative acceptance, including synthetic tests
for checkpoint replay and a stop inside an emission loop. All10tests pass.
Actual requests used here have no checkpoint restores, so synthetic coverage is
not relabeled an observed rollback experiment.

A bounded diagnostic interposer captures existing EAGLE FC inputs, decoder states,
pre-norm outputs and native candidates. Its first actual run misread backend
top10compact logits as dense; those old logits/proposals are explicitly unusable.
Corrected v2 reads each native candidate ID/count/logit pair. Its actual test
matches110/110debug proposals and preserves the full target response and20/110
acceptance counts. No model operations or verifier are modified; synchronization
and file IO make these traces unsuitable for performance claims.

The source of a concrete input discrepancy is measured. The actual server splits
the55-token prompt into51+4. Conditional capture processes55in one call. The
first51rows are byte-exact, but the last4have8.58% aggregate relative RMS feature
drift. Actual round2proposes567where conditional replay proposes11. Replaying
actual recorded step inputs restores that decision and matches106/110native
proposals. All20actually emitted native successes and all55accepted-prefix
lengths match. All four discrepancies are near ties on rejected drafts, with
native margins0.00878–0.05835; no native success is discarded from evaluation.

An additional pass recomputes CPU FC outputs from the actual raw native features
and carries CPU pre-norm states through recurrent source links. It matches107/110
decisions and retains all20successes, including both depth1successes. This is a
bounded forward check on actual native token/cache paths up to first gold
divergence, not autonomous trajectory parity or autograd equivalence. It supports
proceeding with actual-state training and native decision/protection checks rather
than further floating-point parity investigation. Fresh broader trajectories
remain necessary. The conditional replay's two protected candidates overlap the
actual successful set at only one slot, demonstrating why aggregate conditional
acceptance must not manufacture a preservation mask.

Retention must keep the immutable native success mask but freeze margin targets
from the same CPU parent on those actual states. Native numerical margin targets
would spuriously penalize the unchanged CPU parent on10/20protected rows (maximum
margin difference0.13209). Owned-state CPU margins remain positive on all20slots.

Evidence: `runs/w1a8-modern-serving-trace-20261103-01/` has original three traces;
run02preserves the flawed diagnostic outputs; run03has corrected
`{root-trace-audit,feature-cadence-comparison,protection}.json` under prompt02.
`runs/w1a8-modern-actual-actor-replay-20261103-01/` retains recorded-state and
owned-state reports, independent audits, exact source snapshots and saved arrays.
Root rehashes526initial replay dependencies/outputs and independently recomputes
the110decision join and all20native protections. The actual69token embedding
rows have separate native get_rows ancestry.

## Follow-up: compact training and broader modern pilot

The packed-only actor constructor reproduces all34earlier CPU reference outputs
exactly without loading HF or target weights. Scale-only mode freezes signs and
norms and trains nine scale tensors /76,880values. Compact checkpoint state keeps
immutable initializer/config/source ancestry and exact scale offsets. Native
export changes only the nine scale byte ranges, with correct partial Q/K row
order; zero export equals the initializer file. Eight CPU tests cover metadata,
ancestry, state guards and optimizer/RNG/cursor restoration.

A separate real RTX5080 plumbing test does two teacher-forced CE updates on
seven eligible reused diagnostic rows, then repeats update2after restoration.
Gradients are finite for all nine scale tensors, signs remain frozen, and the
continuation is exactly reproducible. Checkpoint size is approximately0.95MB.
All16updated native teacher proposals match the GPU actor. This is three physical
optimizer updates /two unique trajectory updates, not generalization training,
a validated quick screen or an improved candidate. Successful overall wall is
39.25s; setup/readout/export are included and are not called trainer time.

A frozen balanced24fit/12development pilot was recovered from512authenticated
raw TRAIN groups. Native rendered prompts are36–223tokens, with no identified
exact or normalized overlap against44modern input artifacts. Whole-group roles
remain disjoint for this modern arm, but historical F16 TRAIN exposure prevents
pristine-holdout claims. The original roster/freeze survive; version02pins them
and records added API/window fields. The brief in-place alias extension and exact
restoration are disclosed in its report02.

The conditional teacher collection finishes all36groups in116.22s and retains
868,024,320raw array bytes. Root independently hashes all arrays, recomputes raw
argmax and response-label alignment, and verifies the full forced history:

| Role | Groups | Teacher rows admitted | Shifted steps eligible | Supported eligible labels |
| --- | ---: | ---: | ---: | ---: |
| Fit |24|407/408|382/384|368/382|
| Development |12|204/204|192/192|174/192|

One fit-prose near-tie teacher row is excluded, blocking both adjacent actor
transitions. Unsupported draft-vocabulary labels remain in the serving coverage
denominator. This teacher packet is not relabeled actual speculative history.
Actual parent-state capture/protection across the36groups is a separate next
gate. Untouched independent quality, token correctness and five-repeat native
request timings still have not confirmed an improved modern candidate.

The first collection attempt stops before GPU launch because its conservative
storage floor is short by64.8MB. Exact duplicate task-owned non-index CUDA cache
blobs are then consolidated with every path and all content reverified. Measured
WSL recovery is982,757,376bytes; C:delta0. Unique caches, mutable indexes, paid
checkpoints and original states survive. Full-state training remains subject to
the existing pending storage question; compact scale experiments fit.

Evidence: modern compact/GPU-training qualification runs, native replay native02,
`runs/w1a8-modern-generalization-data-20261103-01/{roster02,freeze02,overlap-audit}.json`,
`runs/w1a8-modern-generalization-capture-20261103-02/{summary,root-independent-audit}.json`,
and the native-replay run's `cache-dedup-{inventory,plan,result}.json`.

## Actual parent capture completed

The separate actual native-parent collection finishes all36groups in183.30s.
Every raw response matches the unchanged frozen target-only reference in full
token IDs, text and stopping. Actual EAGLE input states, FC inputs, pre-norm
outputs, explicit sampled candidate pairs and per-round success masks survive.

| Role | Groups | Returned tokens | Proposed draft tokens | Emitted native draft successes |
| --- | ---: | ---: | ---: | ---: |
| Fit |24|1,810|2,814|356|
| Development |12|937|1,439|193|

Root reruns the parser against every raw response, checks every preserved native
success token at its exact response index, rehashes all19,651trace files /
596,690,320bytes, and verifies all36server groups reaped/absent. Source/protocol
pins and actual library maps are retained. This is preparation for real
actual-state training and preservation evaluation; it does not qualify the CPU
actor on all36groups or establish improved generalization, quality or throughput.

Evidence: `runs/w1a8-modern-actual-parent-capture-20261103-01/capture01/`
`{freeze,summary,root-independent-audit}.json`; each group contains raw response,
trace, protection and release evidence. Incremental cost ledger is
`runs/w1a8-generalization-20261102/cost-ledger06.json`:10terminal supervisors,
two unique diagnostic model updates /three physical updates, with trainer and
whole-job costs separated. Native/CPU state checks are bounded diagnostics;
further parity work must address a concrete loss/decision/protection risk.

Next: build source-indexed actual-state training views without copying full
prefixes; freeze CPU-parent retention margins under the immutable native success
mask; compare matched compact candidates using native full requests and untouched
quality/timing confirmation. Historical full-state retention arms remain intact
and pending sufficient storage. Screening/predictor/backlog and four-target
DFlash confirmation remain later prerequisites. No push or new selected release.

All owned jobs and workers are finished; GPU compute is released. The integrated
source worktree/branch are retired after byte-verified recovery snapshots.
`runs/w1a8-modern-replay-source-20261103-01/{source-recovery,release-and-storage}.json`
retain recovery instructions, terminal identities and storage accounting. A
byte-identical zero-export fixture also shares its immutable initializer inode,
preserving both paths and reclaiming40,808,448allocated bytes. Final free space
is approximately2.19GiB WSL /3.43GiB C:. No paid checkpoint or unique cache was removed.
