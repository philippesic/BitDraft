# DFlash adaptive precision search

## Decision and scope

Agreed November 4, 2026 after root review with an independent Astra xhigh agent.
The user wants several useful size/latency/acceptance tradeoffs, including an
extreme W1A1 option, before progressively adding experimental features. This is
the Stage 1 plan under [CANDIDATE_STAGES.md](CANDIDATE_STAGES.md), governed by
[CANDIDATE_BENCHMARKING.md](CANDIDATE_BENCHMARKING.md). It records a search method,
not measured feasibility, a precision winner, or a new GPU/training allocation.

First qualify the common essential-feature foundation: binary signs, row scales,
row midpoints, shared gate/up input gains, separate down gains, existing norm
gains, healthy updates, export and native execution. Preserve the vanilla control
and its existing 1,800-second allocation. Prototype arithmetic or a reference
packed container alone does not qualify a native candidate.

The initial family/component campaign and its execution optimization are now
complete. The user-selected follow-on plan is
[REMAINING_PRECISION.md](REMAINING_PRECISION.md): head, embeddings, drafter KV
and scalar storage on optimized G4/D8, with matched recovery and combination
confirmation. Its KV/scalar scope extends the initial protected set below;
target KV/precision and attention arithmetic stay fixed. This is a follow-on
plan, not a retroactive change to frozen measurements or a new GPU allocation.

## Four starting families

G is gate/up input activation width; D is down input activation width. All listed
FFN matrices initially have binary weights. Other tensor formats stay fixed.

| Family anchor | Gate/up | Down | Purpose |
| --- | --- | --- | --- |
| G8/D8 | W1A8 | W1A8 | Unchanged foundation continuation/control |
| G4/D8 | W1A4 | W1A8 | Medium precision |
| G2/D8 | W1A2 | W1A8 | Compact precision |
| G1/D8 | W1A1 | W1A8 | Extreme exploratory precision |

These are starting points, not four uniform whole-model formats or presumed
rankings. Down-input search includes A8, A4, A2 **and A1**; reserve an explicit
G1/D1 trial. Intermediate widths are diagnostic choices, not mandatory steps
before testing A1. Even G1/D1 retains declared protected operations and is not a
whole-model one-bit claim. Gate/up initially share width and input quantization;
separate them only for a diagnosed need and charge the extra execution cost.

A1 needs a qualified two-level quantizer: levels, signedness, scale granularity,
threshold, zero representation and zero-input behavior, saturation, gradients,
initialization, export and native implementation. Do not treat it as an A2 flag
change. For every width, midpoint and signed dot product use the SAME gained,
quantized input specified in Stage 0. Learned clipping/step-size qualification
remains part of precision selection; missing support means untested, not inferior.

## Search sequence

1. Freeze one essential-feature foundation checkpoint and its source/data/runtime
   pins. Enumerate exact supported formats and quantizer contracts per component.
   Identify native, reference-only and unsupported configurations separately.
2. Screen all four anchors without adaptation on a small, already-consumed DEV
   cohort. Measure native prefix acceptance and cost where native execution is
   available. Give each qualified anchor matched short adaptation, including an
   unchanged G8/D8 continuation. Poor unadapted A1/A2 acceptance alone is not an
   elimination rule; reserve their bounded recovery opportunity.
3. Within each qualified family, probe output-head weight reduction and down-input
   reduction independently against the same parent. Neither must complete its
   ladder before the other receives adaptation. Prioritize the finite trial queue
   using full-draft cost, recoverable bytes/memory, observed damage and readiness.
4. Expand component probes to fusion weights, attention weights (Q/K separately
   from V/output), and embeddings. List exact available formats before the tranche;
   do not interpret "lower precision" as an unlimited format search. Skip a head
   Q6_K-to-Q4_K step if the starting head is already Q4_K. Historical DSpark head
   results motivate the probe but do not establish modern DFlash behavior.
5. Test useful reductions together. If a reduction fails, use bounded selective
   layer restoration or an interaction/reversal trial rather than restoring every
   layer automatically. Example: G2/D8 plus a Q4_K head may succeed where lowering
   down inputs too fails. Individual successes do not prove their combination.
6. Lock finalists on DEV, confirm the comparisons below, then evaluate held-out
   inputs. Retain a tradeoff frontier and separately select any deployment winner.

Keep residuals, normalization arithmetic, scales/midpoints, softmax/accumulation
and KV at declared protected precision in this tranche. KV/attention-arithmetic
changes, trees/retries, new heads and other topology changes are separate later
decisions. Do not invent DSpark auxiliary heads in the DFlash architecture.

## Finite first tranche and pilot

Before launching the sweep, the pilot protocol must fix standard adaptation
exposure, checkpoint positions, per-run/total resource ceilings, DEV cohorts,
numeric screening envelopes for acceptance/useful progress, and total allocation.
An envelope may differ for compact versus extreme research candidates. Until it
is declared, report acceptance rather than label it "reasonable." Set these
rules before candidate selection, not retroactively to favor a result.

Default administrative bounds agreed in the debate:

- Four matched anchor arms, including the unchanged A8 continuation.
- At most two adapted component mutations per anchor family; reserve one A1-family
  mutation for G1/D1 if correctness, trainability and resources qualify it.
- At most two additional interaction or reversal trials across all families.
- At most two recovery extensions globally, each at most one further standard
  adaptation window. Short initial losses do not prove permanent inferiority.
- One active search branch per family plus two reserve slots globally for useful
  alternatives or unresolved near-ties. Archive other artifacts and results;
  limited active branches do not erase measured frontier points.

These counts bound exploration, not all jobs: every distinct parent/budget may
need an unchanged continuation. Budget those controls, pilot, conversion/export,
evaluation and finalist confirmation before spending the allocation. Reuse a
control only when parent, data order/exposure and training policy match. Cheap
unadapted probes also come from the finite format list and resource budget.
Exhausting a bound ends the tranche with open questions recorded; it does not
authorize another tranche. Counts are practical defaults, not optimality claims.
No precision-sweep allocation is supplied by the old vanilla training budget.

## Training and attribution

Use the same TRAIN examples/order and supervised exposure for matched arms,
recording update counts and compute separately. Save unadapted, intermediate and
final checkpoints; choose the actual short exposure from pilot step cost and
recovery curves. Equal wall time alone is not equal training. Save exact RNG,
optimizer/schedule, data cursor, precision history and lineage; label adaptation
and its optimizer reset policy separately from exact resume.

Initially train only foundation parameter groups plus qualified quantizer
parameters. Keep other master weights frozen. A reduced head/fusion/attention/
embedding must actually use its chosen quantized arithmetic in the adaptation
forward pass. Unlocking other master weights is a separately labeled experiment.

Each exploratory mutation has an unchanged continuation from the same immediate
parent. Final precision-attribution comparisons instantiate locked configurations
from the frozen foundation with matched exposure, data order and training policy.
Equal total tokens do not remove the effect of different intermediate precisions.
If a candidate needs a special curriculum, compare complete training recipes and
name that difference. A previously trained artifact can still be benchmarked if
confirmation is unaffordable, but its advantage cannot be assigned solely to
precision. Reserve confirmation cost before exploration.

## Shared tensors and execution fairness

Target precision and verifier behavior stay immutable. Lowering a shared head or
embedding requires a drafter-only representation, without changing target tensors.
Declare whether draft head/embedding remain tied; do not silently introduce
independently trained copies or extra trainable capacity. Charge private copies,
conversion, metadata and runtime costs. If separation is unsupported, mark the
reduction blocked instead of changing the target.

An A1/A2 configuration executing through INT8 may establish quality and the actual
latency of that implementation. It does not establish a hypothetical packed-kernel
speedup. Keep execution format visible; reference Python timings are not native
deployment timings. Fewer activation bits need not reduce model-file size or
physical buffer size, especially with unchanged W1 weights or expanded execution.

Freeze target/verifier, hardware, prompts, sampling, context, output limits,
proposal depth/policy and runtime within a comparison. If runtime engineering
changes, remeasure all relevant arms including the external comparator under a
common qualified runtime; retain historical results as a distinct series. Do not
attribute a runtime change to precision. First lane is RTX5080, modern Qwen3.5 4B
Q4_K_M target, with the frozen DFlash Q4_K_M settings. Later qualification on
4B/9B × Q4_K_M/NVFP4 uses target-specific models/ancestry and separate charts.

## Measurements, charts and decisions

For each configuration/checkpoint, retain:

- Exact per-component weight format, activation format/quantizer, protected map,
  feature switches, parameter eligibility, runtime/kernel and artifact/data pins.
- Accepted/proposed counts, per-position prefix survival, emitted tokens per
  verification round and verification-round count; do not rely on one percentage.
- Complete draft latency including quantization/packing, gains/midpoint, head,
  required feature catch-up; verification latency where measured; decode and
  full-request throughput. Prefix damage and downstream verification cost matter.
- Actual packed bytes including scales/metadata/private copies, incremental
  drafter memory and total peak runtime memory, with shared storage counted once.
- Training history/exposure, resource cost and acceptance recovery curves.

Produce an acceptance-versus-native-draft-latency chart with size and precision
family indicated, and a separate total-throughput comparison. Distinguish
unadapted/short-trained/final checkpoints, execution formats, target lanes and
controlled versus recipe/artifact comparisons. Missing timings are missing, not
zero. Low-bit anchors are experimental categories, not performance conclusions.

The candidate frontier retains useful storage/memory/latency/acceptance tradeoffs;
it does not require every point to beat DFlash Q4_K_M throughput. A deployment
incumbent requires matched native correctness and useful complete-request
throughput evidence under the existing benchmark protocol. Preserve uncertain
near-ties. Small smokes guide trials, while promotion follows two warmups and at
least five alternating measured repetitions. Timing repeats do not substitute
for prompt diversity or independent training runs.

Select and lock the finalist shortlist and recipes on DEV before opening held-out
results. Tuning on those results consumes that evidence; it is no longer sealed.
The search yields a bounded observed frontier, not a globally optimal allocation
or proof that a short-run ranking predicts long-run convergence.

## Debate resolution

The independent Astra xhigh reviewer agreed conditionally on eight revisions;
root accepts all eight: explicit G1/D1, cost/sensitivity-led head/down order,
bounded exploration including controls, precise ancestry/parameter eligibility,
frontier versus deployment separation, shared-tensor/runtime fairness, locked
held-out selection, and reconciliation of the existing stage/protocol documents.
No material design objection remains. This is design agreement, not validation.

Review: <t3-thread://v2/92296584-2a2c-4998-8f7d-4ecb7d7f5ff2/thread:delegated-task:command%3Amcp%3Af5603e63-bb40-42ba-9075-59c15144f9d8%3Adelegate-task%3Af10adfbb-088b-490e-b088-ee50b0e2d840>
