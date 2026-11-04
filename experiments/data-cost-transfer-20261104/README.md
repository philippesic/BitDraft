# Data breadth and complete decoding costs

**Completed: the broader training bank and cost diagnostics are ready for joint
design. The new data mix did not establish a speed gain, and complete draft cost
reverses the apparent narrow draft-latency advantage.** Scope: fixed Qwen3.5-4B Q4_K_M target,
row-W1A8 draft, RTX5080 and the frozen matched DFlash runtime. No new
architecture/precision/feature recipe or long-QAT allocation was selected.

## Reusable preparation

The admitted bank contains192 additional public-catalog TRAIN groups (64 each
code/reasoning/prose), producing11,391 windows. Historical/current/consumed and
opaque partition exclusions were applied before selection. Whole user prompts
are limited to280 UTF8 bytes and native prompt+80 output tokens to384 positions.
This broadens short-input source coverage; it is not long-context evidence.

Every window and required native embedding materialized successfully. Shared
old/new embeddings match exactly for346 overlapping token IDs. Sharded arrays
and a two-manifest/two-map cache keep the reader bounded; caller-retained views
and OS page cache are outside those cache limits. Captured features remain tied
to the recorded target, feature taps and native history contract. The source
roster and ancestry are reusable when a future design needs new captures.

Apply [the TRAIN exposure index](train-exposure.json) before future held-out
selection. Failed capture attempts remain TRAIN. No untouched final inputs were
opened or evaluated.

The first C: capture hit its1200s limit. Recovery reused authenticated completed
groups, then performed new capture on WSL. Preparation took518.11s for a chunk
retaining18 NTFS source paths versus roughly16s for later WSL chunks. These are
different groups and not a paired storage benchmark. Keep future small-file
capture working sets on WSL; paid source paths and failure evidence are retained.

Two partial-UTF8 token pieces in native debug logs exposed an audit-reader bug.
A process-scoped, lossless reader now preserves exact log bytes with
`surrogateescape`; HTTP JSON remains strict. Both real failed logs reparse,
new captures pass, and targeted CPU tests cover byte preservation and strict JSON.

## Controlled training comparison

Both arms start from the same paid5760 checkpoint and receive1920 additional
updates. The control keeps two original CE windows per update. The broader arm
uses the first original window plus one deterministically sampled new window.
The fixed4 denominator, static reachability masks, old356+added407 replay,
optimizer, binary signs and76,880 trainable row scales stay fixed. All192 new
groups are visited. Equal updates do not imply equal labels or equal seconds:4,599 versus4,688
trainable CE labels, with identical2,064 old+2,106 added replay tokens.

Before continuation, the old48-update control exactly reproduces the prior
parameters, Adam/RNG/exposure and export. Broader continuous48 versus24+24 resumed
also matches exactly. Both real exports pass native loader/output smoke checks.
The bank's extra work costs365.35s of training-loop wall versus238.80s control;
these clocks include materialization and are not pure GPU execution time.

All four broader-versus-control comparisons remain unresolved:

| Additional updates and panel | Broader-data throughput change | 95% interval |
| --- | ---: | ---: |
|480 / DEV12,80-token limit|−0.64%|−3.57% to+2.84%|
|960 / DEV12,80-token limit|−0.19%|−2.02% to+1.47%|
|1920 / DEV12,80-token limit|−2.01%|−5.09% to+0.51%|
|1920 / consumed natural24,256-token limit|−0.32%|−1.61% to+0.80%|

Five interleaved repetitions per panel;20,000 paired prompt-within-domain ×
repetition bootstrap draws. Intervals are exploratory and unadjusted. There is
one training trajectory per arm. Repetitions reduce timing noise; they do not
create new independent prompts. These are previously consumed inputs, not fresh
confirmation or a universal predictor. [Recomputable contrasts](transfer-analysis.json).

On the natural24 panel, current5760 measured165.31 tokens/s, old-data7680
167.40, broader-data7680 166.86, and DFlash250.18. The broader arm needed15,310
rounds versus15,460 control for the same25,815 output tokens, yet did not establish
a full-request speed win. Acceptance alone is insufficient. Do not mix these
rates with short-output DEV rates or treat the small mean ordering as a promotion.
The recorded current candidate remains unchanged.

All15 training/comparison jobs passed. Three main panels cover1,760 exact
responses, including1,560 measured requests, plus separate export qualifications
and smoke checks. Checkpoints are preserved; no permanent pruning is applied.

## Complete decoding cost

Five interleaved instrumented/uninstrumented repetitions preserve every output
and draft/round counter. Measured instrumentation overhead is unresolved around
zero: current−0.59% (95% interval−2.36% to+0.95%), DFlash−0.53% (−2.59% to+1.47%).
These negative means are timing variation, not an instrumentation speedup.

| Generation work per speculative round | Current W1A8 | DFlash |
| --- | ---: | ---: |
|Narrow draft generation|1.534ms|1.766ms|
|Draft catch-up|0.806ms|0.354ms|
|Complete draft, including acceptance bookkeeping|2.340ms|2.120ms|
|Emitted tokens per counted round|1.682|2.466|

Current narrow draft generation costs13.17% less (paired95% interval10.81–15.75%
less), but complete draft work costs10.36% more (7.86–12.61% more). Prefill catch-up
also costs67.44ms/request versus14.70ms on this DEV12 panel. For identical4,685
output tokens the current candidate uses2,785 rounds versus1,900. The practical
design priorities are accepted continuations **and complete draft/catch-up cost**.

These are disjoint outer host spans, not isolated GPU critical paths. Asynchronous
GPU work may be charged at a later synchronization. A fixed-other-cost sensitivity
calculation still leaves current full-request speed at94.5% of DFlash even after
subtracting all measured draft spans. This is a conditional diagnostic, not a hard
ceiling: changing a design can also change round count, target costs and overhead.

Two bounded Nsight traces preserve exact outputs/counters on three preselected
consumed DEV cases each after two warmups. The frozen50k-row W1A8 head rule
identifies27.85ms of head GPU kernels:31.7% of draft kernel duration,2.51% of all
kernel duration. Summed kernel times can overlap and cannot be treated as removable
request time. DFlash head kernels are not separately identified by that W1A8 rule.
Most measured current draft kernels already run in CUDA graphs (13,590/13,860 in
generation;5,100/5,240 in catch-up), so blanket graph enablement is not an untested
obvious fix. The traces expose a basis for targeted work without claiming a new
optimization or selecting a head design.

Clock alignment uses five matching host/NVTX generation markers; margins are under
3.3us. No measured kernels straddle request boundaries.4,819 DFlash kernels
(11.40ms) have boundary-ambiguous enqueue phases and remain explicitly unattributed.
Initial profiler startup attempts exposed limitations in main-thread process discovery,
Nsight's appended preload libraries, and an additional system image; the successful
adapter authenticates the exact command, task environment, process identity and
loaded image pins. Failed attempts are retained as logs/freezes; paid panels were
not repeated. [Cost evidence and assumptions](decoding-costs.json).

## Validation, cost and retention

All2,498 qualification/serving/diagnostic responses match target tokens, text and
stopping; these include repeated prompts and warmups, not2,498 independent examples.
TRAIN capture checks are separate. Five CPU tests pass. The exact-state checks,
per-request raw native audits, traced/untraced counter check and kernel trace
checks passed. No final pool was opened, no architecture/precision was selected,
and no permanent rejection or new generalization predictor is claimed.

3,888 physical updates include48 extra split/resume verification updates;3,840
updates contribute the two comparison trajectories. Their update wall totals
639.33s (10.66min). Nonoverlapping supervised job walls total7,334.20s (122.24min),
including failed attempts, capture/preparation and serving. These clocks overlap;
do not add them or call them pure GPU time. Source work, standalone CPU checks,
analysis and idle gaps are separate. [Cost ledger](../../runs/data-cost-transfer-20261104-01/cost-ledger.json).

451 recorded owned process identities are gone and the GPU compute list is empty.
The task retains about5.30GB of WSL raw data/evidence and0.73GB of C: captures,
checkpoints and exports. Source paths in the worktree remain pinned by paid
provenance, so that worktree is deliberately retained. Three failed startup-only
profiler reports and task-local bytecode were removed; useful full traces and all
paid states remain. No push. [Completion](../../runs/data-cost-transfer-20261104-01/completion.json),
[release/cleanup](../../runs/data-cost-transfer-20261104-01/cleanup-release.json).

See [how to use this in joint design](design-use.md) and the
[versioned adapters](../../scripts/data_cost_transfer/README.md).
