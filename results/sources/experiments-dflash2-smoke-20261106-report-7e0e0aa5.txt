# DFlash2 smoke and Csmallbroad continuation

Completed November 6, 2026 on RTX 5080. Ordinary DFlash Q4_K_M remains fastest in
all four target cells. Genuine DFlash2 is measured on both 4B targets using a
qualified, explicitly declared precision fallback. No matched 9B release was
found in the bounded search. The accepted Csmallbroad continuation completed
1,800.073 actual update seconds; before/after throughput did not improve
consistently. These consumed-DEV smoke results do not establish a statistical
ranking or justify promotion. Parent owns review, main integration and the exact
queued external message. This worker has **not sent that message**.

[Standalone interactive charts/table](comparison.html) · [PNG](comparison.png) ·
[SVG](comparison.svg) · [Diagnostic PNG](diagnostics.png) ·
[Diagnostic SVG](diagnostics.svg) · [JSON](results.json) ·
[Full CSV](comparison.csv) · [Readout trajectory CSV](trajectory.csv) ·
[Continuation/recovery](continuation.json) · [Rejected attempts](attempts.md).

## Measured request throughput

Tokens/s is total output tokens divided by complete request wall time within
each target cohort, including verification and host work. Parent chose retained
Csmallbroad by the prior campaign's slightly higher native request rate over HE4;
that small difference was not an established ranking. HE4 remains unmodified.

| Fixed target | Target only | Ordinary DFlash Q4 | DFlash2 Q4 + F32 down | Csmallbroad before → +30 min | HE4 unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B Q4_K_M, depth 2 | 173.92 | **233.67** | 184.03 | 156.39 → 152.97 | 155.55 |
| 4B NVFP4, depth 3 | 178.18 | **245.77** | 179.64 | 145.69 → 147.93 | 149.12 |
| 9B Q4_K_M, depth 3 | 118.88 | **205.16** | Unsupported: no matched release found | Untested: no 9B weights | Untested: no 9B weights |
| 9B NVFP4, depth 3 | 125.65 | **206.21** | Unsupported: no matched release found | Untested: no 9B weights | Untested: no 9B weights |

The DFlash2 fallback reaches 78.75% and 73.09% of ordinary DFlash throughput on
4B Q4 and NVFP4 respectively. Candidate before/after changes are −2.19% and
+1.54%; mean complete request latency changes 1.2896→1.3184s and
1.2653→1.2461s respectively. No extra matched 30-minute control was allocated,
so these are continuation observations, not an isolated precision benefit.
Intermediate +5/+15-minute native measurements are retained in the charts/CSV.
All distinct target/model/precision cohorts remain separate.

## Protocol and correctness

Immutable registry: [baseline.json](../w1a8-standing-20261102/baseline.json).
One common unchanged component runtime04 serves fresh comparators in every cell:
canonical `runs/component-campaign-20261105-01/runtime04`. Context 4096, output
cap 256, greedy T0/seed1234, concurrency1, cache_prompt=false, flash attention on,
F16 target KV, and `GGML_CUDA_DECODE_CONSISTENT=32`. Per-cell baseline depths are
preserved. NVFP4 files retain MTP metadata, but execution uses the ordinary fixed
verifier; no MTP mode was enabled. No old/new-runtime gain is attributed to models.

Three consumed DEV prompts (code, prose, reasoning), two same-process warmups,
and two alternating measured process repetitions per method. This is a quick
smoke, not the formal panel, fresh/sealed heldout evaluation or a depth search.
Twenty measured model/target combinations yield 120 measured requests and 80
excluded warmups. All 200 outputs exactly match their cell's newly established
target-only greedy token signature. Diagnostic traces match clean-run proposal
counters and are excluded from clean request latency. No 4B checkpoint was
transplanted to 9B. Two repetitions provide no useful confidence interval.

CSV/JSON retain accepted/proposed counts, prefix survival, emitted/round, full
request latency/throughput, complete disjoint host draft spans, GPU event metrics,
and actual memory. GPU timing sums captured CUDA graph regions per invocation:
DFlash2 has two regions and ordinary DFlash one. It excludes CPU/inter-region work
and copies outside captured regions, so it is not complete device/host draft time.
Both graph regions and this scope are explicit in CSV/JSON. Separate GPU replay
uses two repetitions × three prompts × three neighboring contexts per draft;
head/selector audit and tracing are outside clean latency.

Memory records distinguish immutable/shared target storage, private draft model,
KV/recurrent/cache/rotation allocations, CPU RSS/PSS/private/shared residency and
sampled whole-device usage (including desktop/target/workspaces). Native allocator
logs round to 0.01MiB; sampling is not an exact allocator peak. One busy 9B Q4
ordinary-DFlash timed sampler returned no in-window device sample; its resident
memory comes from separately labeled warmed post-request snapshots, with no
replacement throughput. See each row's `residency_sampling_scope`.

## Genuine DFlash2 and the native fallback

The [community Qwen3.5-4B DFlash2 release](https://huggingface.co/nchapman/Qwen3.5-4B-DFlash2/tree/8d8bb62873a4fe6165e644163e9ff76e06ab78ba)
is byte-identical to the [vLLM CI mirror](https://huggingface.co/vllm-project/Qwen3.5-4B-DFlash2/tree/50d7919bb821601f23301bc72d2c34f4046a055d).
It has five layers, dynamic convolutions, a rank256/top-k16 selector, full
248320 vocabulary and target taps2/10/17/24/29. Tokenizer and shared Q6_K
vocabulary bytes match both frozen 4B targets. The metadata adapter preserves the
upstream hidden-state indexing convention; all 81 source tensors are finite.
The checkpoint was trained against a GPTQ 4B verifier; transfer to these two
native fixed verifier quants is an observed compatibility result here.

Original Q4_K_M **and F16** native execution produced nonfinite logits and selector
scores. These runs are invalid and quarantined outside the accepted results.
Numeric audit located the first nonfinite values at final FFN down arithmetic:
large finite GLU input exceeded the legacy half-range input path. The fallback
stores/executes the five original released FFN down matrices as F32; all other
76 tensors are byte-identical to the Q4 conversion. The five F32 tensors exactly
match the released BF16 values converted to F32. No selector weights, learned
values or capacity were invented. Tensor/metadata authentication is in
[the byte audit](../../runs/dflash2-smoke-20261106-01/release-audit/fallback-byte-audit.json).
This is mixed Q4/F32 execution, **not a fully Q4 DFlash2 result**.

Both 4B cells passed actual native selector lattice reads (16 valid IDs and 256
finite scores), all-vocabulary finite-logit audits, top-k inclusion of the unary
argmax, and CUDA F32 down-projection buffer/type checks. Native top-k results are
unordered; qualification checks membership rather than assuming entry0 is argmax.
The selector path's mere presence was never accepted as measurement. Old
`dflash-canonical-dflash-v2` helper revisions are not DFlash2 measurements.

The bounded HF/owner searches and reproducible receipts are under
`release-audit/bounded-9b-search-summary.json`; absence there is the concrete
9B blocker, not a universal assertion that no repository could contain a release.
No new DFlash2 training architecture or vocabulary/target-family change was made.

## Exact continuation, quality and cost

Accepted trajectory restores the original Csmallbroad step4096 retained full
state: all104 masters, full Adam/groups and learning rates, RNG/order/cursor,
40 TRAIN sequences,3234 overlapping windows and3794 unique label positions.
It preserves R1 upstream compensation and frozen head; no objective, capacity,
trainable groups or scheduler reset. G4D8 learned clipped W1 all18 FFN,
A4 gate/up and A8 down, F32 row scale/midpoint, F16-native gain/clip/norm with F32
math, Q4_0 fusion attention, target-shared Q6_K embedding/head and actual
Hadamard K128/V64 Q4_0 cache are unchanged. Binary dot and midpoint use the same
gained quantized input. Csmallbroad remains ordinary binary DFlash.

Zero-update native export is byte-identical to the paid parent. Actual synchronized
forward/backward/optimizer update time is **1800.072727s**, **9998 updates**,
step/cursor4096→14094. One-update budget overshoot is0.072727s (last update0.192332s).
The same data contributes149970 additional supervised positions, no new labels.
Setup/qualification169.660s, final full-state write61.131s and process wall2169.916s
are separate. Post-zero readout/evaluation/export cost81.441s; remaining loop,
logging and state construction overhead is57.611s. Four standalone scratch resume
probes across accepted/rejected attempts cost57.976s of GPU update/probe wall time
and are not counted toward the accepted1800s budget.

Consumed Q4-capture DEV quality (135 label positions): cross entropy
6.9173→7.1406 (worse), top1 agreement12.59%→16.30% (higher). Intermediate readouts
and rank/margin are in trajectory CSV. These proxy metrics disagree and are not
NVFP4-specific quality or fresh heldout evidence. Inherited full HF/native graph
parity remains unresolved; the matched native output/counter checks are distinct.

Two implementation failures consumed **2100.022849 additional actual update
seconds**: attempt01 lost300.008s/1562updates at its first export; attempt02
spent1800.015s/9923updates but was rejected because scratch Adam restoration
aliased a caller CPU scalar step tensor. The corrected loader clones scalar
state, checks the caller's complete optimizer fingerprint, and the accepted run
asserts live Adam/progress equality at every update. These failed/invalid costs
are not hidden inside the successful30-minute trajectory. Attempt02's paid full
state was retained for parent review and then retired in the
[user-authorized storage cleanup](../storage-cleanup-20261106/report.md).
All attempts' source/raw/native evidence remains preserved and labeled invalid.

Final archive authentication, exact native byte recovery and actual GPU next
update14094→14095 all pass; caller optimizer remains unchanged. Final state is
fully resumable with all104 masters/Adam/RNG/cursor, qualified optimizer groups
and original dictionary dependencies. See [final recovery](../../runs/dflash2-smoke-20261106-01/candidate/final-recovery/result.json).
The retained archive and native hashes/bytes are in continuation JSON; no hashes
or process identifiers are needed to read the outcome.

## Reproduction, preservation and handoff

Maintained entry points: `python -m scripts.dflash2_smoke.{acquire_release,prepare_release,train,measure,publish}`;
`verify_range`, `verify_final`, `diagnose`, `export_range` and `freeze_sources`
provide qualification/preservation helpers. Run with canonical `.venv`, from this
source worktree. `measure --skip-recovery --cells 4B-Q4_K_M 4B-NVFP4
--dflash2-model <raw-root>/models/dflash2-4b-Q4-F32down.gguf` reproduces the valid
4B panel; omitting `--cells` covers all four cells. The default DFlash2 model now points
to the qualified fallback; the original Q4 file remains preserved and invalid. Commands,
source pins and exit state are retained in the canonical `remote_job.py`
supervisor receipts: train03, measure02/03/04, memory01/02. **Do not automatically
re-run a GPU command from this report**; first claim the lane and recheck storage.
The frozen helper source bytes for all three training attempts are authenticated
in `source-byte-retention.json`, including one recovered historical native helper.
The successful training source checkpoint is linked by its source-pin receipts.

Raw root: `/home/philip/binary-eagle-decoding/runs/dflash2-smoke-20261106-01`.
New resumable archive:
`/mnt/c/binary-eagle-dflash2-smoke-20261106-01/Csmallbroad-exact-plus1800-14094.pt.bsz`.
The rejected step14019 optimizer archive was intentionally retired after review;
the accepted step14094 archive remains protected. See the November 6 cleanup record.
Original component frontiers, parents, captures, source/runtime04 and old charts
remain intact. Intermediate/invalid native exports use lossless authenticated
archives against retained zero dictionaries; do not remove those dependencies.
[Retention and inventory](../../runs/dflash2-smoke-20261106-01/milestone-retention.json).
One explicitly authorized obsolete one-step diagnostic was retired to reserve
checkpoint headroom: [receipt](diagnostic-retirement.json); its old diagnostic is
now unresumable. Current free space is only about2.7GiB guest/1.1GiB Windows;
no further deletion is implied. Large raw artifacts remain outside Git and have
no independent-backup claim. Authored source and compact results are in this
working branch; p-stack is not their backup.

All12 owned supervisor runs have exited; no owned worker/passive process group
or local job tmux session remains. Fresh GPU audit shows1102MiB desktop usage,
0% utilization and no compute applications. [GPU release receipt](../../runs/dflash2-smoke-20261106-01/gpu-release.json).
Parent must review/integrate this pushed branch and then deliver the pending exact
message once; this worker has not read or sent the other thread's message.
