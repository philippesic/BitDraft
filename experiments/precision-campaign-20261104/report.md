# DFlash precision frontier on RTX 5080

Retention update, November 6: superseded optimizer states from this completed
search were intentionally retired in the [authorized storage cleanup](../storage-cleanup-20261106/report.md).
Measurements, source/configuration, native exports and their dictionary bases
remain. B0 and the essential foundation are retained; old checkpoint references
in historical receipts do not imply those exploratory states remain resumable.

The initial campaign and one bounded clipping-control follow-up are complete:
**258 measured panel/arm records**, with separate held-out cohorts. Open the
[combined interactive chart](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-clip-control/comparison.html),
[combined table](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-clip-control/full-table.md), or
[matched clipping-control report](clip-control-report.md). The original initial
240-point artifact package remains frozen; its results are summarized below.

The initial comparison is complete. DFlash Q4_K_M retains the throughput
incumbent: **242.0 full-request tok/s** on fresh held-out inputs. The fastest
W1 recipe, compact G4/D8 with learned ranges and Q4_0 fusion/attention weights,
reaches **163.1 tok/s**, or **0.674× Q4** [95% paired bootstrap: 0.657–0.691].
Target-only reaches 174.7 tok/s. The DEV near-tie with target-only did not persist
on this fresh cohort; no W1 deployment promotion is supported.

The compact drafter is 174.94 MiB versus 363.78 MiB for Q4 (51.9% fewer drafter
bytes). Charging the unchanged target once, total deployment bytes decrease
by 6.16%. Sampled total device peak is 6031 versus 6209 MiB; incremental peak above
target-only is 1396 versus 1574 MiB. This is a size/memory trade-off with lower
throughput, not a faster replacement. Dynamic A1/A2 recovery remains weak under
the tested absmax policy; this does not establish their general inferiority.

Artifacts: [interactive HTML](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-initial/comparison.html),
[full 240-point table](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-initial/full-table.md), [JSON](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-initial/comparison.json),
[CSV](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-initial/comparison.csv), [domain CSV](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-initial/domains.csv).
Exportable PNG/SVG pairs cover acceptance/latency, throughput, size/memory,
recovery and the 16-point zero grid. The publication receipt is
[campaign-completion.json](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/campaign-completion.json).

There are 240 measured panel/arm points, not 240 unique configurations. Formal
confirmation contains 53 DEV arms and 6 held-out arms: 8,980 actual responses,
including 7,800 measured responses after excluded warmups, all target-identical.
All native paired counters and prefix counts pass the raw audit. The initial
charts are frozen; any later follow-up uses a separate artifact directory.

## Fresh held-out comparison

Each row uses 5 alternating paired repetitions across 24 fresh source components.
Rates below use plain execution; draft time is a diagnostic traced host span.

| Arm | Accepted/proposed | Acceptance | Full draft ms/round | Request tok/s | Decode tok/s | Request/Q4 [95%] | Drafter MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| target | — | — | — | 174.69 | 189.65 | 0.722 [0.686,0.762] | 0.00 |
| q4 | 17105/23300 | 73.41% | 2.158 | 241.98 | 281.97 | 1.000 [1.000,1.000] | 363.78 |
| G8D8-learned-clip-2048 | 10845/35760 | 30.33% | 2.876 | 156.24 | 172.44 | 0.646 [0.629,0.662] | 462.44 |
| G4D8-learned-small-2048 | 10765/35935 | 29.96% | 2.405 | 163.13 | 180.54 | 0.674 [0.657,0.691] | 174.94 |
| G2D8-2048 | 3670/50020 | 7.34% | 2.930 | 115.42 | 123.73 | 0.477 [0.457,0.498] | 462.44 |
| G1D8-2048 | 6450/44505 | 14.49% | 2.893 | 127.68 | 138.26 | 0.528 [0.507,0.548] | 462.44 |

G8/D8 and G4/D8 learned rows use trained bounds initialized at 1. The compact
G4 recipe additionally freezes fusion, attention Q/K and attention V/O weights
at Q4_0; the full G8 recipe retains BF16 protected weights. G2/D8 and G1/D8 use
dynamic absmax bounds and BF16 protected weights. All W1 FFN rows use F32
scales/midpoints and positive gains; all share the unchanged target Q6_K head.
Their actual activation arithmetic is INT8-code DP4A at every logical width.

## Matched dynamic precision anchors on consumed DEV

All four rows start at the same frozen foundation, reset Adam identically,
and receive 1,024 updates / 15,360 supervised position exposures in the same order.
This table isolates logical gate/up precision; the recipe finalists above do not.

| Anchor | Accepted/proposed | Acceptance | Request tok/s | Request/Q4 | Drafter MiB |
|---|---:|---:|---:|---:|---:|
| G8D8 | 3800/19615 | 19.37% | 141.72 | 0.556 | 462.44 |
| G4D8 | 2060/23085 | 8.92% | 122.51 | 0.481 | 462.44 |
| G2D8 | 1705/23780 | 7.17% | 118.83 | 0.466 | 462.44 |
| G1D8 | 2985/21220 | 14.07% | 132.26 | 0.519 | 462.44 |

This report covers the modern Qwen3.5-4B Q4_K_M lane. DFlash Q4_K_M is the
frozen performance comparator; target-only is the correctness reference. Other
target sizes/precisions and historical Qwen3/DSpark results are outside this lane.

## Comparison protocol

The finite search contains 28 named zero-phase arms: a 16-point gate/up × down
activation grid, bounded component/range mutations and independent Q8_0/Q4_0
weight probes. One attention-VO zero control duplicates a probe's effective
format. Four dynamic gate/up anchors and eight mutations receive 1,024 matched
updates from the same frozen essential-feature foundation. Two global
interactions and two one-window continuations follow the predeclared caps.
Four recipes, selected before final confirmation training, finish matched 2,048
updates with exact optimizer/RNG/data-order continuation. The four dynamic 1,024
anchors isolate activation precision under common history; finalists that also
change range policy or protected weight formats are recipes, not pure bit-width
attribution.

The foundation trains binary signs, nonnegative row scales, row midpoints,
positive channel gains and 26 original norm gains. It starts from preserved vanilla 1800 and trains 2,048 updates; midpoint/gain removal and unchanged vanilla continuations
use the same parent/order/exposure. Gate/up share widths and gains; down has
separate gains. The actual training equation is `u = Q_A(g*x)` and
`y = s*sum(sign*u) + m*sum(u)`: the midpoint uses the same quantized stream.
Protected weights remain frozen; explicit weight reductions quantize those
frozen sources. A private drafter head is charged in full. The target's tied
Q6_K embedding/head is unchanged and charged once in combined deployment bytes.
Private embedding reduction is blocked.

All logical activation widths execute the actual native INT8-code DP4A path
with packed 1-bit FFN sign weights and F32 affine/gain metadata. A1 is a distinct
two-level signed quantizer (zero maps positive; dynamic all-zero scale is 0).
A2 uses three levels. A4/A8 use symmetric signed RNE grids; saturation/scaling
and straight-through gradient contracts are documented in the campaign checkpoint.
There is no qualified packed-activation binary kernel in these results.

Formal DEV uses 53 arms and 12 already-consumed natural prompts (4 per domain).
Four frozen 2,048 recipes plus target/Q4 use 24 fresh singleton components
(8 per domain), selected only after model/runtime/context-policy lock. Native
vocabulary admission checks full prompts before ranked selection. Each panel
uses 2 excluded warmup requests inside every server, 5 alternating repetitions,
and paired plain/traced execution. Speed uses plain requests. Token IDs, text,
stop semantics, native counters and prefix survival are audited against raw
responses and target-only. Actual cache reuse is 0. Target, runtime, F16 KV,
4,096 context, 256 output reserve, greedy policy and frozen depth 2 settings match.

## Exposure and qualification

The preserved vanilla reference completed 14,231 updates in 1,800.021702 measured
update seconds; zero/300/900/1800 checkpoints remain unchanged. This campaign
performed 27,658 physical updates, 3,634.355101 measured update seconds and 414,870
supervised position exposures, including qualification. A standard 1,024-update
window has 15,360 position exposures over 1,750 unique native label positions;
overlapping windows reuse labels across 24 TRAIN captures. Exposure is not a
count of distinct source tokens.

Exact CLI resume restores parameters, Adam/RNG and sampler state; explicit
adaptation resets Adam and preserves the inherited data cursor. Checkpoints
exclude frozen-buffer duplication and retain full recovery state losslessly.
Native exports are real GGUF. Dictionary-compressed GGUF archives hydrate to
verified original bytes before loading; archive ratios are never model-size or
inference-compression claims. Admission, deployed protected codes, 36 N4 and 36
N16 gained-stream CUDA cases, 12 CUDA gradient cases and actual full-model
training/native panels provide separate evidence. Seventeen CPU fixtures
qualify final-log recovery and source-selection geometry.

## Limits and retained failures

Full-model HF/native numerical parity was not established: the historical
untrained gate failed despite isolated FFN relative error near 1.24e-7. Native
features/labels/verifier ancestry is preserved; integer operation checks do not
prove full-model parity. Actual draft acceptance and throughput are measured
in addition to final target-output correctness.

Runtime03 flushed tiny signed values in a generic gain operation; runtime04
fuses gained-stream packing with explicit rounding and passes independent
byte-exact checks. Affected controls were remeasured; old runtime series remain
separate. DEV's live log reader sampled two adjacent requests late. Complete
reaped-server logs reconstruct every boundary and agree with independent HTTP
accepted/proposed counts; original progress/raw responses/timings are retained.
Two post-DEV wrappers failed before GPU work or fresh source access and were
replaced; native 16-slot/tokenizer qualification then passed before heldout lock.

Traced host spans include generation, packing/gains/midpoints/head execution,
catchup and acceptance. Asynchronous GPU work may be charged at later sync;
these are diagnostic host costs, not CUDA-event timings. Host-span subtraction
is not a throughput upper bound. Sampled device peaks include display/unrelated
baseline and can miss transients. Model-buffer allocations are separate from
main-context-only allocator summaries. Confidence intervals resample whole
prompts and repetitions; they do not measure independent-training variance.
Short inputs, one target lane, small TRAIN coverage and untested calibrated
A1/A2 range recipes limit generalization. Keep overlapping near ties visible.
