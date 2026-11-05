# Fixed versus learned ranges: bounded follow-up

One additional matched control completed after the initial full chart was
published. At 1,024 updates, compact G4/D8 with **fixed unit bounds** is faster
than the identically exposed learned-bound recipe on a new held-out cohort:
**167.51 versus 162.21 request tok/s**, fixed/learned **1.0327×**
[95% paired bootstrap: 1.0180–1.0484]. Acceptance improves by 2.62 percentage
points [1.45–4.01]. DEV gives no clear difference: ratio 0.9999
[0.9817–1.0198]. Retain both outcomes; this is one matched training history,
not independent-training variance or a claim about all quantizers.

Q4 still wins: 245.61 request tok/s on this cohort, target-only 172.79.
Fixed W1 reaches 0.682× Q4 and remains slower than target-only. No W1 deployment
promotion is supported. Learning bounds is not automatically beneficial for
this recipe/budget. Keep fixed unit bounds as a useful control before extending
range learning; other initializers, families and longer histories remain untested.

[Combined interactive chart](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-clip-control/comparison.html) ·
[all 258 records](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-clip-control/full-table.md) ·
[JSON](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-clip-control/comparison.json) ·
[CSV](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/charts-clip-control/comparison.csv) ·
[publication](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/clip-control-followup/completion.json).

## New held-out measurements

Five alternating repetitions, two excluded same-process warmups per server,
paired plain/traced execution and 24 new whole source components. The cohort
label is `clip-heldout24`; it is distinct from the initial `heldout24`.

| Arm | Accepted/proposed | Acceptance | Request tok/s | Decode tok/s | Request/Q4 [95%] | Raw drafter MiB |
|---|---:|---:|---:|---:|---:|---:|
| target | — | — | 172.79 | 188.36 | 0.704 [0.666,0.748] | 0.00 |
| q4 | 16245/21205 | 76.61% | 245.61 | 288.68 | 1.000 [1.000,1.000] | 363.78 |
| G4D8-learned-small-1024 | 10090/33500 | 30.12% | 162.21 | 180.04 | 0.660 [0.642,0.679] | 174.94 |
| G4D8-fixed-small-1024 | 10620/32435 | 32.74% | 167.51 | 186.46 | 0.682 [0.661,0.704] | 174.94 |

Both W1 rows use packed 1-bit FFN weight formats, G4/D8 logical activations,
INT8-code DP4A execution, frozen Q4_0 fusion/attention weights, F32 row
scale/midpoint/gains/norms, target-shared Q6_K head and F16 KV policy. Native
inference code and target/verifier are unchanged. Both start with bound 1;
only training those bounds differs. Learned native bounds span 0.995856–1.009629.
This intervention also changes the later FFN training trajectory; it is not an
inference-only bound swap. Model bytes are identical. No A1/A2 packed-activation
kernel, new architecture, loss or target precision was introduced.

## Matching and recovery evidence

The control starts at the same frozen essential foundation, resets Adam and
consumes exactly the same 1,024 `(sequence,anchor,update,cursor)` records as the
learned control: 15,360 supervised position exposures, final cursor 17,303. Actual
fixed training takes 146.605614 measured update seconds; no continuation or
replacement training is added. Qualification and all raw panels are audited in
`data-completion.json`. Initial held-out hashes, groups and topics are exclusions
before new source selection; new source-component disjointness passes. No model,
epoch, initializer or range was selected using the new held-out outcomes.

The streamed Zstd checkpoint retains the full F32 state without a second raw
file on disk. Exact optimizer/RNG resume fixtures pass; actual full-model recovery
restores 92 trainable Adam states and CUDA RNG. Eighteen fixed clip constants have
no Adam state. The first recovery checker counted those constants incorrectly;
its replacement reuses the same completed training. A checkpoint-derived native
GGUF is byte-identical to the model-resident update1,024 export. The paid archive
is 4,700,294,541 bytes. Owned RAM temporary free space returns to its measured
starting value; no previous checkpoint is moved/deleted and no Windows archive
relocation is performed.

The combined package contains 258 panel/arm records, including 67 formal confirmed
rows and 10,580 formal responses (9,240 after excluded warmups), all target-identical.
There are 30 raw panel audits across initial and follow-up data. Smaller diagnostic
panels remain explicitly labelled. All 16 initial publication artifacts retain
their original hashes. Host spans are diagnostic, sampled memory includes desktop
baseline, and full-model HF/native floating parity remains unresolved. The
clipping effect intervals resample paired prompts/repetitions, not training seeds.
Do not compare raw rates between the two distinct held-out cohorts as a training
curve. The deployment incumbent remains DFlash Q4_K_M.

## Release and retained evidence

All owned native/training jobs are terminal, and all recorded owned process
identities/groups are absent. RTX 5080 is idle at the measured 1352 MiB desktop
baseline. No pushes or parent integration were performed. The active broader
nine-model goal remains open; this completes the modern 4B/Q4 campaign and its
bounded control, with no further training allocated.

All paid recovery checkpoints, captures, native export archives and their common
raw dictionary basis remain protected. One qualified reference-format example
is retained; 28 redundant reference containers and 42,529 bytes of completed
scratch were removed. Logical reference-cache removal is 1,527,527,166 bytes.
Measured WSL free increased 9,193,504,768→10,721,099,776 during that pass; Windows
free was 10,931,777,536 at both measurements. Final small scratch cleanup measured
WSL 10,373,267,456→10,373,308,416 and Windows 11,559,485,440 unchanged. Concurrent
activity can affect these observations; no Windows/VHDX reclamation is claimed.
The full retained inventory and process proof are in
[final-storage-process-inventory.json](/home/philip/binary-eagle-decoding/runs/precision-campaign-20261104-01/final-storage-process-inventory.json).
