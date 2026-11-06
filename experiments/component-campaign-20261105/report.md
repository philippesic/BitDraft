# Remaining precision component campaign

Execution is in progress. No component frontier or throughput promotion is claimed.
The exact learned-small2048 optimized G4/D8 candidate remains B0.
Target Qwen3.5-4B Q4_K_M, target/verifier F16 KV, decode-consistent32 and frozen
external DFlash Q4_K_M remain unchanged on the local RTX5080.

## Actual qualification

The full recoverable B0 checkpoint was authenticated against its native sidecar.
A32-update R1 pilot took4.882actual update seconds; median0.143224seconds/update,
projecting586.65seconds per4096-update existing-parameter arm. Peak torch CUDA
allocation was10,597,126,656bytes. Full F32 masters, Adam, CUDA RNG and cursor
recovered, with byte-identical native export. The streamed byte-shuffled checkpoint
is4,307,213,897bytes, with no raw disk duplicate. R1 freezes vocabulary weights
but includes the actual selected quantized head/embedding in upstream gradients.
It is not full vocabulary-master QAT.

Exact private Q6_K copy, private Q4_K head, private Q6/Q4 embedding and one tied
output-placed CUDA Q4_K matrix admit natively. The six requested K/V pairs and
F16-storage scalar groups also admit and produce target-identical final tokens
on three already-consumed64-token captureDEV requests each. K/V native logs
confirm real buffer types and Hadamard rotations, K128/V64. CUDA fixtures reproduce
Q8_0/Q4_0/Q5_0 write/read quantization exactly on2432values/format; gradients pass.
All short native screens are unwarmed diagnostic evidence, not final speed claims.
The first tied-input allocation was slow; the retained runtime03 aliases the
embedding to a single CUDA output tensor. Prior qualified runtime files are unchanged.

## Current recovery

H0 is continuing to4096additional cumulative updates, resuming the exact32-update
pilot state. At256updates, consumed-DEV CE is6.67338 versus6.89435 at zero;
agreement14.07% versus11.11%. H4 follows from the same frozen B0 with matched
example order/exposure and optimizer policy. Each primary recovery window retains
0/256/1024/4096exports and native readouts. Embedding, KV, scalars, finite head
probe, slow-starter reserves, combinations and locked confirmation remain pending.

## Data and storage limits

Current TRAIN consists of24native DFlash captures,1426overlapping windows and1762
unique supervised label positions. A4096-update arm consumes61440supervised
positions through repeated exposure; these are not fresh labels or broad new
generalization evidence. DEV is already consumed and remains separate from TRAIN.
No new sealed heldout payload has been opened.

All pre-campaign paid checkpoints, source weights and raw GGUF dictionary bases
remain protected. Only bounded NEW campaign rolling states retire after verified
replacement. Native snapshots use authenticated tensor-slice archives with exact
full-file byte reconstruction; archival size is not inference memory. Windows
conversion once hit its512MiB free-space floor; the original survived, disposable
partial was removed and a subsequent conversion passed with measured headroom.
The retained native bases charge their true bytes, private Q4 adds357,580,800bytes
to the immutable target deployment, and private Q6 adds521,472,000bytes.

## Evidence and live ownership

Canonical raw root: `/home/philip/binary-eagle-decoding/runs/component-campaign-20261105-01`.
See `freeze.json`, `budget.json`, `qualification/result.json`,
`checkpoint-storage-qualification.json`, `zero-smoke-summary.json`,
`native-head-embedding-zero02/result.json`, `native-remaining-zero/result.json`,
and `arms/*/{resolved,readouts,result}.json`.

Provisional [interactive charts](/home/philip/binary-eagle-decoding/runs/component-campaign-20261105-01/charts/comparison.html),
[full JSON](/home/philip/binary-eagle-decoding/runs/component-campaign-20261105-01/charts/comparison.json),
[CSV](/home/philip/binary-eagle-decoding/runs/component-campaign-20261105-01/charts/full-table.csv)
and PNG/SVG recovery/tradeoff artifacts are regenerated from actual readouts.
Null fields remain unmeasured. Formal two-warmup/five-alternating-repetition
comparisons and final charts are still required.

Live supervisor ownership and stop procedure are in
[COMPONENT_CAMPAIGN](../../docs/COMPONENT_CAMPAIGN.md). No pushes or gitlink updates.


Checkpoint15:29Pacific: the first head, embedding and tied head+embedding R1
windows completed4096updates each; H3 also completed the finite lower-precision
probe. Cache and scalar primary recovery now follows the actual CUDA read gate.
The initial gate error was an unsupported generic fixture copy, corrected to the
native flash-attention conversion. All Q8/Q4/Q5 fixture values then match exactly.
Formal latency, tied allocation audit, combinations and heldout remain pending;
this checkpoint is not campaign completion.


## Completed initial windows —16:14Pacific checkpoint

Every row below has4096additional logical updates from B0 and four actual native
readouts,0/256/1024/4096. Exact H6/E6 frozen-Q6 forward controls reuse H0's
identical physical training; all other rows consumed their own4096updates.
CE is full-vocabulary cross entropy on135consumed DEV positions (lower is better).
Native counts are three64-token consumed-DEV requests and remain diagnostics,
without final timing or independent training-seed confidence intervals.

| Component | Final DEV CE | Native accepted/proposed | Added private vocabulary tensor bytes |
|---|---:|---:|---:|
| H0 sharedQ6 control |7.05519|55/263|0|
| H6 exact privateQ6 head |7.05519|55/263|521,472,000|
| H4 privateQ4_K head |7.32178|49/273|357,580,800|
| H3 privateQ3_K head |7.09648|58/257|273,152,000|
| E6 exact privateQ6 embedding |7.05519|55/263|521,472,000|
| E4 privateQ4_K embedding |7.07413|57/259|357,580,800|
| HE4 tied privateQ4_K |7.12048|55/261|357,580,800|
| K8F Q8_0K/F16V |7.18649|54/265|0|
| KF8 F16K/Q8_0V |7.47662|55/262|0|

The private head does not replace the immutable target head: even Q3 adds memory
versus target-sharing. These are exact vocabulary tensor payloads, not GGUF file
size or archive ratios; actual CPU/GPU buffer placement and peak are separate.
HE4 has one vocabulary tensor in its native file; native pointer/device-sharing
audit is still pending. The target/verifier keeps F16KV in every row. K88 is now
training, followed by the three lower-bit cache pairs and six scalar windows.
No new heldout data has been opened, no precision frontier or throughput promotion
has been declared, and the campaign is not complete.


Checkpoint17:45Pacific: all6draft-cache windows and the first scalar window are
complete. Sscale stores251,904fewer tensor bytes, has zero new scale underflows,
and ends with61/250accepted/proposed on the short native diagnostic. Its dense
DEV loss is higher than H0, while token agreement is19/135versus18/135. These
are actual consumed-DEV outcomes and do not establish a deployment benefit.
Smidpoint and the other four scalar windows remain in the serialized queue.
