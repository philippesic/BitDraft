# Saved controls let complete HE4 training checks fit within two minutes

The routine check runs only the candidate on 60 broad DEV prompts. Saved target,
DFlash baseline and prior-leader records supply the comparisons. Keeping the trainer
alive and mapping its just-saved state avoids rebuilding it after every check.

## Outcome

| Exercised path | Measured complete interruption |
| --- | ---: |
| Largest qualified capture, first check | 81.1 s |
| Largest qualified capture, second check | 77.0 s |

These intervals include checkpointing, GGUF export, all 60 requests, selection,
device restoration and readiness for the next training sample. Native screen execution
itself took about 60–68 seconds. The capture contained 8,591 tokens; its feature cache was
released before evaluation, so the next sample required cold materialization. Initial
startup and recovery downtime are separate. Earlier 75–84 second receipts ended at sampler
entry; the adopted clock additionally includes actual materialization.

Two earlier campaigns continued original update 220095 for three qualification updates on
the same 64-record FP16 TRAIN corpus. The normal campaign completed three screens. Another
campaign deliberately lost its native server after one request, retained the incomplete
attempt without selecting it, then resumed and completed three screens. All 430 model,
Adam and RNG tensors matched the normal campaign exactly afterward, including loss,
gradient norm, cursor and exposure. The final recovery checkpoint and the provisional
acceptance leader were different states and both remained preserved.

The largest-capture campaign also lost its final native server deliberately. It resumed
at the update guard, retried only that incomplete screen, and finished with exactly three
completed evaluations. Its 430 tensors and native export matched a separate uninterrupted
three-update control exactly. Selection stayed pending until the complete pause passed;
a deterministic over-budget test confirmed that such a proposal cannot replace the leader.

| Candidate-only broader panel | Q4KM target | NVFP4 target |
| --- | ---: | ---: |
| 120 ordinary requests |268 s |253 s |
| 18 prefix variants |30 s |25 s |
| 12 live tasks |108 s |87 s |
| Combined panels |6.8 min |6.1 min |

The complete two-precision suite took 13.1 minutes including admission and bookkeeping.
All fresh requests and trajectories matched saved target signatures. Target, baseline
and previous-candidate rows were historical; only the selected candidate executed.
Prefix reports retain all three instruction styles; live reports separate families and
models. Each target completed 8 of 12 live tasks; four model failures remained denominated.
These measurements do not show HE4 beating DFlash. Tiny differences between HE4 states
are descriptive historical comparisons, not evidence that two training updates improved speed.

Q4 controls came from complete historical panels with exact unchanged task projections.
Missing NVFP4 controls were established once. Its separate candidate-only 60-request
screen took about 56 seconds. Completed suite replay preserved the original result;
changed source was refused.

Earlier complete pauses of 122–179 seconds failed qualification. Keeping an entire
CPU model copy also exhausted swap under the host's agent memory limit. Those attempts
and their unfavorable timings remain evidence. The adopted handoff releases completed
gradients, retains frozen GPU buffers and uses file-backed checkpoint mappings for
trainable parameters and Adam moments.

## Scope and limitations

This qualifies execution on the RTX 5080 for the existing HE4/Qwen3.5-4B recipe. Three
updates are an integration test, not evidence of useful training or convergence. The
120-window teacher-forced diagnostics remain separate from native acceptance. FP16
storage adoption has its [own matched qualification](../fp16-storage-adoption/result.md).

The cohorts are reused DEV data, not independent confirmation or measured user traffic.
Historical timings remain descriptive even with matching hardware and serving settings;
close speed decisions require fresh matched trials. Live model failures stay in task
denominators. A request score cannot override worse application outcomes.

The injected failed screen reached its deadline and stayed unselected. Its elapsed cost
was retained. The gap across controller recovery has no continuous-pause measurement;
it is not counted as a successful sub-120-second interruption.

## Decision

Retain 60×128-token candidate-only screens with 1800–3600 optimizer seconds between checks.
Require authenticated compatible target/baseline records before training. Reuse the
initial candidate and later leaders when matching results exist. Keep first canonical
reference observations immutable and preserve deliberate additional observations.

Keep broader requests, prefix sensitivity and live outcomes separate. Qualify a new
model, precision, hardware or serving policy before reusing its measurements. Production
continuation must start from the original complete220095 checkpoint, preserving Adam/RNG;
qualification descendants do not replace it.

Closed, unselected checkpoints from discarded handoff attempts were retired through
Store receipts. Original data/state, selected states, exports, source snapshots and raw
execution records remain preserved; exact retired checkpoint blobs are unavailable.

## Evidence and reproduction

[Compact measurements and authenticated raw references](result.json) identify the
campaigns, comparison panels, assertions and exact executed snapshots. Raw artifacts
live under `/home/philip/binary-eagle-artifacts/qualification/lean-checks`.

Use the maintained [benchmark commands](../../docs/benchmarks.md): `prepare-checks`,
`record-reference`, `campaign-preflight`, `campaign`, `campaign-report` and `suite`.
Recorded qualification specifications deliberately limit training to three updates.
Resume their exact frozen source and recipe; never turn them into production budgets.
The state-parity receipt compares every model/Adam tensor in bounded chunks and checks
RNG, loss, gradients and exposure, excluding source/clock metadata that should differ.

## Artifact disposition

Retain the compact measurements and executed source linked above. Reusable saved
controls and active data/model inputs retain their dependency closure outside source.
Qualification-only descendants are disposable after their measurements and failure
receipts are verified; exact retired states cannot be resumed.
