# November 10 model-artifact retention

Philip authorized retirement of obsolete one-off checkpoints and generated model
exports, preserving current HE4 work, baselines and useful fallback candidates.
This supersedes historical blanket retention of every intermediate state. It
does not authorize deleting source, measurements, captures or another project's
active work. The HE4broad run is paused with a complete 20,671-update recovery
state; no sustained training restart is part of this cleanup.

## Retained candidates and why

| Retained material | Decision value / evidence |
| --- | --- |
| Current November 10 HE4broad: every current checkpoint, export, original/new source and newest dataset | Active experiment; preserve exact parameters, Adam, RNG, order, cursor and cumulative clocks. [Current report](../he4broad-20261110/report.md). |
| Original HE4-confirm, its B0/dictionary parents, and final November 7 continuation | Faster-draft endpoint and negative longer-training control. The continuation reduced consumed-DEV acceptance from 22.61% to 17.85%; keep final state and results, retire intervening optimizer states. [Continuation](../he4-continuation-20261107/report.md). |
| Csmallbroad original 4,096-update state and exact accepted continuation, with native bytes and dependencies | Compact fallback: 191.76 MiB private model/cache/rotation versus HE4 601.51 MiB. Formal heldout request speed 168.95 tokens/s, 0.714 of external Q4; longer-context synthetic panels improve versus its own matched control. It is not the deployment incumbent. [Formal evidence](../component-campaign-20261105/report.md), [continuation](../dflash2-smoke-20261106/report.md). |
| November 8 source-initialized HE4broad zero and final checkpoints/exports | Prior small-dataset trajectory, distinct from the new run. Preserve the curve and final comparison; intermediate optimizer states are superseded. [Evidence](../he4broad-20261108/report.md). |
| Qwen3.5 target, released BF16 DFlash source, external DFlash Q4, baseline registry and runtime04 | Fixed correctness/source/throughput references for current decisions. All current `spec-shortlist` model files remain. |
| Frozen modern W1A8 baseline and resume qualification; DSpark Q4 comparison and source model directories | Retain representative baselines and reproducible controls without retaining every historical one-off fork. [W1A8 protocol](../../docs/CANDIDATE_BENCHMARKING.md). |
| Component/precision parent states, dictionary bases, source and compact findings | Recovery dependencies for retained native/model states must not be removed merely because a child result is preserved. |

The initial reviewed batch contains 61 large superseded model artifacts, about
227 GiB logical size. A broader per-file inventory includes smaller candidate
exports and Windows-side intermediate states. [Initial batch](large-model-plan-summary.json)
and [retirement summary](retirement-summary.json) record scope and outcomes.
Raw per-file plans and deletion receipts are under
`runs/he4broad-20261110-01/pause-fix-20261110/`.

## Evidence and deletion rules

All original non-model result/config/chart/log files remain in their historical
run folders. Existing Git-visible campaign reports, figures and machine-readable
results remain. [Selected retired-campaign summaries](retired-campaign-evidence.json)
preserve bounded result/config excerpts with their source hashes when the full
inventory completes. Retirement does not claim deleted states are resumable.

The maintained [retention command](../../scripts/he4fresh/retire_history.py)
protects the current run, selected baselines/frontiers, source trees, input data,
recovery bases and directly opened files. Execution verifies each planned file's
type, device, inode, size and modification time before unlinking. Hard links can
make logical totals larger than actually recovered space; measured filesystem
space is reported separately. Model deletion is intentional, not an archival
backup claim.

Before Philip clarified the retirement policy, 43.47 GiB of inactive reproducible
CUDA caches was cleared and a bulk historical offload was started. The bulk
offload and retirement worker were then stopped. Completed verified cold copies
and their local symlinks remain inventoried; incomplete copies are disposable.
The newest training dataset is being copied to local ext4 and will be admitted
once in full before removing its 2080 Ti source. Its runtime loader will not
rehash or rescan full feature arrays on every window.
