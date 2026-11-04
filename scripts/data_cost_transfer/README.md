# Bounded data and decoding-cost campaign

These are versioned adapters for the retained Qwen3.5-4B Q4_K_M / row-W1A8
reference on RTX5080. They do not choose an architecture, precision or long-QAT
recipe. The general runner, storage readers and comparison gates remain in
[`contender_pipeline`](../contender_pipeline/README.md).

The campaign lives in `runs/data-cost-transfer-20261104-01/`. `scope.json`,
`training-recipe.json` and the frozen job plans define its bounds. Large arrays,
checkpoints, native responses and raw logs are outside Git.

## Reusable training assets

`bank-inputs/bank.json` contains192 new public-catalog TRAIN singleton groups,
64/domain. Selection authenticates historical, consumed and opaque partition
exclusions before selecting inputs. Whole user prompts have at most280 UTF8
bytes; capture additionally requires prompt+80 output tokens to fit384 positions.
This is a short-input bank, not long-context coverage or pretraining novelty.

`registered-bank/registry.json` indexes admitted chunk manifests;
`registered-bank/shards.json` references bounded F32 shards. Every training window
and required native embedding was checked at admission. `training_bank.py` uses
the common metadata/array readers with two cached manifests and two mapped array
shards. Those limits do not bound retained caller views or OS page cache.
Captured features remain tied to the recorded target, taps, initializer and native
history semantics. Different choices may require recapture; source prompts and
exclusion ancestry remain reusable.

**Apply `train-exposure.json` before any future held-out selection.**
`exposure.apply` adds raw/normalized identities, source IDs, groups and topics.
Groups remain TRAIN even if capture or training fails. This does not make
previously consumed evaluation inputs fresh again.

## Capture and recovery

`bank.py` reuses the authenticated native producer and its output/trace checks.
`local_capture_v2.py` explicitly places new captures in the campaign's WSL
filesystem. It preserves runtime settings and byte bounds. The earlier C: path
was substantially slower for this many small trace files.

`native_log_text.py` decodes only native `server.log` files with lossless
`surrogateescape`, checking an exact byte round trip. Native token pieces can
contain incomplete UTF8 sequences. HTTP JSON remains strict; no token IDs or
bytes are replaced. Install this compatibility reader only in the scoped
capture/preparation/native-comparison process.

`resume_capture_v2.py` reuses completed, authenticated and released groups from
an earlier attempt, in a new directory. Failed outputs are preserved. The older
adapter files are retained because captured provenance references their exact
paths and bytes; do not edit or remove pinned sources in place.

## Fixed data comparison

`train.py` narrowly derives the frozen fixed-objective trainer. The control keeps
its original two CE windows per update. The broader arm uses the first original
window and one deterministically shuffled new window. Both keep the fixed4 CE
denominator, static reachability masks, old356+added407 replay, optimizer, parent
and representation. `training-recipe.json` fixes1920 additional updates/arm and
native checks at480/960/1920. Equal updates are not equal seconds or label counts.

Before continuation, `check_states.py` requires exact old-reference parameters,
Adam/RNG/exposure and export, plus exact broader-data split/resume state/export.
The common native adapter then checks new exports. Intermediate checkpoints are
exported without extra updates. Native panels compare the same frozen baselines;
`consumed_panel.py` separately reuses the prior natural24 performance cohort.
Neither is fresh confirmation. `analyze_transfer.py` computes paired,
domain-stratified prompt/repetition intervals and never applies permanent pruning.

Use the common launcher to execute frozen plans through explicit stages. Never
retry into existing outputs or edit a live plan. `checkpoint.json` and STATUS
identify the active supervisor, identity receipt and next authorized stage.

## Decoding costs

`cost_trace.cpp` interposes the pinned runtime ABI and records disjoint outer
host spans for target processing/synchronization, draft generation, catch-up,
acceptance and verification sampling. Nested spans are diagnostic and must not
be added to outer totals. It does not add synchronization or change model calls.
Asynchronous GPU work can be charged at a later synchronization.

`host_profile.py` qualifies the instrumentation. `paired_profile.py` interleaves
five traced and untraced repetitions, requiring identical outputs and draft/
round counters. Uninstrumented results remain the speed measurement.
`kernel_profile_v4.py` uses bounded Nsight CUDA/NVTX traces for additional GPU
attribution. `analyze_costs.py` checks raw span totals and reports instrumentation
overhead. Its zero-draft sensitivity holds other measured costs and rounds fixed;
it is not a hard performance ceiling or an architecture predictor.

The successful kernel adapter authenticates a server independently of a main-thread
PPID chain, while requiring its frozen arguments, unique port, task-specific trace
environment, birth identity and loaded image pins. Earlier versions remain because
failed plans pin them. `analyze_kernels.py` maps measured CUDA work to request and
enqueue phase with a bounded host/NVTX clock margin; uncertain phases remain
unattributed. `cost_contrast.py` supplies paired uncertainty for complete draft
host cost, and `cost_ledger.py` separates physical updates from supervised wall.
