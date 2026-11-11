# Remaining port stages

This is the consolidated plan for the work that remains after stage 0, the non-source foundation already in this repository at the former path `/home/philip/binary-eagle-foundation`. The remaining stages are source port, benchmark, GPU qualification, and skills. Nothing advances a stage without Philip's explicit go. This document is planning only; it does not authorize training, data transfer, cleanup, or any later stage.

## Sources

- `/home/philip/.t3/scratch/2026-11-10-read-only-binary-eagle-porting-bc9c678e/PORTING_PLAN.md` and `SUMMARY.md` on the 5080 (DESKTOP-E78LJSI), prepared 9 November 2026. Cited as the porting plan.
- `/home/philip/.t3/scratch/2026-11-10-read-only-binary-eagle-foundation-c04593ea/FOUNDATION_PROPOSAL.md` and the stage-0 build that followed it.
- This repository's `README.md`, `FINDINGS.md`, `docs/benchmarking.md`, `docs/training.md`, `docs/data.md`, `docs/runtime.md`, `docs/system-map.md`, and `docs/decisions/`.
- `/workspace/research/migration/BRIEF.md` and `/workspace/research/migration/binary/foundation-build.md`.

## Current state

Stage 0 is complete as a local repo with no remote, no model, trainer, capture loader, ledger importer, or benchmark. The check command is `uv run --locked --no-sync --offline python tools/check.py`. Runtime verify is a documented refusal until the hardware stage. The old repo `/home/philip/binary-eagle-decoding` stays frozen. The plan's foundation estimate was 2.5–4 engineer-days; later work 13.5–22, for 16–26 total, plus acquisition and review, not including actual training.

Philip's names stay in force: HE4 is W1A4 with A8 on the down projection and a Q4K head. W1A8 is old DSpark W1A8. DSpark Q4KM is the old baseline. Baseline is the current external DFlash Q4KM drafter. No claim that HE4 beats the Baseline.

## Source port

### Scope

Build one auditable HE4 data, training, and export path. The plan's source steps, in order:

- Ledger importer from actual catalogs, selectors, captures, and consumption, with per-row count and reason reconciliation; immutable train, dev, cal, and heldout roles; `trained-by` coverage. Metadata only at first (plan step 3).
- One bounded capture, window, and shard reader and teacher-adapter contract; finalized 2080 Ti manifest intake and recapture reconciliation. No bulk transfer until separately approved (step 4).
- Numerical, model, loss, init, and export primitives; one HE4 constructor; explicit head, embedding, and storage precision (step 5).
- One trainer with fresh, resume, and branch, synchronized measurement, cumulative cost, exposure, role, and disk preflight, and a cooperative complete-update stop (step 6).
- Transactional full-state and export retention, 30-minute DEV and serving evaluation protocol, atomic keep-best selection, and overfit stop (step 7).

What is not ported here: the benchmark CLI, native runtime04 rebuild as a performance entry, GPU jobs, copied campaign trainers, run trees, checkpoints, or skills. heldout24 never enters training or calibration.

### Entry criteria

Philip's go after reviewing stage 0. Comment stripping and derivation relocation accompany every extraction. Capacity and output placement should be decided before large reads or writes; at plan time WSL had 5.3 GiB free and C: had 6.4–8.6 GiB, which failed the proposed 8 GiB post-write reserve.

### Exit evidence and gates

- P1 and P2: full per-row accounting; two absent-topic rows independent; exact exclusions and aliases; heldout24 denied at every training seam; expected counts and disk, data, and exposure limits reject before a run. The 5,505 versus about 5,497 prompt difference must be explained before admission.
- P3 and P4: source BF16 latents, scales, masks, and parameter groups match; tiny CPU forward, loss, gradient, and update references within stated tolerances; exact sign, bit order, packing, and GGUF metadata. CPU equivalence does not certify native rounding.
- P5: exact CPU resume of parameters, Adam counters, moments, RNG, sampler, and exposure, without tensor alias mutation; failed write preserves the old complete state; keep-best is exercised with a deliberately peaking trajectory.

Plan effort: 2–3, 1–2, 2–3, 2–3, and 1–2 engineer-days for those five source steps.

### Risks

Data provenance can drift when corrected batches, null topics, and aliases change totals. Historical states have different optimizer schemas. Training surrogates differ from native quantized arithmetic. More labels are not a guarantee against overfit. The plan authorizes no cleanup to make room.

## Benchmark

### Scope

Sole public entry: `uv run --locked python -m benchmarks`, with `plan`, `run`, `verify`, `report`, and `qualify` modes (plan section 11 and steps 8–10). Reconstruct the runtime04 source, patch, and toolchain closure in an external clean build. Discover baseline and workload adapters from separate files. Training uses one typed CLI and calls the benchmark API for scheduled scoring; it does not shell into a second comparator.

Workloads in the plan: native smoke; capture correctness; DEV diagnostic; formal runtime04 panel (initially consumed DEV3, depth 2, context 4096, greedy, cap 256, two excluded warmups, five alternating process repetitions); larger serving selection; heldout confirmation; context or resource stress; recovery and operational.

What is not in: `--force`; mixing clocks, eras, or workloads; treating smoke timing as a ranking result; pooling host-traced draft time with GPU draft event time; promoting on consumed DEV alone.

### Entry criteria

Philip's go after source parity review. Resource preflight before compilation. No absolute dependency on old run objects.

### Exit evidence and gates

- P0 and P4 for the clean native build and binary identity.
- P7: complete matched process blocks; explicit metric clocks and era; metadata mismatch refusal; failed attempts retained; every deliberate mismatch corruption fails; verified complete raw counts.
- Selection objective is valid complete-request throughput versus a matched Baseline and target-only panel, with exact greedy output correctness. CE and acceptance are diagnostics, not proxies for serving superiority.

Plan effort: 2–3 days for the runtime closure, 2–3 days for the CLI and refusals, 0.5–1 day to reconcile docs and inventory.

### Risks

Runtime04 links into multiple ignored execution trees; a missing source closure is a blockage, not a reason to reuse opaque binaries. Formal legacy defaults are replication references, not universal sufficiency.

## GPU qualification

### Scope

Separately authorized 5080 work (plan step 11): tiny real verify, then source and native parity probes, then Baseline, target, and candidate smoke, then formal consumed-DEV reproduction, then a real next-update recovery only if authorized. Then finalize a new training-run plan for Philip. Training allocation still needs its own approval.

The planned real verify uses two fixed synthetic, non-heldout prompts, at most 32 prompt tokens, greedy cap 16, context 256, concurrency 1, depth 2, identical pins and F16 KV. Every emitted token ID, text, and finish reason must agree; the drafter must actually be active. On one prompt, capture at most 16 generated positions and verify taps, geometry, and full-logit argmax. Caps proposed in the plan: 128 MiB total output and temp, 5-minute GPU-wall timeout. These caps were not measured.

What is not in: calling this a proof that HE4 beats Baseline; 2080 Ti serving qualification; other models, stochastic decoding, or longer contexts, unless separately extended.

### Entry criteria

Philip's go after the benchmark review. Disk preflight must pass. At plan time, 5.3 GiB WSL free failed even this probe under the proposed 8 GiB reserve.

### Exit evidence and gates

P6 and P7 with hash-matched check plus verify receipts. Bounded disk and GPU cost. All failures retained. Binary presence or HTTP 200 is not enough. If no formal panel completed, status must say "no serving-selected checkpoint," not "best."

Keep-best, already designed in the source stage, must hold on hardware: HE4broad DEV top-1 reached 17.7778% at 14,400 seconds and ended 13.3333% at 28,800 seconds; formal serving was scored only at 0 and 8 hours. The hour-4 archive existed; the missing piece was a validated best-serving selection policy.

Plan effort: 1–2 engineer-days plus approved hardware time.

### Risks

Numerical mismatch between training and native paths. Device timing needs declared event and graph semantics. Interrupted scoring must never replace best.

## Skills

Out of scope. The porting plan and migration brief leave agent skills for a later request. There is no inventory, entry criterion, exit gate, or effort estimate.

## Open questions awaiting Philip

From the plan's section 14. Foundation used recommended defaults for stage 0 only; these remain open for later stages.

1. Permanent name, home, and remote; license and publication visibility. The directory is being renamed to `binary-eagle-decoding-next` to match the original repo name. That does not choose a public remote.
2. Disk, artifact capacity, output placement, and backup. Default: external storage plus finalized 2080 Ti manifests and a bounded read cache. Do not delete old states.
3. Initial support breadth. Default: HE4, target-only, and external DFlash Q4KM on modern Qwen3.5-4B Q4_K_M, runtime04-equivalent, RTX 5080. Preserve Csmallbroad, B0, and foundation identities; keep DSpark and other routes historical.
4. Capture recovery acceptance. Default: finalized prompt-level manifest and an explanation of the eight-prompt difference (5,505 versus about 5,497). Gross 2080 Ti counts (10,150 sequences, 970,121 windows, 1,111,811 labels, reported by Philip) stay provisional until admitted.
5. Training budget and exposure. Proposed guards, not an approved run: at least 1 million admitted unique positions, at most 2 window epochs, mean and max at most 30 visits.
6. DEV, serving, and stopping. Default DEV60 every 30 update-minutes; official serving at frozen mid-run points (illustrative 0/2/4/6/8 hours if an 8-hour run is authorized); three-check 5% relative CE deterioration plus no top-1 improvement stops.
7. Deployment margin and an untouched confirmation source. Historical heldout24 stays evaluation-only and already consumed.
8. How much historical state-schema conversion to implement first.
9. CI approval wiring and build-artifact hosting. The full comment ban is settled and is not an open question.
