# Port the HE4 path from the predecessor repository

## Goal

One auditable HE4 data, training, export, runtime and benchmark path in this repository,
replacing the per-campaign scripts of `/home/philip/binary-eagle-decoding`.

## Authorized

Identify and prepare the complete functional port before benchmark redesign and agent
operations. CPU investigation is autonomous; GPU work follows the task-ownership rule
in AGENTS.md. This inventory does not allocate a new training campaign or authorize
artifact deletion. Philip will link the remote; artifact backup and retirement automation remain deferred.

## Done when

The retained HE4 workflow can be installed, supplied real data, trained on the supported
device, stopped and resumed, exported, and loaded by a rebuilt native runtime with
correctness evidence. Current docs describe the supported product without requiring
port history. Benchmark redesign and agent operations follow this milestone.

## State

Done on CPU: metadata admission and the modern index (see FINDINGS F011), HE4 numerics,
transactional states, exact resume and branch, bounded supervision, retained source and
intent per attempt, verified GGUF export. Source inventory: [evidence](../evidence/source-inventory.json).
Agent exercises so far: [readiness](../evidence/agent-readiness.json),
[routine repair](../evidence/agent-window-exercise.json),
[failure diagnosis](../evidence/agent-diagnosis-exercise.json).

Not done: retained-artifact parity, native runtime, benchmark, selection, GPU
qualification. Runtime04 links objects across ignored run trees in the old repository;
a missing source closure blocks the runtime stage and is not a reason to reuse old
binaries. Historical receipts list Python 3.11.15, torch 2.14.0 with CUDA 13.0,
NumPy 2.4.6, zstandard 0.25.0, GCC 15.2.0, nvcc 13.1.115 and Nsight Systems 2025.5.2;
transformers was observed at 4.57.6 against a declared 4.57.1.

## Next

Work through the prerequisite inventory below. It identifies work, not a claim that
these capabilities have been completed. Preserve original artifacts and live jobs.

## Prerequisites before benchmark redesign

1. Reproducible installation: rebuild numerical dependencies in a fresh environment,
   and document the native compiler/library setup once recovered. Check-only setup works;
   current numerical verification reuses host libraries.
2. Real source/data equivalence: validate actual retained inputs, masks, forward/loss,
   gradients, updates and exports. Current reduced CPU fixtures and reconciled index
   metadata do not prove full-size equivalence. Resolve the eight-prompt discrepancy
   from F005 before claiming that recapture is fully reconciled.
3. Checkpoint continuity: verify retained state availability and its compression/source
   dependencies, then implement explicit conversion/import for supported HE4 states.
   The worker accepts its own schema 3; historical state locators are metadata records,
   not demonstrated import/resume support. Never silently reset optimizer or exposure.
4. GPU training and recovery: intents currently reject non-CPU devices and the CLI
   masks CUDA. Implement the real device path, resource handling, interruption and exact
   resume, then exercise it on hardware within an assigned task. GPU-aware state fields
   alone do not qualify this path.
5. Native runtime: recover the full runtime04 source/build inputs and rebuild it.
   Load an export from this repo and compare target-only versus speculative token IDs,
   text and stopping behavior. Confirm finite logits and actual drafter activation.
6. Data capture: current code reads and authenticates existing captures; no public fresh
   capture workflow is present. Provide the supported capture path and verify taps,
   label offsets, end-of-generation handling, provenance and admission end to end.
7. Basic evaluation/reporting: demonstrate useful training readouts and export/inference
   correctness through normal entry points. Scientific/selection fields currently remain
   unknown or unimplemented. Keep workload selection, comparative statistics and the
   eventual serving-based best-checkpoint policy for the benchmark phase.
8. End-to-end acceptance and docs: exercise a small real capture/import -> train ->
   stop/resume -> export -> native verification workflow from documented commands.
   Check failure recovery as well as success. Replace temporary port instructions with
   current setup/use docs, preserve scoped evidence, and delete completed task files.

## Scope to settle when implementation reaches it

HE4 remains the development priority. Preserve historical alternatives and evidence.
The depth of legacy checkpoint conversion and support for other model families must
be explicit before that work expands. Training budgets, evaluation cohorts and stopping
policies are scientific choices, not defaults inferred from old campaigns. Historical
benchmark workloads remain replication references, not approved new requirements.
