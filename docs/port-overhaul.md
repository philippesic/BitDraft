# Agent first port overhaul

This is the proposed replacement port plan following the November 9–10, 2026 audit. Stage zero has a useful, passing foundation, but its present rules and future design do not yet support the source port coherently. The highest priority is to connect the checks to the real development and execution paths, remove contradictory constraints, and make scientific risk visible before an expensive run starts.

This document proposes work; it does not implement the overhaul or authorize training, cleanup, downloads, publication, or a benchmark design. Once Philip scopes an implementation task, agents should complete its included stages and exit checks autonomously. Stage labels are dependency milestones, not additional permission requests for already authorized work.

The initial product is the modern DFlash/HE4 family on the RTX 5080. Preserve selected reference identities and valuable retained artifacts without promising compatibility with every historical checkpoint. The repository remains local during porting. The old checkout remains a reference, not a runtime dependency.

## Decisions that constrain this plan

- Keep the complete ordinary-comment/docstring ban. Improve names, types, decomposition and APIs; put mathematical derivations and decisions in linked documentation. Necessary machine/legal lines retain exact exceptions.
- Keep one root AGENTS.md, the CLAUDE.md pointer, and the existing forbidden document-name components. Use this document for migration design, FINDINGS for conclusions, and generated reports for live execution state.
- Keep full checks at commits, paired failure/control specimens, required-test enforcement, strict typing, dependency analysis, locked environments and enforced ownership by default. Change individual mechanisms only where this audit identifies a concrete failure or unnecessary maintenance cost.
- Replace the exact repository file allowlist with forbidden-path/content lint plus a small required-entry inventory. A normal source file or document should not require editing checker source.
- Preserve one trainer and one benchmark entry. Optional skills stay late unless they unlock a concrete workflow; necessary run reporting and recovery are product infrastructure, not optional agent preferences.
- Do not inherit the old proposal's numerical training limits, stopping policy, serving cadence or workloads as approved requirements. Their correctness and scientific usefulness need separate design. The infrastructure must execute and validate a supplied policy.

## Audit evidence and limits

The audit read the foundation source, tests, policies, decisions, failure analysis, remaining-stage notes and the original port proposal. Opus 5.5 through Cursor at high effort independently inspected the foundation and selected old source. Existing uncommitted README/configuration edits and the two planning documents were stable inputs and remain separate work.

The ordinary command `uv run --locked --no-sync --offline python tools/check.py` passed all ten enforcing stages, 26 CPU tests, 125 intended rejections and 125 passing controls. One timed invocation took 24.68 seconds with peak RSS 198,700 KiB. This is one local observation, not a latency distribution or a future source-suite estimate. No GPU workload, native build, installation, capture transfer or cleanup was performed.

Independent in-memory probes against the same passing checker produced the following findings. These are coverage limits and design contradictions, not claims that the existing tests failed.

| Finding | Evidence | Consequence and proposed disposition |
| --- | --- | --- |
| Product cannot use the proposed scoring API | `tools/boundaries.py` rejects a trainer importing `benchmarks.api` with ARCH_DIRECT; the remaining-stage proposal requires that call | Keep the dependency boundary. Compose training and evaluation through a contract and an application-level adapter. |
| Operational time is blocked | A trainer calling `time.monotonic()` receives TIMING_OWNER; `time.monotonic_ns()` passes | Separate injected operational clocks from benchmark measurement. Close known alternate-call gaps without claiming a complete Python sandbox. |
| Shared preflight validators are in the wrong layer | Importing `tools.contracts` from the product receives ARCH_DIRECT | Extract domain contracts into a pure product package; tools and execution import the same implementations. Checker diagnostics stay in tools. |
| Write ownership depends on strings | `open("best_effort.log")` receives OWNER_PATH, while `os.replace(tmp, "best.pt")` passes | Enforce module/API boundaries and output-root ownership; replace substring matching. |
| Known unsafe import pattern is missed | `sys.path[:0] = [...]` passes the boundary checker | Add representative historical syntax to the regression corpus, including dynamic loading and import-time effects. |
| Ordinary native patch is rejected | A comment-free C++ change with hunk `@@ -10,1 +10,1 @@` receives COMMENT_PATCH | Replace the line-one-only patch assumption with per-file patch validation against a pinned base. |
| Future entry points cannot be used | The reserved `w1a1_eagle/__main__.py` main guard receives PUBLIC_ENTRY | Derive permitted entries from the same declared command surface checked by documentation. |
| Foundation shape is frozen in several places | `FOUNDATION_FILES`, source/output bans, exact ignore lines, active system-map flows and qualification strings | Replace foundation-specific shape with explicit supported stages and semantic rules. Do not merely delete all gates. |
| Metadata checks accept invalid semantic inputs | A capture label at position -1 passes `capture_offset`; Python indexing wraps it to token zero | Typed construction plus boundary/property tests must reject invalid ranges before indexing. |
| One correctness specimen excludes target-only execution | `compare_correctness` requires positive drafter proposals for every request | Separate generic output validity from role-specific mechanism activation. Future benchmark design supplies the role requirements. |
| Strictness meets an untyped numerical boundary | Torch module calls expose Any; product numerics do not yet exist here | Prove a tiny typed tensor integration before copying model code. Prefer typed facades and exact exceptions over blanket relaxation. |
| Local policy identity is not approval | Configuration hashes can be edited with their checker; `verification_report` correctly says REVIEW_REQUIRED | Retain hashes as drift/identity checks. Report policy changes and applicable task authorization; do not invent local authentication. |
| Rename left a stale installed hook path | `.git/hooks/pre-commit` names the former `binary-eagle-foundation` environment | Add setup/doctor and relocation checks; distinguish tracked hook configuration from an installed working hook. |

The reviewer also found gaps for `spec_from_file_location`/`exec_module`, import-time filesystem/CUDA effects and numpy/torch randomness. These should become small isolated regressions when their libraries enter the product. Do not require an entire old script to produce exactly one diagnostic: isolate the offending construct and its valid counterpart.

At inspection the WSL filesystem had approximately 3.0 GiB free and the foundation environment occupied 122 MiB. The current 2.5 GiB reserve leaves little headroom. This is enough for compact audit documents and the existing check, not evidence that a new torch environment, build or capture cache fits. No Windows backing-volume measurement or reclaimable-space audit was performed by the parent.

## Research and applicability

The closest verified match to the requested analogy is Camunda's [From Dark Factory to Software Garden](https://camunda.github.io/agentic-engineering-explainers/dark-factory.html): code changes remain as the ground future changes depend on, so the structure needs tending. Its [enforcement ladder](https://camunda.github.io/agentic-engineering-explainers/enforce.html) prioritizes types, lints and architecture checks over remembered instructions. Applied here, the first repair is a coherent API and state model, followed by checks that exercise it. More prose prohibitions are the last resort.

Camunda's [agent-user design](https://camunda.github.io/agentic-engineering-explainers/agent-users.html) also supports structured outputs, explicit errors, dry runs and derived help. For this project, one execution command should expose a bounded JSON view and a concise human explanation of the same record, rather than making agents parse logs to infer whether a run is valid.

OpenAI's [harness engineering account](https://openai.com/index/harness-engineering/) supports a short instruction map, repository-local knowledge, executable architecture rules and recurring documentation maintenance. Adopt those mechanisms, not its entire document layout or large-team workflow. Its experience is a case report, not proof that this repository will achieve a particular productivity gain.

Anthropic's [long-running harness account](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents) emphasizes recoverable incremental work and explicit verification across sessions. Its [tool-design guidance](https://www.anthropic.com/research/writing-tools-for-agents) favors meaningful, economical responses and evaluation on real tasks. Here that means durable run state, actionable refusal messages, and end-to-end failure exercises, not another mandatory agent role.

Cursor's [scaling account](https://cursor.com/blog/scaling-agents) reports both coordination failures and gains from removing unnecessary orchestration. Its [follow-up](https://cursor.com/blog/self-driving-codebases) concerns very large agent populations; this project's occasional parallel tasks do not justify recreating that system. Lauren Tan's original [workshop page](https://maven.com/p/e23d9c/how-cursor-turned-ai-agents-into-better-engineers) identifies verification, refactoring and a Dune architecture section at 41:00. The public page establishes the reference; this audit does not claim access to Dune's implementation or an independently verified transcript.

For disk handling, systemd's [temporary-directory guidance](https://github.com/systemd/systemd/blob/main/docs/TEMPORARY_DIRECTORIES.md) distinguishes bounded temporary data from persistent state and describes protecting active temporary trees with locks. These principles support owned scratch cleanup. They do not make old scientific checkpoints safe to delete based on age. A reclaim skill is an interface to an explicit retention policy, not its replacement.

## What to add

1. Shared typed domain records and validators used by both check and execution: identities, provenance, roles, data admission, resource estimates, task/run intent, lifecycle transitions and evidence references.
2. A small lifecycle controller with injected operational clock, explicit process ownership, durable transitions, bounded retries and recovery. It coordinates existing components without becoming a generic workflow engine.
3. Preflight and report paths that explain the decision, dataset size, planned reuse, initialization identity, uncertainty, costs and stopping conditions in plain language as well as structured data.
4. Integration and failure-injection tests that exercise real owners, storage, readers, subprocesses and recovery. Keep pure synthetic checks, but make their scope explicit.
5. A reproducible selected-source inventory and disposition record inside this repository. Preserve compact port evidence currently found only in external scratch; do not import raw campaign trees.
6. A setup/doctor path that checks entry points, hook installation, environment identity, capacity and supported capabilities without silently installing or running hardware work.
7. A bounded disk inventory and reclaim proposal interface, with optional skill/delegate use after the deterministic ownership and retention checks exist.
8. Freshness checks for public commands, supported capabilities, evidence locators and report schemas; a small task-based acceptance exercise for agent usability.

## What to remove

1. The exact authored-file allowlist and duplicated literal rule/case totals. Keep a required critical-file set and a checked generated policy lock.
2. Substring-based writer ownership, impossible future-entry rules, and stage-zero bans that prohibit the very source directories the port requires.
3. Product imports from tools or benchmarks. Remove the plan's assumption that the trainer directly calls a benchmark implementation.
4. Universal constants copied from proposed experiments into infrastructure. Examples may remain clearly marked fixtures, never silent defaults for a real run.
5. Obligatory support for inactive legacy trainers, model families and optimizer schemas. Preserve their evidence and artifact dependencies without importing their execution paths.
6. Repeated approval stops inside an already scoped task, duplicated live-state prose, and an unqualified overall “healthy” message.
7. Claims that a green synthetic suite establishes source parity, native correctness, effective monitoring or scientific sufficiency.

## What to adapt

| Existing mechanism | Retained purpose | Adaptation |
| --- | --- | --- |
| One full check | Reliable commit gate | Keep it; add focused developer invocations that explicitly cannot issue a full qualification receipt. Measure stage costs before caching or splitting. |
| Paired specimens | Prove real rejection and success | Retain pairs; add real historical syntax and malformed-domain cases. Preserve independent oracle/outer tests so generated manifests do not validate themselves. |
| Rule/test rosters and hashes | Prevent silent loss and establish identity | One authored rule registry, generated checked lock/counts, discoverable tests plus mandatory critical scenarios. Changes show a compact policy diff. |
| No required test skips | Prevent false green | Keep within each declared capability suite. Unsupported GPU capability is reported unavailable, never counted as a passing/skipped required CPU test. |
| Strict typing | Reject invalid composition | Keep strict core contracts; narrowly isolate unavoidable numerical-library typing holes with typed interfaces and runtime shape/dtype checks. |
| Architecture checks | Keep seams and ownership clear | Derive rules from one policy; cover supported import syntax; test representative forbidden calls. State that static Python checks are not a hostile-code sandbox. |
| Comment ban | Keep source self-explanatory | Preserve it. Validate multi-file and noninitial patch hunks, strings, new files and necessary legal directives against pinned post-images. Do not mistake unchanged upstream context for newly authored comments. |
| Documentation | Fast navigation and truthful claims | Keep FINDINGS and decisions; derive command/capability details. Keep deliberate design prose without banning ordinary headings through another heuristic lint. |
| Full environment pins | Reproducible checks and execution | Keep a small offline check environment and separately locked numerical/native profiles. Verify actual loaded dependencies; never force torch into every lint-only run. |
| Approval receipt | Expose safeguard changes | Preserve REVIEW_REQUIRED where genuinely applicable; bind validation to exact content and task scope. A local ref is a comparison baseline, not authentication. |
| Optional skills | Reusable procedural judgment | Add only around working commands and evaluate with realistic cases. Avoid a permanent multi-agent babysitting hierarchy. |

## Target source architecture

Names below express responsibilities, not a demand to create empty packages. Keep the existing `w1a1_eagle` namespace initially unless an implementation task deliberately includes a rename. Prefer small cohesive modules over one module per abstract concept.

| Layer | Owns | Dependency direction |
| --- | --- | --- |
| Domain contracts | Validated immutable identities, roles, intent, artifact/result references and transitions | Standard library only; no tools, torch, process launch or I/O |
| Configuration and operational clock | Explicit configuration inputs, monotonic deadlines, injectable time | Domain contracts; environment access only at configuration boundary |
| Data | Ledger admission and consumption, bounded capture/shard/window readers | Contracts; tensor decoding isolated from pure provenance logic |
| Numerics and model | HE4 construction, loss, quantization, typed tensor transformations | Contracts and tensor library; no orchestration, filesystem or benchmark imports |
| Training worker | One fresh/resume/branch update loop and immutable snapshots | Data/model interfaces, supplied clock/device interfaces; no serving subprocess management |
| Artifact store | Transactional states, exports, retention references and pointer publication | Contracts and storage; never mutates live optimizer tensors |
| Native adapter | Pinned build/load/export/capture integration | Contracts and explicit external paths; no model selection policy |
| Lifecycle | Segment scheduling, worker ownership, policy transitions and reporting | Core interfaces plus injected evaluator/process adapter; no benchmark import |
| Application composition | Wire workers, stores and evaluator adapters at `w1a1_eagle.__main__` | Product interfaces; the process adapter invokes the declared benchmark command without importing benchmarks |
| Benchmarks | Implementation of the separately designed evaluation contract and result storage | Contracts and native interfaces; no trainer dependency |
| Product diagnostics | Doctor probes and typed readiness reports | Explicit product/configuration interfaces; no tools import |
| Tools | Static checks and specimen runner | May consume pure product diagnostic results/contracts; product never imports tools |

The composition root at `w1a1_eagle.__main__` supplies an evaluator implementing a narrow protocol. Its process adapter launches the declared benchmark entry with an argument vector and reads verified results through domain contracts; it never imports `benchmarks`. Keep the namespace-wide forbidden import edge. The GPU serving adapter runs in a separate process to release training allocations and isolate failures. Both CLI and lifecycle reach the same benchmark execution implementation through that adapter. This preserves one comparator without requiring in-process GPU coexistence or hardcoding shell commands throughout training. Phase A needs a passing specimen for this composition and a rejecting specimen for a direct benchmark import.

The public command surface is the existing full check, a doctor mode, the lifecycle run command, and the benchmark entry. Doctor and lifecycle are modes of `python -m w1a1_eagle`; benchmarks retains `python -m benchmarks`. Training worker and other necessary subprocess entries are explicitly private, registered and launched only by the owning adapters. They do not create alternative user-facing trainers or comparators. Help, entry permissions and the system map derive from this declaration.

Operational clock values express deadlines and progress. Measurement records carry separately typed, declared timing scopes supplied by the eventual benchmark implementation. A small device adapter owns CUDA synchronization/events where needed. A heartbeat must not depend on importing the benchmark package or initializing CUDA.

Constructors validate untrusted JSON into typed domain values. Do not pass dictionaries with booleans such as `complete=true` and `durable=true` directly to selection. A verified-result reference and a committed-artifact reference must come from their validating owners. Unknown schema, missing fields, invalid ranges, nonfinite numbers and incompatible identities fail at entry.

Single ownership applies to each authoritative state/result, not all filesystem I/O: read-only diagnostics can inspect records, and scratch writers have bounded owned roots. Architecture, types and APIs establish invariants before lints and instructions reinforce them. Ownership is enforced in layers: narrow storage/process APIs, import restrictions, focused checks for raw effects, and integration tests that reconcile expected outputs. An output manifest can detect undeclared files under an owned root; it cannot prove nothing was written elsewhere. Add OS-level isolation only if an actual workflow needs it, not as a claim about the AST checker.

## Source extraction and compatibility

Use the original proposal's inventory and P0–P7 parity responsibilities as source material, not as proof of acceptance. The original proposal lives at `/home/philip/.t3/scratch/2026-11-10-read-only-binary-eagle-porting-bc9c678e/PORTING_PLAN.md`; its compact relevant inventory and derivations should be migrated into tracked evidence during implementation. The pre-existing uncommitted `docs/remaining-stages.md` and `docs/issue-analysis.md`, plus the tracked [data requirements](data.md), [training requirements](training.md) and [runtime requirements](runtime.md), remain historical inputs to this replacement proposal until its changes are approved. This audit checkpoint does not silently absorb the two uncommitted input documents.

| Extraction | Reference responsibility | Required evidence before replacement |
| --- | --- | --- |
| Pure contracts and ledger | Existing foundation validators; old capture grouping/catalog/role code | Two unrelated null topics remain distinct; aliases preserve protected roles; one accounted disposition per input; unknown ancestry remains unknown; malformed positions and schemas reject |
| Reader and sampler | Sharded/window readers and selected modern capture contract | Bounded memory/cache bytes, identical windows across shard boundaries, truncation/hash/shape rejection, deterministic sampler recovery and exact position accounting |
| HE4 numerical core | Selected modern component route, original initializer, quantizers and loss | Independent scalar/tiny tensor oracles for forward, gradients, update, sign/packing/rounding, shared tensors, dtype and trainable groups |
| Trainer | Existing campaign loops as characterization references | One configured implementation; explicit fresh/resume/branch semantics; RNG/sampler/exposure restoration; no source-text execution or copied campaign globals |
| Store and export | Recovery/aliasing regressions and selected archive/export codecs | Immutable snapshots, atomic publication, dependency closure, checksums and exact next-update recovery; failed export cannot damage prior state |
| Native closure | Runtime04 source, ordered patches, build receipts and loaded libraries | Rebuild outside the frozen checkout without ignored old-run object dependencies; documented source/license/toolchain closure and later hardware parity |

Declare supported reader/schema versions explicitly. An intentional migration creates a new validated artifact with source lineage and preserves the original; unknown versions or producer/consumer disagreement refuse admission. Choose only the state schemas needed for the named starting point and parity references. Unsupported historical schemas produce an explicit refusal with preservation instructions. No automatic rewriting or deletion of original archives. A model refactor that changes numerical behavior is a new research change, not successful parity; separate it from the structural port checkpoint.

Keep one illustrative extension for each required adapter seam. Do not build a plugin marketplace or support matrix for hypothetical backends. A second model/backend is the trigger to generalize further, with a compatibility test and a stated need.

## Benchmark interface without benchmark design

The benchmark redesign is a separate research deliverable. This plan specifies the execution boundary, not workloads, prompt sets, repetitions, metrics, thresholds, ranking rules, evaluation cadence or statistical tests. Training scientific policies are likewise supplied inputs rather than infrastructure defaults.

Reserve the existing public modes `plan`, `run`, `verify`, `report` and `qualify` until the redesign decides their precise semantics. Avoid renaming or merging them merely to simplify this audit. Internally require these capabilities:

- Planning validates a versioned specification and returns its identity, supported capabilities, required artifacts, resource estimate and refusals without executing the workload.
- Execution accepts an admitted intent and artifact references, records every attempt, and emits bounded progress plus a terminal outcome. Retries retain prior attempts and consume the declared budget.
- Verification checks evidence integrity, declared compatibility and completion against that specification. It does not trust caller-supplied success flags.
- Reporting reads records only. Missing observations remain unknown; reporting cannot rerun work to fill gaps or silently select favorable attempts.
- Qualification names the precise tested capability and source/environment identities. A foundation check cannot certify native execution or serving performance.

The result envelope contains schema version, intent/source/environment identities, input artifact identities, attempt and parent identity, declared measurement definitions and units, evidence locators, outcome/reason, validity and verification references, and resource accounting. The redesigned benchmark owns its metric payload and comparison policy. Separate target-only and drafter activation requirements rather than forcing all execution roles through one positive-proposal rule.

The lifecycle consumes a verified terminal result through the protocol. Updating a selected-checkpoint pointer additionally requires a durable artifact and a supplied selection-policy decision. Interrupted, invalid, stale or incompatible results cannot replace a valid selection. Generic simulated scores test the transaction; they do not define the real scoring policy.

## Bounded autonomy and clear communication

Authority, readiness, evidence validity, selection and scientific usefulness are separate facts. Authorization cannot make corrupt or insufficient evidence valid; valid evidence cannot authorize execution. Editable hashes and approval fields grant neither authority nor independent approval. An authorized task needs an objective, completion condition, allowed operations, resource envelope, and reasons to stop or ask. A run intent adds initialization/state identity, admitted data identity and denominators, planned exposure, policy identifiers, cost estimates, uncertainty, recovery policy and notification destination. Capture task authorization once and carry it through the task; a later scope expansion requires a new decision.

Admission and terminal reports expose the same facts in human and machine form, including the intended mechanism and actual activation witness (or explicit absence), starting identity, limiting risk, evidence scope, budget and stop action. Preflight renders the facts that matter before launch: what decision the run informs; what is initialized or resumed; how much distinct data exists; how often it is expected to be reused; what can invalidate the result; what it will cost; and what will stop it. Values are computed from admitted manifests and the supplied schedule, with assumptions and unknowns visible. A fixed one-million-position floor cannot prove scientific adequacy across all experiments.

Examples of required report distinctions:

- Operational: whether updates advance, input is stalled, capacity is adequate and owned processes are responsive.
- Scientific: whether the configured data, exposure, validity and learning-quality conditions hold. A missing or stale evaluation is unknown, not healthy.
- Objective: whether the latest valid evidence supports the task's goal, with the exact comparison scope and age. A proxy improvement is not automatically objective progress.

The headline leads with the most consequential issue and its action. For example: “Updates are running, but the learning-quality stop condition fired; the run stopped at a complete update and retained the last recoverable state.” Explain cause as confirmed, supported hypothesis, or unresolved. A validated initialization and a worsening trajectory are separate facts; neither proves latency is wrong or that initialization caused the decline.

The old [F004](../FINDINGS.md#f004) already states that the hour-four development peak does not establish the best formal serving checkpoint. Correct the failure-analysis wording during reconciliation: the demonstrated gap is insufficient information to identify the best intermediate serving state, not proof that a superior serving state was discarded. Preserve the negative result and uncertainty without inventing a new training diagnosis.

Persistent monitoring belongs in a small process/service or timer, not an agent burning tokens in a polling loop. It emits transition events and bounded periodic summaries; a wakeup adapter uses the harness's own delivery channel to resume an agent only when judgment or a handoff is needed. An external webhook, if later required, uses that service's authentication rather than an invented local approval token. The run's budget and stop rules remain effective if the agent or notification service is unavailable. Test missing/duplicate/delayed notifications, restart after host/harness interruption, and stale owner identity. Do not automatically retrain or repeatedly restart without a declared retry budget and reason. Count failed attempts against cumulative cost and retry limits even when notifications are lost. Recovery within the authorized bounds proceeds autonomously; changed semantics or reduced capabilities are explicit decisions, never quiet fallbacks. Bind stop/cancel/retry controls to the owner and attempt; reject stale, foreign and replayed controls through the existing channel access boundary. Authenticated channel access proves neither policy approval nor tenant isolation.

Use a worktree for concurrent edits that conflict with pinned source. Give each task an output root and owner token; identify processes with host/boot/process-birth identity rather than PID alone. Bind immutable attempts to dirty source content, generated executable inputs and actual loaded binaries/dependencies, not just HEAD. Drift invalidates the affected qualification. Never kill another task by process-name matching.

Coordinate exclusive GPU use through an existing external owner or a narrowly permitted nonblocking busy-refusal primitive in a shared host/device namespace used by participating projects. A repository-local lock cannot exclude another project. Check ambient occupancy; uncooperative or unexplained work remains uncertainty requiring refusal or escalation, not a claim of exclusivity. No scheduler, queue or lease service is mandated. This is a future design obligation, not authorization to install GPU locking during the current foundation stage.

## Recovery and storage

Write a complete state to a temporary location on the destination filesystem, validate it, make the data durable as required by that filesystem, then atomically publish its manifest/pointer. Keep latest recoverable state and policy-selected state distinct. Reference dependencies such as compression bases and dictionaries explicitly. Recovery selects a complete committed state and reconciles subsequent partial attempts; it does not infer completion from the newest filename.

Acceptance exercises inject failure during an update, snapshot serialization, archive write, export, pointer publication, evaluation and notification. Assert that prior complete state remains readable, incomplete work cannot become selected, restored next updates agree with the uninterrupted tiny reference under declared determinism, and cumulative work/attempt accounting is truthful. A process-kill test proves process-crash handling; it does not prove power-loss durability. Test the actual atomic-write protocol separately and document filesystem assumptions.

Disk preflight accounts for environments, caches, simultaneous old/new checkpoints, exports, logs, build scratch and backing-volume capacity. Recheck capacity during long operations and stop cooperatively before an unrecoverable write. Account for declared concurrent work through the existing execution owner and recheck observed headroom before writes. Estimates and advisory ownership do not reserve disk space at the OS level; do not create a capacity reservation service. Unknown backing-volume availability is an explicit unsupported preflight state.

A disk skill may invoke a deterministic report and propose reclamation. Optional delegation helps inspect a large inventory, but does not grant broader deletion authority. Classify candidates as owned disposable scratch, reproducible cache, retained scientific artifact, or unknown/shared state. Only the first two may be automatically reclaimed within an existing authorization and after checking active users and offline dependencies. Scientific artifacts need retention/dependency and recoverability evidence; unknown/shared state is not reclaimed by default. No autonomous deletion service, arbitrary shared-cache cleanup or automatic loss of scientific evidence is permitted.

Revalidate identities and active ownership immediately before deletion. Stay within declared roots, reject symlink escapes, retain a deletion receipt and verify reclaimed capacity. Quarantine/rename is optional where useful but does not itself free disk. Prefer prevention and explicit external artifact placement to repeated emergency cleanup. Do not promise that a skill can recover enough space on this host; capacity placement remains unresolved.

## Checks and maintenance that stay out of the way

Keep the ordinary full gate at commits. Hosted CI and PR workflows are optional future choices, not port or acceptance prerequisites; if adopted, they run the same gate. Add documented focused checks for editing, with output that says their qualification scope. Report per-stage elapsed time and actionable diagnostics: rule, file/symbol, actual problem, expected contract and the relevant fix or decision link. Setup failures should identify the missing pin/tool rather than only a generic SETUP_ERROR. Bound output and offer structured diagnostics.

Consolidate rule definitions and generated inventories without removing independent tests. A generated lock can detect accidental drift but is not independent approval. Test omission, disabled stages, malformed fixtures and unexpected diagnostics from outside the rule implementation. Cache only after measurement shows a bottleneck; key by exact relevant source/toolchain/configuration and validate invalidation before relying on it. Always retain an uncached acceptance path. Report actual executed versus reused checks and their source/environment identities. Cached CPU proofs are labeled as such; never memoize hardware performance measurements or present historical hardware qualification as a fresh observation.

Oracles must be independently derived: a second agent or a fixture generated by the implementation does not establish independence. Combine manually derivable tiny cases, historical negative cases and producer/consumer conformance across Python/native boundaries. Unknown schemas or disagreement reject qualification. A fake CPU construction names exactly what it proves; safety-critical runtime admission and stop checks use explicit validation that remains active under optimized execution, not removable assertions.

The first source increment must prove that a normal new file, test and document can be added without editing checker implementation. Conversely, adding forbidden outputs, stale instructions, an unsafe import or an unregistered exception must still fail. Expand tests based on demonstrated failure classes rather than targeting a larger rule count.

Keep README as the stable front door, AGENTS as a short map of working constraints, the system map as capability navigation, decisions for rationale, and FINDINGS as the conclusion index. Generated run views are disposable projections of durable records. Fix stale claims when changing their owner; do not introduce a new heading blacklist or a second narrative chronology. Compact authored migration evidence belongs in Git, not only in scratch or an ignored run directory.

Agent usability is an acceptance property: give a fresh agent a routine source change and a deliberately failed simulated run. It should find the owner, use focused checks, pass the full gate, and explain the run's operational/scientific/objective outcome with correct evidence and next action. Judge correctness and avoidable clarification, not token count alone. Repeat this exercise only when interfaces or recurring failures justify it; it is not a permanent expensive agent swarm.

## Implementation sequence and exit evidence

These phases describe dependency order. Philip can authorize several together. Every phase ends with a coherent local commit, evidence and updated supported-capability claims; remote setup and pushing wait for the later remote decision.

| Phase | Deliverable | Exit evidence |
| --- | --- | --- |
| A Foundation evolution | Update scoped authorization in AGENTS first; replace file allowlist; consolidate policy data; fix entry/clock/patch rules; move domain contracts; improve diagnostics and doctor | Existing suite except explicitly re-derived specimens, plus reproduced audit cases; normal file addition works; forbidden files still reject; relocated hook works; composition pair passes/rejects correctly; no product import from tools; policy diff lists each semantic correction |
| B Minimal executable spine | Typed intent, injected clock, tiny worker/evaluator stand-ins, lifecycle transitions and transactional artifact references | Synthetic end-to-end start, stop, crash, recovery and report; no CUDA/data downloads; malformed intent and stale receipt refuse; authority/readiness/validity remain distinct; foreign/replayed controls reject |
| C Data and numerical port | Selected ledger/readers, HE4 constructor/loss/quantization/init/export, typed tensor boundary | Metadata reconciliation; bounded reader proofs; independent numerical/gradient/packing references; no frozen-checkout runtime dependency; chosen reader/schema matrix explicit; migration preserves originals; cross-boundary disagreement refuses |
| D Training and durable operations | One fresh/resume/branch worker, real snapshot/store integration, process supervision and capacity accounting | Tiny CPU next-update equivalence and alias checks; failure injection at publication boundaries; monitor survives agent/notification absence; failed retries consume cumulative budgets; optimized execution retains runtime guards; user-facing risk report exercised on simulated and retained real trajectories under a supplied policy |
| E Benchmark and native integration | Implement the separately approved benchmark design behind the agreed interface; establish clean native build closure | Offline contract conformance, mismatched/stale/partial result refusal, artifact relocation and cleanup tests; dirty/generated/loaded identity drift invalidates evidence; cached proof versus fresh execution distinguished; benchmark scientific acceptance supplied by its separate design |
| F Hardware qualification and cutover | Authorized bounded 5080 parity/recovery/qualification, selected state compatibility, docs reconciliation | Actual device evidence bound to source/build/input identities; no mock-only parity claim; retained dependencies preserved; old execution route retired only after replacement acceptance |
| G Optional skills and remote | Validated run-review/recovery/disk skill interfaces; remote and CI setup when requested | Realistic skill cases including refusal and recovery; hosted checks and approval wiring actually exercised before claiming protection |

Phase A's first implementation commit updates AGENTS to reflect the authorized task: included phases proceed through their exit checks without renewed permission; every safeguard change appears in the handoff policy diff; weakening outside scope or merely to obtain a pass requires a decision. Retain the present 1 GiB stage write cap and 2.5 GiB WSL reserve until an authorized replacement resource envelope is recorded; do not silently carry those numbers into a larger run or silently drop them.

Re-derive questionable specimens from explicit reference semantics when extracting contracts. List corrections in the policy diff: negative capture positions; role-specific activation; parent exposure versus advancing sampler position; and the distinction between accounting for an observed historical bug exclusion and admitting a healthy dataset. A terminal reason of `bug` can describe historical accounting without making the dataset eligible for training. Passing an old control is not evidence that its semantics are correct. Real-trajectory replay tests policy execution and report fidelity without selecting new scientific thresholds or asserting that a noisy development peak is a serving optimum.

Resolve disk capacity before any phase needs a new large environment/build/artifact. A–B can advance with the current small environment, subject to observed reserve. Phase C needs an explicit numerical-environment capacity plan. License/source rights must be resolved before copying source that the current stage policy does not authorize. These dependencies do not justify stopping unrelated pure-contract work.

For cutover, complete a selected-source disposition table: ported and verified, intentionally retired, historical evidence only, or blocked with a reason. Reconcile the older remaining-stage proposal, source docs and system map so new agents do not receive two competing plans. Before removing obsolete-looking modules, settings, overrides or keys, record their actual behavior, consumers and state/safety dependencies in that disposition. Controls share applicable correctness guards with candidates; do not silently port a broken baseline or alter treatment semantics to obtain parity. Keep historical measured evidence immutable; supersede conclusions by reference rather than rewriting the measurements. Retired records may remain accessible but must no longer function as active instructions; each current document has a responsible capability owner and explicit supersession links.

## Implemented source boundary

The authorized pre-benchmark increment has implemented foundation evolution, the CPU
execution spine, selected modern metadata/capture intake, HE4 numerics and streaming
GGUF export, own-schema fresh/resume/branch states and bounded durable CPU operations.
[Execution](execution.md) owns the supported reader/runtime claims;
[checks](checks.md) owns verification and fresh-agent exercise scope; and
[findings](../FINDINGS.md) retains scoped results and failures. The source disposition
is in [the inventory](evidence/source-inventory.json). The older remaining-stage and
issue-analysis files are preserved input proposals, not competing execution authority.

CPU end-to-end evidence does not close E/F: actual native build/load/parity and GPU
cutover depend on their separate qualification and capacity/allocation gates. Benchmark
methodology, evaluator/selection integration and scientific stopping policy remain
with E; optional high-level operations and remote/hosted setup remain with G. No
scientific thresholds, universal old-checkpoint compatibility or serving-performance
claim was introduced to close the CPU stages. Retained originals and the old execution
route remain protected until replacement acceptance.

## Decisions still needed before dependent work

1. Artifact/environment placement, backing-volume reserve and retention scope. This audit authorizes no reclamation.
2. Exact current HE4 initialization and selected checkpoint schemas required for the first port. Older retained artifacts can remain protected without runnable compatibility.
3. Separate benchmark/training-policy design: useful objective, data adequacy, exposure and stopping rules, selection method, validation workload and hardware budget. No such values are set here.
4. Source license/provenance approval where required, then future remote visibility and approval wiring. Local-only work remains the current policy.

## Debate disposition

Two substantive reviews used Opus 5.5 through Cursor at high effort. The first agreed on the major architecture and checker failures. Parent reproduction confirmed the import, clock, writer, patch and entry-point examples and added negative capture-offset and target-only validity cases. In the second round, Opus accepted all six parent objections and found no blocking contradiction after four clarifications, now incorporated above. There is substantial agreement on the add/remove/adapt list; another debate round would not change the implementation direction.

The resolved disagreements are: preserve a subprocess adapter behind an injected evaluator instead of hardcoding CLI orchestration throughout lifecycle; use policy diffs and scoped authority instead of local-ref authentication; use documentation ownership rather than a new heading blacklist; leave scientific adequacy and thresholds to the separate policy design; leave metric decomposition and benchmark-mode changes out of this interface plan; and permit optional disk delegation only over deterministic ownership and retention checks. The parent does not adopt the stronger claim that no safe reclaim can help this host; the available evidence only fails to establish sufficient reclaimable capacity.

The review's four final clarifications made composition placement and its import rule explicit, moved the AGENTS authorization update to the first implementation commit, allowed documented correction of flawed specimens, and declared public/private command entries. The subsequent cross-project agreement replaces the overly broad lease wording with shared host/device busy-refusal semantics and observed disk headroom. Wakeups use the harness delivery channel; policy/report testing includes a retained trajectory. Numerical claims about the next run or the significance of a development peak are not accepted findings.

The November 10 cross-project synthesis with ShaderLOD and the cache project adopts the applicable obligations in the existing sections and phase gates above. Local full-commit checks, instruction/comment restrictions, filenames and HE4 support scope remain unchanged. No common framework, additional reviewers, benchmark methodology, runtime implementation or hosted-CI requirement is introduced. The plan amendment used Philip’s explicit task-only exception of at most 16 MiB additional disk below the normal reserve; that exception does not authorize subsequent implementation or cleanup.
