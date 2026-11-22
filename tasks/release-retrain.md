# Release retrain plan

## Goal

Retrain the release drafter on licence-clean, broader and much larger TRAIN data. Training starts
fresh from the calibrated conversion of the public Baseline weights, as before. Judge the result on
a frozen public benchmark gate, against Baseline both as released and fine-tuned on the same data.
Then publish a release whose claims, data card and licences hold up to outside scrutiny.

## Authorized

Stage 1 (Prep) authorized by the owner on 2026-11-21. Extended 2026-11-22: "go thru as much as
you can before rental", with builds done in this thread rather than delegated. Authorized: all of
Stage 1, all of Stage 2 (local screening), and the local parts of Stage 3 that precede the rental
(one local end-to-end chunk, rental scripts). Not authorized: any rental, including the one-hour
5090 test; production generation of the full corpus; the final runs. The agent proposes gate
margins and freezes them provisionally; the owner approves them at the rental checkpoint, before
any final run.

## Done when

- The owner has decided the open questions under `State`.
- The rights audit, data card, public harness and release gate are frozen before TRAIN is acquired.
- The final run and every comparison arm are measured, closed through close-results and recorded
  in FINDINGS.
- The release ships with a licence, model card, data card, pinned runtime and reproducible eval
  commands, and an outside checkout reproduces one published result.

## State

Owner: release-retrain planning thread. Branch: `main`. Stage 1 in progress since 2026-11-21.
Owner decisions 2026-11-21: Q4KM is the training target, and the frozen public gate also runs on
NVFP4 in Stage 4 as a measured secondary result. The first release is fork-only, with an upstream
llama.cpp PR possibly later. GPT-6.2 Astra reviewed
the plan on 2026-11-21. Its accepted corrections are folded in below.

### Stage 1 progress (2026-11-22)

- **Public gate built:** `artifacts/benchmarks/release-gate-01/gate-v1` from `prepare.json`
  (seed 9201). 1,922 tasks: general 880, thinking-on 320, sampled 242, long-input 180,
  agentic-replay 60 (60 reserved repositories), tool-calls 120, held-out ReAct 120. SPEED-Bench
  masked prompts resolved (0 masked). Comparisons accept these files with `tier: gate`.
- **Gate port qualified on two files** (`release-gate-01/qualification/`, runtime v10, depth 3,
  Q4KM): every arm's outputs identical to target-only. Tool calls (120): Baseline 63.6% acceptance,
  leader 87.3%, leader 1.28× Baseline. Agentic replay (60): Baseline 51.5%, leader 64.8%, 1.06×
  Baseline. The agentic file is from `gate-v2`, which cuts histories at 6K–24K rendered tokens as
  the definition says (`gate-v1` used a character proxy, up to 37.8K tokens, and is superseded;
  its other files and decontamination outputs are byte-identical).
- **Prompt bank:** pilot `artifacts/bank/retrain-01/pilot-v1` (3,428 rows) is good. The full bank
  (`bank.json`, seed 9301, 3% DEV, agentic DEV split by repository) first gave 88,146 TRAIN and
  2,664 DEV; long agentic sessions, PG-19 books and tool rows ran short. Abundant cells are topped
  up by 12K (`bank-v1` rebuilding).
  Source revisions are in `artifacts/prompt-sources/retrain-01/revisions.txt`. Agentic quota
  is 5,000 (6,306 SWE-rebench issues in 1,823 repositories); tools take the other 2,000.
- **Local chunk end to end (pipeline only, not a learning measurement):** 64 pilot rows,
  `artifacts/datasets/retrain-01/`. `write-bank` at 16 slots gave 430–510 generated tokens/s
  with 32-row shards (tail idle; production uses larger shards); `replay-tokens` 149 s
  (4.0 GB raw, about 41 KB per token); fusing 34 s (511 MB, about 8 MB per record, 13 GB RAM);
  200 training updates in 127 s with no refusals.
- **Fixed on the way:** parquet now streams in batches (the 2 GB SWE-rebench file had used
  24 GB); the Hermes tools block parser; gate agentic histories no longer shingle-match every
  trajectory (they are decontaminated by repository); bounded candidate pools.
- **Staged driver works:** `staged-train`, two 32-row stages of 100 updates
  (`artifacts/datasets/retrain-01/staged-smoke`), resumed after a fix with the finished stage
  skipped; continuation drops the fresh-only calibration setting.
- **Full bank done:** `bank-v1`, 100,146 TRAIN and 3,024 DEV; 19,629 thinking, 25,679 sampled;
  tool conventions native 90,035, JSON 6,522, Markdown 6,613.
- **Private DEV rebuilt** (`artifacts/benchmarks/retrain-dev-01/`, from `bank-v1` DEV rows,
  one per group, 11 scenarios including multilingual and each tool convention): `screen` 96
  requests (caps 128/256) and `panel` 292 (caps 512/1024), runtime v10, depth 3, Q4KM. Target,
  Baseline and leader (284 MB file) references are registered; `screen-candidate.json` and
  `panel-candidate.json` check a new state against them. Panel, leader against Baseline:
  acceptance 47.6% against 58.2%, speed 0.994×. By scenario the leader wins only native tool
  calls (1.23×) and loses multilingual (0.86×) and chat-writing (0.97×); others 1.01–1.07×.
- **Stage 2, batched generation decided by measurement (2026-11-22):** on the 50 greedy rows of
  the 64-row test, replay found the stream token differing from the target's exact choice at
  1.71% of positions for 16-slot batched answers and 1.68% for one-slot server answers (5 of 50
  answers identical between them). The drift is the server path against replay, not batching.
  One slot gave 190–196 generated tokens/s, 16 slots 430–510 (short shards). Labels stay exact;
  the 1.7% near-tie histories are far closer to greedy than the 25% sampled rows already in the
  recipe, and two-seed training arms could not resolve an effect that small, so the training-arm
  pilot is replaced by this measurement and batched generation is adopted. Owner reviews this
  at the rental checkpoint.
- **Preflight: scaling mechanism check (2026-11-22).**
  - Question: at a fixed short budget, does more distinct new-mix data beat repeated data, for
    one-bit and for the 4-bit control? It sets whether, and how long, the spending check runs.
  - Decision-changing result: one-bit 256-group arm above its 64-group arm on the DEV screen by
    more than the measured seed spread (second seed of the one-bit 256 arm) means data
    distinctness already matters at this length; the spending check then runs at the planned
    length. Otherwise the result is inconclusive by design and the spending check sets its own
    length from the learning curve (about +1.8 points per doubling).
  - Preconditions and checks: at 2,370 updates and about 12.5 updates per record per pass, the
    64-group arm makes about 3 passes and the 256-group arm about 0.74 (training reports log
    unique positions and visits; the small arm must show mean visits of at least 2). Controls
    differ only in data: same start, seed, order rule, schedule (40 warmup, cosine to 2,370).
  - Cost: data about 25 minutes (256 batched answers, replay, fuse; the 64 subset replayed from
    the same streams), training 5 arms of about 12 minutes, screens about 2 minutes each: about
    1.5 GPU-hours.
  - Kill: any refusal or nonfinite loss; a training arm over 20 minutes; generation over 30.
- **Pending in Stage 1:** owner approval of the frozen gate and source list (at the rental
  checkpoint, with the margins).

### What the previous train taught us

- **More distinct data is the leading hypothesis, but it is not isolated.** S4 flattened after
  about two passes and then mostly memorized (F025). The flattening coincided with the cosine tail,
  and no control isolated data. The long fused run changed size, mix, storage and length together.
  It was still gaining about 0.3 points per segment at the end (F033). TRAIN is 13,394 records.
  DFlash used about 800K samples and DFlare 2.4M.
- **The data, not one bit, carried the speed result.** A 4-bit control trained for 66 minutes
  matched the leader's speed (F034). Its acceptance on ordinary requests was flat between 2,000 and
  21,000 updates. It kept gaining only on agentic sessions, from 57% to 69% first choice. That is
  the owner's "about 1% and then it stopped learning".
- **One-bit kept gaining with length.** The gap to the control narrowed from 10.8 to 7.8 panel
  points between 2,000 and 21,000 updates (F034).
- **The leader's cheaper round has two parts.** The one-bit feed-forward kernel saves 0.16 ms and
  our smaller vocabulary table saves 0.17 ms. Together they are worth about 3.5 acceptance points.
  A 4-bit drafter can also take the second saving (F034).
- **Tool-call format sensitivity.** On 12 DEV tasks, JSON calls dropped the leader's acceptance
  from 79.7% to 46.7%, while Baseline only fell from 61.8% to 59.9% (F040). This shows sensitivity
  to the format. It does not establish what caused the original advantage.
- **Evaluation state.** The six protected request suites were independent and passed (1.9–5.6%
  over Baseline, F040). The routine DEV panels are reused selection data. The protected agentic
  suite completed 0/15 in every arm: eleven assignments hit the repeated-assistant rule and four
  the turn limit, while eight final workspaces passed their checks. The owner policy needed merged
  strata to produce a verdict.
- **Licence problems in TRAIN.** MATH (`EleutherAI/hendrycks_math`) is AoPS content, which Hugging
  Face disabled after an AoPS DMCA notice on 2025-01-02. The legacy prompt sources
  (`legacy-freeze-002`, `legacy-gsm8k` and others) have no recorded licence audit. UltraChat is
  MIT-licensed but GPT-3.5-generated, so check its terms.
- **Carries over:** calibrated start (+7.4 points), 4-slot blocks for depth 3, cosine schedule,
  `baseline-k` frozen tables, fused F16 storage, the vocabulary shortlist, the shortlist-only file,
  the recovered tensor-core kernel and exact runtime-v10/v11.
- **Comparators any claim must name:** stock Baseline, the 4-bit control, the target's built-in
  draft head (57.9% acceptance, no file, 0.964× Baseline; F037) and the stock 2-bit and 3-bit files.

### Owner decisions (2026-11-21)

- **One-bit only.** Three-valued weights and a second plane are out of scope.
- **Qwen3.5-4B first.** A larger target is a later question (see `Scaling to larger targets`).
- **Rentals.** Do all setup and validation locally. Rent only once the owner is certain the
  remaining step is "rent, train, release". Whether response generation rents depends on its
  measured local duration.
- **Budget (later on 2026-11-21).** A rented run costing hundreds of dollars is too expensive.
  Under discussion: an all-local or nearly free plan at a smaller data scale. Also proposed: train
  the 4-bit control only until its DEV curve plateaus, instead of at matched length, and show that
  curve with the claim.
- **Benchmarks.** The goal is no remaining uncertainty about benchmarking. Proposed form, pending
  confirmation: local acceptance checks for selection during training, and a frozen public gate
  for release (workstream D).

### Industry practice (web research, 2026-11-21; key claims checked by the review)

- **Data scale.** EAGLE-3 reproductions use ShareGPT plus UltraChat, about 320K conversations.
  DFlash uses about 800K samples. DFlare (Peking University and Tencent, June 2026) uses 2.4M from
  Nemotron Post-Training v2, CodeAlpaca and Step-3.5-Flash-SFT. Open-PerfectBlend has about 1.3M
  samples from eight components (chat 17.6%, maths 39.4%, code 38.9%, instruction following 4.1%).
  SGLang's SpecBundle and NVIDIA's NeMo DFlash and DSpark recipes use it.
- **Regenerating responses with the target is standard practice**, using its recommended sampling
  settings. Regeneration does not by itself clear rights. Earlier assistant turns, quoted answers
  and tool outputs survive inside later prompts, and dataset cards carry per-sample licences and
  downstream-terms warnings.
- **Licences seen.** Nemotron Post-Training v2 is mostly CC-BY-4.0, with WildChat prompts ODC-BY
  and StackOverflow CC-BY-SA, and per-sample licences. Its card warns about Qwen and DeepSeek terms.
  Open-PerfectBlend includes 395K MetaMathQA examples, which are augmentations of GSM8K and MATH
  training problems, so part of it inherits the AoPS problem. Its card is an inventory, not a
  clearance. xLAM-60k is gated, reportedly CC-BY-4.0. Hermes and ToolACE are unverified.
- **Training frameworks.** SpecForge, NeMo AutoModel and vLLM Speculators support DFlash- or
  EAGLE-style training, usually computing target features online at large scale.
- **Objective.** DistillSpec (ICLR 2024): under speculative sampling, expected token acceptance
  equals 1 minus the total-variation distance between drafter and target at a shared history.
  On-policy data matters, and the best divergence depends on decoding mode. This motivates, but
  does not predict, a top-k KL gain for a multi-slot greedy block drafter.
- **Multilingual.** Several papers report that drafters accept much less outside English, roughly
  tracking how much data a language has. Language-specific data helps.
- **Evaluation.** DFlash reports τ and speed-up on GSM8K, MATH-500, AIME25, HumanEval, MBPP,
  LiveCodeBench and MT-Bench, and DFlare on six of these, greedy and at temperature 1. Shared task
  names do not make numbers comparable across target precision, block size, runtime, hardware,
  prompt format, output caps and speed denominator.
  SPEED-Bench (NVIDIA, ICML 2026) is the newest dedicated benchmark. Its qualitative split draws on
  18 sources selected for semantic diversity, with per-category acceptance. Its throughput split
  covers 1K–32K inputs across concurrency levels. Licence: NVIDIA Evaluation Dataset License, with
  some examples masked behind their framework. That framework supports vLLM, SGLang and
  TensorRT-LLM, not llama.cpp.
- **Evaluation pitfalls.** Batch 1 alone overstates gains, which shrink as batch grows.
  Random-token throughput tests distort results. Report τ separately from wall-clock speed-up, and
  state prefix caching, output caps, temperature and the no-draft baseline server.
- **Closest prior art.** "Quantize the Target, Quantize the Drafter" (Nota, July 2026, an ICML
  AdaptFM competition report) uses our target, Qwen3.5-4B. It trains a 5-layer DFlash drafter on
  the BF16 target, adapts it to the INT4 target, then quantizes it to GPTQ INT4 (group 128) and adds
  sliding-window attention. Its 3.57× is against the BF16 target alone, and target quantization
  alone gives 2.16× in the same table, so it is not a drafting-only gain. Two searches found no
  one-bit drafter. ML-SpecQD (Intel, 2025) only suggests a one-bit target copy as a second-level
  draft. Novelty is plausible but not established.

### Workstream A: scope decisions

1. **Release form.** Decided: one-bit only.
2. **Targets.** Decided: Qwen3.5-4B first. Open: Q4KM only, or also NVFP4 and BF16.
3. **Response and feature source.** Q4KM in our runtime, as now, or BF16 responses from a batched
   server replayed through Q4KM for features. The redesigned mismatch pilot decides this.
4. **Serving path for users.** Our llama.cpp fork, an upstream llama.cpp PR, or also vLLM/SGLang.
5. **Licence and provenance.** Undecided (`README.md`). Training starts from a conversion of
   Baseline's weights, so Baseline's licence and Qwen's (Apache-2.0) constrain the release. The
   model card must say "trained from converted Baseline weights", not "from scratch".
6. **Claim.** Decided 2026-11-21: both. The release claims "faster than stock Baseline" and
   "comparable speed to a Baseline fine-tuned on the same data, while smaller". The gate
   tests both, with margins the owner sets before any run. Size wording decided 2026-11-21 (option
   b): the release ships the shortlist-only file with its own vocabulary rows (284 MB today,
   74.5% of Baseline's 381 MB) and claims "about 25% smaller", which keeps vocabulary-table
   training open. Both claims name that one exported file and the actual fine-tuned comparator.

### Workstream B: production TRAIN data

- **Rights audit first.** Per source, record licence, revision, attribution duty, which fields
  survive into prompts (earlier turns, quoted answers, tool outputs), ancestry and a verdict. Drop
  MATH, MetaMathQA and other MATH-derived sets, unaudited legacy sources, and anything whose terms
  forbid this use. Audit the Baseline checkpoint's licence and data notices the same way.
- **Volume.** About 100K conversations and one pass (about 1.25M updates), reduced if the scaling
  pilot does not justify it. All-local fallback: about 250K candidate updates (about 17.6 training
  hours, about 20K old-record equivalents).
- **Base prompt pool.** Nemotron Post-Training v2 prompts (CC-BY-4.0 subsets) as the backbone, plus
  targeted sources for long documents, multilingual and tools. Open-PerfectBlend's shares are a
  reference point, not our mix.
- **Mix (draft shares).** Chat and instruction following 20%. Coding 20%. Agentic and tool sessions
  15%. Maths and reasoning 15%. Long documents (RAG, summarization, QA up to 32K) 10%. Writing 8%.
  Multilingual 7%. Structured output 5%.
- **Candidate sources, each to audit.** Chat: Nemotron v2, oasst2, WildChat (ODC-BY). Maths: GSM8K
  train (MIT), Orca-Math, synthetic problems written by an open Apache model. Code: CommitPackFT,
  permissive OSS-Instruct-style sets, our own repository tasks. Long documents: QASPER (CC-BY),
  public-domain government reports. Tools: Glaive, xLAM/APIGen, ToolACE, Hermes.
- **Tool-call diversity.** At least four conventions: Qwen native XML, JSON inside the native
  delimiters, OpenAI-style function JSON, and diff-style editing agents. Use several harnesses,
  system prompts and MCP-style schemas, with parallel and multi-turn calls. Hold one convention
  out of TRAIN as a transfer test.
- **Generation.** Thinking on and off. Greedy plus sampled with Qwen's recommended settings, since
  sampled is our weakest scenario. Regenerate every assistant turn of multi-turn sessions.
- **Hygiene.** Exact and near-duplicate removal, decontamination against the frozen public gate,
  the heldout24 and confirmation exclusions, and per-source caps.
- **Data card** shipped with the release.

### Workstream C: pipeline and compute

Measured scale (F033, train-bank expansion): about 10.6 MB per fused record, 12.5 updates per
record per pass, and 6.7 s per record for single-stream generation plus capture. A regenerated
multi-turn conversation need not match an old record's size.

- **Storage.** 100K records would be about 1.06 TB fused. Staged shards keep only a working set,
  but a restart needs durable token streams, cursor and optimizer state.
- **Generation.** About 186 local hours single-stream at 100K. Batched writing measured 2.2–2.5×
  faster on a small greedy writing sample only. Sampled, long and multi-turn writing and replay
  are untimed, and the batched writer currently expects existing one-at-a-time answers, so
  production use needs a build.
- **Training.** About 88 local hours for the candidate's 1.25M updates.
- **Recommended local pipeline: staged replay.** Save token streams only. Generate, replay
  features, train and discard features in bounded shards, with a deterministic mixing schedule and
  resume cursor. Qualify one full cycle locally, including export, restart and byte and label
  checks. The rented run executes the same code.
- **Online capture** is an alternative only if a representative long-input memory measurement fits.
  That measurement must cover target weights, KV cache, optimizer, activations and both runtimes'
  workspaces. The last trainer already used 14–15 GiB of 15.5 GiB.
- **Pre-fusion taps.** Fused records already contain the frozen fusion table's output. Training the
  fusion table (workstream F) needs pre-fusion taps, which changes the storage design. Decide
  before generating features.
- **Free disk.** Retire obsolete states and captures after owner approval of the list.

### Workstream D: evals and the release gate

The gate is frozen before TRAIN is acquired. Decontamination depends on it, and the claim scope
decides which data and measurements matter.

1. **Selection tier (private DEV, rebuilt).** Drawn from the new mix's sources, held out by group,
   covering every scenario including multilingual and each tool convention. For selection only.
2. **Public gate.** Core: SPEED-Bench's qualitative split for per-category acceptance, and its
   throughput split for input length and concurrency, if its licence allows use through our
   harness. Add the DFlash task set for context: GSM8K, MATH-500, AIME25, HumanEval, MBPP,
   LiveCodeBench, MT-Bench. Do not redistribute MATH-500 or AIME prompts or outputs. Add BFCL tool
   prompts under several conventions and a LongBench subset. MBPP and GovReport through LongBench
   overlap the spent confirmation reserve's ancestry, so mark those results as previously exposed.
3. **Reserve.** heldout24 stays untouched. A claim of independent confirmation needs fresh source
   families.

Freeze before the first run: prompt revisions and IDs, rendering, thinking mode, sampling filters,
EOS and caps, cache state, depth, request arrival pattern, timing boundaries, denominator, failure
handling, and exact τ accounting. τ counts accepted draft tokens per round. Emitted tokens per
round add one target token, so publish both. Keep client concurrency and server batch size as
separate settings.

Qualify the port first: run target-only, stock Baseline and the old leader through our harness and
check outputs and counters against known values.

Gate rule (proposal, numbers set by the owner before any run):

- One preregistered weighted request-throughput ratio. Its paired lower bound must exceed 1.00
  against stock Baseline. If the claim covers the matched fine-tuned Baseline, its lower bound
  must also exceed 0.97 against that arm.
- Simultaneous per-category guardrails set in advance, exact greedy outputs, sampler validity.
- Size the test from stock-versus-stock reversed-order repeats, which measure session drift, and
  from resampled historical source groups, aiming to detect a 2% gain.
- An unavailable interval or an unsupported concurrency setting cannot pass.

**Agentic.** Replaying recorded target-only agent sessions supports a request-speed claim. A
claim of faster useful agent work also needs live target-only successes across several projects,
identical greedy trajectories, and tool time and failures retained.

**Reporting.** τ and wall-clock speed-up separately, against a no-draft server. State prefix
caching, caps, temperature, context and concurrency. Use real prompts only.

### Workstream E: comparison arms

- target-only, the speed-up denominator;
- stock Baseline;
- **4-bit control trained to a plateau** (owner, 2026-11-21) by our maintained trainer, with an
  equal small tuning allowance for each precision. Freeze one group-stratified, deterministically
  mixed manifest of conversations, token streams and supervised spans. Publish each arm's coverage,
  tokens and updates, and the control's whole curve. If the control stops on a prefix, the claim
  reads "fine-tuned on a stratified subset of the same corpus" with the fraction, and same-prefix
  checkpoints of both arms are compared. Plateau rule (proposal): checkpoint spacing and a minimum
  worthwhile gain tied to the release margin are fixed in advance. After warm-up, two successive
  doublings of exposure must show paired upper bounds on further DEV gain below that threshold,
  accounting for repeated looks, on the weighted outcome and on agentic, sampled and tool-format
  strata. Extend if any of these still improves, intervals stay too wide, or a decayed step size
  could explain the flat curve. F034's ordinary panel was flat while agentic first choice rose
  from 57% to 69%. Confirm the chosen checkpoint with served timing at calibrated depth;
- 4-bit control with the same shortlist, for attribution;
- the target's built-in draft head, the stock 2-bit and 3-bit files, and the old leader `ef068eba…`.

Report equal-depth attribution separately from each arm's best calibrated depth for deployment.
Split every gain into what only one bit earns and what Baseline would also gain.

### Workstream F: features to screen at the new scale

Screen locally in matched short runs (about 1–2 GPU hours per arm, two seeds), and promote only
what survives on served outcomes. "Before" says why it was not done or not done properly.

| Feature | Before | Screen | Build |
| --- | --- | --- | --- |
| Vocabulary-table training | Deferred for memory, never started (item 7) | The 68,728 shortlisted rows only, Q4_K rounding in the loop | Moderate; ships the 284 MB form instead of 185 MB |
| Attention-table training, done properly | Half batch for memory, Q4_0 export cost about 1.7% speed, export rounding cost about 1 DEV top-1 point (F031) | Q4_K rounding in the loop and a Q4_K export via the runtime's own quantizer; attention only first, since the fusion table needs pre-fusion taps | Q4_K export |
| Low-rank correction on frozen tables | Never tried | Baseline's tables kept exactly, a small learned correction | Runtime change |
| Input rotation before 4-bit rounding | Never tried | Targets the largest loss at initialization (60.8% to 41.3%) | Serving kernel |
| One-bit attention tables | Not started (item 5) | Smaller file and possibly faster rounds; large acceptance risk | Owner decision on scope, then kernel reuse |
| Soft targets | Never tried | Hard labels against hard labels plus top-32 KL with a residual bucket, served sampled outcome | Capture of top-k |
| Batched generation | Rejected as a precaution; only 8 of 32 answers matched one-at-a-time; never trained on | Matched training arms on batched against one-at-a-time data | None |
| BF16 server responses | Never tried | BF16 responses replayed for Q4KM features | Server setup |
| 32K training context | Long sources ran out | New long-document data | None |
| Held-out tool convention | Only one convention existed | Acceptance on a convention absent from TRAIN | None |
| Data scaling | Partly (F025, F033) | Mechanism and spending checks under `Pilots and preflight` | None |

Second review (2026-11-21): arm length does not set precision. Earlier short two-seed screens
resolved +7.4 points but left about one-point effects unresolved (F032, F039). Estimate each
screen's detectable effect from prior paired differences and allow an inconclusive verdict.
Order and gating:
- first: batched generation (with sampled, long and multi-turn records and replay cost) and data
  scaling;
- freeze the held-out tool convention before acquisition, and say whether it is selection or
  independent evidence;
- verify long supervised windows, memory and long DEV groups before 32K training;
- vocabulary tables: run item 7's cheap saved-slot probes first, and rebuild the shortlist from
  new TRAIN labels;
- attention tables: attention only, as a separate build from rounding-aware training and Q4_K
  export (F031 trained attention and fusion together);
- soft targets: measure retained mass, shortlist misses and capture cost first; build full capture
  only with enough sampled requests to resolve a timing threshold;
- deferred behind probes, with no builds yet: input rotation (removing input rounding from the
  trained model barely moved DEV), low-rank correction, one-bit attention (bound its benefit from
  F030's timing breakdown first), and BF16 responses unless they save measured writing time.
Serving depth is recalibrated with `calibrate-depth` if acceptance rises.

Deferred: the two-channel bypass retest, a deeper or wider body, a standard-trainer 4-bit control
with GGUF conversion, the 9B round profile, and extra GPUs. Excluded by the one-bit decision: higher
precision for salient weights only (about 1.1 bits). None of these is needed to decide
whether broader clean data improves the one-bit 4B release.

Previously dropped: zero-skipping and slot-reuse kernels, pre-summed lookup tables on GPU,
activations below 4 bits, Q2_K/Q3_K vocabulary tables on this runtime, the full-precision drafter
teacher (owner, 2026-11-16), three-valued weights and a second plane (owner, 2026-11-21).

### Workstream G: release packaging

- Licence, model card with the claim wording and provenance, data card, pinned target, tokenizer,
  template, drafter and runtime hashes, and reproducible eval commands.
- **Clean-install and serving test.** Reproduce one published result from an outside checkout.
  Check that an incompatible model is rejected, sampler and EOS boundaries, mixed-length concurrent
  requests, cancellation, KV isolation, context limits and fallback.
- Publish cold start, peak resident memory, time to first token, inter-token latency and aggregate
  throughput for the supported settings.
- Lead with the 284 MB form unless the new release earns a different choice (owner, 2026-11-21).
  The 185 MB form saves disk only and adds about 5.6 s of start-up.

### Pilots and preflight

- **Scaling, mechanism check (under 10 GPU minutes per arm).** Ten minutes is about 2,370 updates,
  roughly 190 record-equivalents. Compare 64 against 256 length-balanced source groups at matched
  token budgets and a fixed schedule, for one-bit and the 4-bit control. Log unique groups,
  windows, supervised positions and repeat visits. The small arm must revisit its positions at least
  twice while the large arm still sees fresh ones. A null result here says little about corpus
  scale.
- **Scaling, spending decision.** Two paired seeds, identical starts within each precision, matched
  mix and order, untouched group-disjoint DEV. Estimate the one-bit data benefit and its difference
  from the 4-bit benefit. Worthwhile: +1 acceptance point for one-bit. An interval spanning 0 and +1
  is inconclusive. Run length is set from the mechanism check.
- **Response and feature mismatch.** Step 1: 200 stratified prompts with 32 shared teacher-forced
  positions each. Measure top-choice agreement, probability mass and feature drift on identical
  token IDs, with uncertainty clustered by prompt. Step 2: three matched training arms. They are
  Q4KM responses with Q4KM features, BF16 responses replayed for Q4KM features, and BF16 responses
  with BF16 features. Decide on served Q4KM acceptance and timing with paired seeds, using a
  non-inferiority margin of one acceptance point.
- **Soft targets.** Hard labels alone against hard labels plus one fixed KL weight. Use identical
  windows and steps, two seeds, and top-32 plus a residual-mass bucket. Record retained mass,
  shortlist misses and capture cost. The primary outcome is served sampled accepted tokens per
  round and request speed, with greedy regression checked. Stop if the bound rules out the
  predeclared useful gain.
- **Local shard cost.** Before any rental, run one stratified shard end to end: long prompts,
  thinking, multi-turn tool sessions, generation, replay, training and evaluation. Forecast each
  phase separately from observed token and window counts. This replaces the earlier one-hour
  rental pilot, which conflicted with the rental decision.

### Compute and cost estimate (revised after the second review, 2026-11-21; no rented measurement)

- **GPU choice.** The trainer uses F32 SIMT arithmetic. The RTX 5090 shares the 5080's
  architecture (sm_120) and has 32 GB. Its speed-up is unmeasured, and CPU feeding, memory
  bandwidth and grouping may limit it. Treat 1.0–1.6× as a sensitivity range, not bounds.
- **Serial rental hours for the one-bit arm at 100K.** About 81 local-equivalent hours of
  generation (the 2.3× batched rate applied to all historical generation and capture time, an
  optimistic assumption) plus 88 of training. That is about 169 local hours, or about 105–170
  rented hours (4.4–7 days) before replay, screens and retries.
- **Prices.** RTX 5090 roughly $0.45–0.70 per hour on marketplace or community tiers and $1.39
  on RunPod's on-demand list. That is about $50–120 cheap or about $150–235 on-demand, before
  overhead. With storage, setup, screens, checkpoint transfer and a restart allowance: about
  $60–150 cheap, up to about $300 on-demand. Use an actual instance quote.
- **Controls on spending.** Set a hard dollar ceiling, and stop after the first rented shard if the
  measured rate projects past it.
- **Internet.** Upload code, prompts and token streams (a few GB). Models download from Hugging
  Face on the rental; exports and checkpoints come back. RunPod lists no transfer fees; the cost is
  paid time while bytes move.
- **Cross-host identity.** Use identical token IDs, spans, target and template revisions, replay
  geometry, fusion operand hash and pinned runtime source on both hosts. On the first rented
  shard, compare tokens, labels, features and an export round trip against a local fixture, with
  acceptable drift fixed in advance. Rental screens need fresh 5090 controls, or exports return to
  the 5080. All release timing runs on the 5080.
- **All-local fallback.** About 250K candidate updates (about 17.6 training hours) on about 20K
  conversations, plus generation, control, replay and evaluation, costs $0 rented. It stays a
  credible release experiment if the scaling pilot cannot justify 100K.

### Stages and owner checkpoints (plan frozen by the owner, 2026-11-21)

Changes after this point happen only at the go/no-go points, by rules fixed in advance. No
further redesigns or reviews unless a pilot fails its rule.

Target dates if Stage 1 starts 2026-11-21 and owner checkpoints are answered the same day:
Stage 1 by 2026-11-22, Stage 2 by 2026-11-24, Stage 3 by about 2026-11-29 to 12-02 rented (about
11-26 on the fallback), Stage 4 and release about 2026-12-01 to 12-03. Owner response time and other
threads holding the GPU add to these dates.

1. **Prep** (local, mostly CPU; about 1–2 days at agent pace, gated by builds and GPU qualification; $0). Rights audit and source list; freeze the
   public gate, protocol and margins, the control's plateau rule and the held-out tool
   convention; port and qualify the public gate; rebuild the private DEV check; build the batched
   writer and the staged-replay pipeline. Owner at the start: go, gate margins, fork-only first
   release and Q4KM only (both recommended). Owner at the end: approve the frozen gate and the
   source list.
2. **Screening** (local GPU, about 20–25 GPU-hours, about 1.5–2 days; about $1 if the 5090 test is allowed). Batched
   versus one-at-a-time data, the data-scaling checks, then cheap probes (vocabulary and attention
   tables, target probabilities), with builds only where a probe justifies one. Owner at the end:
   the recipe, the scale (100K rented or the all-local fallback), the spending cap and an explicit
   go for the rental.
3. **Final training** (compute-bound: about 4.5–7 days on one rented 5090, or about 2.5–4 days on two that overlap generation and training at a similar total cost; about $60–150 cheap tier; the all-local fallback takes about 2 days and $0). One
   local end-to-end chunk; the one-bit run on a rented 5090 with the first-chunk identity and pace
   check; the 4-bit control locally until its plateau rule is met. The owner is involved only if
   the first-chunk rule trips, the cap would be exceeded, or the control has not plateaued.
4. **Gate and release** (local, about 1 day; $0). Frozen public gate on the 5080,
   clean-install test, close-results, model and data cards. Owner: licence, claim wording and the
   go to publish.

### Order and timeline (owner arrangement 2026-11-21: 4-bit control local; one-bit data generation and training on rented RTX 5090s; batched generation is piloted)

| Phase | Where | Work | Calendar | Cost |
| --- | --- | --- | --- | --- |
| 0. Decide and freeze | Local CPU | Rights audit, source list, public gate selection, protocol and margins (owner), decontamination list | 1–2 days | $0 |
| 1. Public gate port | Local, GPU hours | Build harness port, qualify on target-only, stock Baseline and old leader | 1–2 days | $0 |
| 2. Pilots | Local GPU, about 2–3 days busy | Scaling mechanism and spending checks; generation pilot (one-at-a-time Q4KM, batched Q4KM, BF16 replayed for Q4KM features; 2 seeds); soft targets; feature screens (vocabulary-table and attention-table training with rounding in the loop and a Q4_K export, input rotation); one-hour rented 5090 trainer benchmark if the owner allows | 5–7 days | about $1 |
| 3. Data and pipeline | Local CPU, plus about 1 GPU day | Prompt acquisition, deduplication and decontamination; new private DEV and references; staged-replay pipeline and rental scripts; one reduced shard end to end on the 5080 | 4–6 days, overlaps phase 2 | $0 |
| 4a. One-bit run | Rented 5090 | Generate 100K conversations, replay features in shards, train 1.25M updates (pilot may shrink this), identity check on the first shard, screens | 5–8 days | about $60–150 cheap tier, up to about $300 on-demand; hard ceiling |
| 4b. 4-bit control | Local, about 1 GPU day | Download token streams for its slice, replay features, train until the DEV curve plateaus, screens | Parallel with 4a | $0 |
| 5. Gate and release | Local | Frozen public gate on the 5080, clean-install test, close-results, licence, model and data cards | 2–4 days | $0 |

Total: dates and durations are set in `Stages and owner checkpoints`, which supersedes this
table's calendar column. About 4–6 days of 5080 GPU time in all, and about $60–150 rented on a cheap
tier ($0 on the all-local fallback). All 5080 work shares one card in one schedule.

Go/no-go points: no data benefit in the scaling pilot stops or shrinks the plan; the generation
pilot picks the writing method; the screens pick the recipe; the rented benchmark sets a cost cap;
the owner gives an explicit go before any rental; a control curve that has not plateaued means
either longer control training or a weaker claim.

### Scaling to larger targets (reasoning, not measured)

- Only the 4B target has been measured. There, target verification is about 80% of a round (F030).
  The leader's 0.33 ms cheaper round, half from one bit and half from the smaller vocabulary table,
  offset about 3.5 acceptance points (F034).
- Drafter cost grows with the target's width, while target cost grows with width and depth.
  Verification therefore takes a larger share on larger targets, and a cheaper drafter buys less
  relative speed.
- At 4B the one-bit FFN (about 56 MB) fits in the 5080's 64 MB L2 cache and 4-bit (about 250 MB)
  does not. Wider drafters push both out of cache, which may raise one bit's saving in
  milliseconds, but its share of the round still shrinks.
- "At least the same speed" on larger targets therefore needs the acceptance gap nearly closed.
  "Smaller" holds at any size. Published one-bit results lose less at greater width, but our
  drafters have not been tested at another width.

## Next

Finish the full bank, rebuild the private DEV check and its references, build the staged
driver, then start the Stage 2 pilots (batched versus one-at-a-time first), each behind a
preflight.
