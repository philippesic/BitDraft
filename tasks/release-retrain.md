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
- **Scaling mechanism check result (2026-11-22)** (`artifacts/datasets/retrain-01/scaling/`,
  2,370 updates, retrain DEV screen, candidate-only against saved references): one-bit 256
  records 47.1% and 47.5% acceptance (seeds 8101, 8102; 0.94× and 0.95× Baseline); one-bit 64
  records 26.5% (0.77×); 4-bit control 256 records 58.0%, 64 records 57.5% (both 0.99×). Mean
  label visits 4.4 against 16.6. The one-bit drafter loses 21 points to repetition, far beyond
  its 0.34-point seed spread; the control barely moves. Distinct data is the one-bit lever, so
  the spending check runs. Grouping uses 2 GiB activations: 3 GiB ran out of GPU memory on
  these sequences (peak 14.25 GiB), so production intents use 2 GiB.
- **Preflight: scaling spending check (2026-11-22).**
  - Question: once the smaller arm is near one pass, as in the all-local fallback (20K records,
    250K updates, about 12.5 updates per record), does 4× more distinct data still raise one-bit
    acceptance by the worthwhile +1 point?
  - Design: 2,370 updates; 192 records (about one pass) against 768 (about a quarter), the 192 a
    category- and length-stratified subset of the 768; one-bit seeds 8101 and 8102 per size, one
    4-bit control pair (seed 8101); same start, schedule and grouping (2 GiB); DEV screen against
    saved references.
  - Decision-changing result: one-bit 768 minus 192 at least +1 point on both seeds supports
    the rented 100K scale; a difference below the 0.34-point seed spread supports the fallback;
    between is inconclusive and the owner decides on cost.
  - Limitation: a 2,370-update proxy of a 250K-update regime; it tests the data-per-pass
    mechanism, not the absolute gain at production length.
  - Cost: write 512 more answers (about 15 minutes), replay 960 records (about 35), fuse, six
    arms of about 10 minutes: about 1.9 GPU-hours. Kill: refusal, out of memory, an arm over 20
    minutes.
- **Scaling spending check result (2026-11-22)** (`artifacts/datasets/retrain-01/spending/`): one-bit
  768 against 192 records at 2,370 updates: 48.3% against 46.7% (seed 8101, +1.6) and 49.4%
  against 47.0% (seed 8102, +2.4), both above the +1 bar; 4-bit control 58.6% against 58.5%.
  Even near one pass, more distinct data still pays for one-bit: the evidence favours the
  rented 100K scale over the local fallback (owner decides at the checkpoint, on cost).
  Qualifications (pre-rental review): paired 96-group intervals +1.64 (+0.60, +2.66) and +2.45
  (+1.61, +3.24); the arms received 1.41 and 0.39 passes by realized update groups, and their
  calibrations fitted different values. The result supports paying for distinct data at 2,370
  updates; it does not isolate volume from calibration or predict the 100K-versus-20K gain.
- **Replay 4.5× faster, byte-identical (2026-11-22):** replay was CPU-bound on one core with the
  GPU mostly idle. The capture helper's finiteness check built a heap string per logit (248,320
  per supervised token), and the per-token argmax ran on one thread. Fixed in `native/capture.cpp`
  (message as `const char *`, one check per row, argmax across up to 8 threads); helper
  `artifacts/native/helpers-replay-v3`. 16 records: 33 s to about 7 s in the record loop
  (42 s to 13 s wall); all 144 tap and label files identical to the old helper. Old rate
  2.2 s per record (631 tokens/s); 100K-record replay drops from about 60 hours to about 13.
- **Shortlist probe (2026-11-22, item 7's cheap probe):** on the 256 pilot answers, labels the
  leader's 68,728-row shortlist cannot score, even after adding every request token as the
  runtime does: multilingual 12.6%, conversation 3.0%, writing 0.5%, all others at most 0.15%.
  This matches the leader's multilingual loss (0.86× Baseline). The 256 answers alone hold 23,673
  distinct labels, 6,996 outside the old shortlist. The shortlist must be rebuilt from new TRAIN
  labels (`vocabulary-variants`) before export. Each extra row costs about 1.44 KB of file
  ((542 − 284) MB over 179,592 rows), so a wider shortlist trades against the "about 25% smaller"
  claim; the size is an owner decision at the rental checkpoint, from the coverage curve of the
  generated TRAIN answers.
- **Shortlist coverage curve (2026-11-22):** 1,376 new-mix answers, shortlist built on one half
  and measured on the other (request tokens added as the runtime does). Multilingual labels
  unreachable: old 68,728 rows 13.6%; plus the 30,000 most frequent new labels (77,008 rows, about
  +12 MB) 8.1%; plus every new label seen (83,109 rows) 6.6%. Distinct labels outside the old
  shortlist keep growing by about 18 per answer (24,534 at 1,376 answers), so full multilingual
  coverage at 100K answers would likely need 150K+ rows (about 400+ MB, above Baseline's 381 MB).
  Coverage and the size claim conflict. The shortlist is an export-time choice, so it does not
  block the rental. Proposed rule for the owner, fixed before the final export: build 3–4
  shortlist sizes from exact replay labels of the generated TRAIN answers (sampled histories
  included), measure each on the DEV panel, and ship the smallest whose speed is within 1% of the
  best, subject to a preregistered file-size ceiling (owner: for example at most 305 MB, 20%
  under Baseline's 381 MB) and a multilingual guardrail (DEV multilingual speed at least 0.97×
  Baseline). If no size passes both, the size claim is dropped or reworded rather than the
  guardrail. Break-even with Baseline is about 136K rows.
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
  sampled is our weakest scenario. Agentic sessions keep their recorded earlier assistant and
  tool turns as context and supervise only the regenerated next response.
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

Gate verdict (one calculation, frozen 2026-11-22 to match `release-gate-v1.json`; margins remain
the owner's to approve):

- Primary: the general scenario (880 prompts, both SPEED-Bench files pooled), speed as generated
  tokens over summed request seconds, paired source-group bootstrap with category strata (2,000
  draws). Pass against stock Baseline: lower bound above 1.00.
- Guardrails: five scenarios (thinking-on, sampled, long-input, agentic replay, tool calls), each
  lower bound at least 0.97 with at most two below 1.00. The held-out convention is reported as
  transfer evidence, not a pass condition (owner, 2026-11-23).
  Categories inside a scenario are descriptive, not guardrails. Multilingual has no gate scenario
  of its own: SPEED-Bench's multilingual category is reported, and the DEV panel's multilingual
  scenario is checked before the gate run.
- Each arm runs at its own calibrated depth (`calibrate-depth`) for the verdict; equal-depth
  results are published for attribution only. Sampled prompts run 3 seeds, pooled as one group per
  prompt. Agentic replay has one task per reserved repository, so groups are repositories.
- Against the trained 4-bit control: general lower bound at least 0.97, every guardrail at least
  0.95; claim wording "within 3% of a Baseline fine-tuned on the same data, or faster".
- Any missing scenario, unavailable interval or invalid output fails. Before the gate, stock
  Baseline runs against itself in reversed arm order on the general scenario to measure session
  drift; a lead smaller than that drift is not claimed. The claim covers the measured batch-one
  request workload only.
- The ±0.6% resolution at 880 requests is extrapolated from the 292-request DEV panel
  (0.984–1.005 around 0.994); the reversed-order run measures it on the gate itself.

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
  Executable form (2026-11-22, after the pre-rental review; amends `control_plateau_rule` when
  the owner approves the margins): the plain control now resumes and continues. It trains one
  fresh constant-step run on the candidate's generated streams in production order, extended by
  resume at 12,500 × 2^n updates so its checkpoints sit on candidate stage exports (12.5K, 25K,
  50K, 100K, ...); outcomes on the DEV screen and a sampled DEV screen (to be built from sampled
  DEV rows, with saved target and Baseline references); the anneal continues on the next unseen
  streams with a cosine schedule. Same-prefix comparisons use candidate exports at those counts,
  noting that the candidate's step size follows its own long cosine;
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

### Rental runbook (revised 2026-11-22 after the pre-rental review; runs only after the owner's go)

Inputs live in `artifacts/rental/`: `production.json` (100 stages × 1,000 bank rows × 8,250
updates = 825,000, about one pass, cosine with 16,500 warmup, telemetry off, 2 GiB grouping, checkpoints every 2,500 updates,
writer at 32 slots without decode consistency, a 294,912-token shared KV pool and the balanced
bank order `order_seed` 9401, replay helper v3, 120 GB per-stage replay ceiling),
`make_production.py`, `make_manifest.py`, `manifest.txt` (43 entries, 20 GB) and `pull.sh`.
Specs pin absolute paths and SHA256s, so the rental checkout must sit at `/home/philip/BitDraft`.
Balanced order: every 1,000-row stage holds 37–39 agentic, 89–92 tool, 84–86 multilingual and
95–98 document rows; worst-case raw replay (prompt plus output cap) is 92–97 GB per stage.

1. Cap and credit: the all-in cap is $70 in `artifacts/rental/watchdog.json`; the owner tops up
   Vast credit as the run goes (alert under 8 hours of credit, clean stop under 1 hour).
2. Host: Vast.ai instance 62255758 (machine 140586, RTX 5090, CUDA 13.2, `ubuntu:26.04`, 450 GB),
   stopped with every input uploaded. Start it with the API; if the GPU was taken, rent a
   comparable host and repeat steps 3–4 (about 1.5 hours of upload at the owner's 3–4 MB/s).
3. Code and inputs: `git bundle create artifacts/rental/code.bundle main`, copy it and fetch
   `main` on the host; rsync `manifest-rest.txt` (only changed files move: the bank, specs,
   scripts). Write `artifacts/rental/expected-source.txt` at home from the bundle's commit.
4. `artifacts/rental/setup.sh` on the host: pinned uv 0.11.32 and Python 3.11.15, both
   environments, CUDA 13.1 libraries and libgomp checked by `ldd`, the source identity against
   `expected-source.txt`, the initialization rebuilt and matched to its SHA256, and the replay
   identity check (16 records, 128 tap files, tokens exact, labels within 3%, layer-2 drift within
   2%).
5. Output: `mkdir artifacts/rental/production-run`, copy `expected-source.txt` into it,
   `echo 6 > production-run/backup-lag` (training may run six stages, about four hours, ahead of
   the home backup before it waits), `touch production-run/hold`.
6. Home: `systemctl --user enable --now bitdraft-pull bitdraft-watchdog` (user services with
   lingering, restart after failure or reboot, sleep inhibited while running; host and port from
   `artifacts/rental/watchdog.json`) runs `artifacts/rental/pull.sh HOST PORT` (two generations, verified newest stage,
   refusal on a different spec or a host behind home, acknowledgment after verification) and
   `artifacts/rental/watchdog.sh` (5-minute checks: restarts the driver up to 3 times in 6
   hours, stops the instance at completion, unrecoverable failure, cap or an unreleased hold,
   and wakes the monitoring agent through jobwatch). The watchdog launches the driver.
7. Stage 0 stops at the hold. The agent projects the run from stage 0's measured times
   (diagnostic screen only: stage 0 ends inside warmup) and, if within the cap, removes `hold`
   and `held.json`; the watchdog relaunches.
8. Interruption on the same host: the driver resumes, or restarts the stage from the previous
   stage when the resume is refused. Host lost: rent a new host, steps 3–4, restore
   `production-home/current`, write `backup-ack.json` as `{"stage": K}` for home's verified
   stage K (the mirrored copy trails by one sync), restart pull and watchdog; stages whose
   features or state were not backed up restart from the newest verified stage.
9. End: after stage 99 is verified at home the watchdog stops billing. Copy anything still
   needed, verify, then destroy the instance (storage costs about $2 a day while stopped).

Forecast (pilot rates, before the representative local stage): generation about 800 tokens/s,
replay 0.42 s and fusing 0.15 s per record, training 3.95 updates/s give 31 + 12 + 4 + 88 hours
plus about 6 of stage overhead at 100 stages, about 141 local hours. The generation rate came
from 512 rows of the old seeded order, which was cheaper than average; the representative local
stage replaces these rates. At an unmeasured 1.0–1.6× 5090 speed-up, 88–141 rented hours before
setup, transfers and retries.

## Next

Status (2026-11-23): Stage 1 and Stage 2 are complete. Stage 2 results are closed in
[release-stage2-screens](../results/release-stage2-screens/result.md) (F041): the recipe is
unchanged (frozen Baseline tables, hard labels, batched writing, about one pass over 100K distinct
records). The rental is held until the owner approves it.

Owner decisions on record:
- 2026-11-22: "not yet" on the rental; full recap requested.
- 2026-11-22: the go covers the whole Workstream F list (screens done; see the result).
- 2026-11-23 about 01:30: overnight plan approved; done, with rounded attention seed 2 held
  because it would have overrun (owner may still ask for it).
- 2026-11-23: fix every finding of the second Astra review before the rental; the held-out ReAct
  convention is reported transfer evidence, not a pass condition.

Second Astra review fixes (2026-11-23):
- Done: the staged driver rebuilds missing replay or fused features after a restore, verifies
  export digests, restarts a stage whose state is missing, retires earlier states on restart,
  allows a full stage of retry attempts, honours `hold`, `expected-source.txt` and `backup-lag`
  control files; `pull.sh` mirrors the run without features, verifies the newest export and state
  and writes `backup-ack.json`; the exported-table diagnostic no longer re-rounds decoded values;
  feature verdicts reworded and closed in the result.
- Host-loss rehearsal 1 (`artifacts/rental/hostloss.sh`, 3 stages × 64 rows): the stage-0 hold
  worked; losing the host during stage 1's replay and restoring the 11 GB pulled copy recovered;
  losing it mid-training in stage 2 did not: the rebuilt replay manifest has a different digest, so
  the stage's derived specifications no longer matched and the restart refused. Fix (on branch
  `replay-overlap`, merged after the pass test): a stage whose features are lost restarts cleanly
  from the previous stage's state (at most one stage of training lost). Rehearsal reruns after.
- Owner (2026-11-23): test one pass (8,429 updates on the long corpus) against production's 1.55
  passes (13,083) locally before deciding (`long/pass-preflight.md`; running, pueue task 47); build
  the low-risk speedups first (training telemetry off, writer slots and batch, replay overlap),
  each confirmed by a pilot (`artifacts/datasets/retrain-01/speedups/pilots.py`).
- To do before the rental: runbook update for the control files and recovery; the 4-bit control's
  executable schedule (below); release clearance (weights licence, attribution for retained
  SWE-rebench context); gate verdict calculator exercised on synthetic rows.

4-bit control schedule (proposal for the owner, executable with the current trainer): train on
the candidate's generated stages in production order. One fresh constant-step run over stage 0's
fused corpus; then `continue` onto each later stage's corpus (exact resume needs the same corpus,
so each new stage is a continuation). Measure at stage boundaries where updates reach
12,500 × 2^n (12.5K, 25K, 50K, 100K, ...), on the greedy DEV screen and the sampled DEV screen
(two seeds, live target and Baseline each time, about 6 minutes, not distribution-qualified).
Stop under the recorded plateau rule; finish with a cosine anneal of 10% of the reached updates on
the next unseen stages; report the corpus fraction used. Candidate exports at the same update
counts give same-prefix comparisons. The ten agentic screen groups cannot resolve a +1-point
subgroup bound; agentic plateau evidence comes from the panel's 30 agentic tasks.

Before the rental, the owner approves: the recipe, the gate verdict and margins, the source list,
the all-in dollar cap and stage-0 stop, the file-size ceiling for the "smaller" claim, the
release licence, and the rental go. Rental provider: a verified datacenter RTX 5090 on Vast.ai
(on-demand), RunPod Community as fallback.

Owner decisions (2026-11-23, after the Astra and Fable pre-launch reviews): fix all eight
must-fix items before renting; checkpoints "somewhat infrequent, just enough so we don't lose
data" (every 4,125 updates, twice per stage; only completed stages' final states go home); an
agent monitors the rental and owns restarts and stopping billing (this thread plus a local
watchdog using a Vast API key stored in `~/.config/vastai/vast_api_key`, mode 600, outside the
repo); keep SWE-rebench sessions and document them in the data card. Cross-GPU check: the
5090-trained export scored 48.40% greedy and 45.97% sampled against 48.78% and 46.19% for the same
500 answers trained on the 5080 (−0.38, −1.16 to +0.39; −0.22, −0.81 to +0.38): no material
difference; disclose that candidate and control train on different GPUs' features.

Owner approval (2026-11-23, before training): the release gate is frozen as written: primary
general scenario (880 prompts, both files pooled) with paired lower bound above 1.00 against stock
Baseline; five guardrails (thinking-on, sampled, long-input, agentic replay, tool calls) each at
least 0.97 with at most two below 1.00; against the trained 4-bit control 0.97 and 0.95; the
held-out ReAct convention reported only; definition `benchmarks/definitions/release-gate-v1.json`
(SHA256 `92eb558ca38acb51e5f6b4b3edc6fa2296324f910ae05fae060237b50d2dc3c8`), workloads `gate-v2`, verdict by `gate-verdict`. The source list
(`retrain-sources-v1.json`) is approved as audited. Any later change needs a new version. The
shortlist is an export-time choice; a multilingual-light variant will be measured beside fuller
ones after training (owner question 2026-11-23).

Frozen verification gate (2026-11-23; plan by the agent, sufficiency reviewed once by Astra; no
further reviews before launch). Step 1, fix the confirmed bugs: done (driver completion receipts
and ancestry, verified-state retirement, refused-resume restart; `pull.sh` newest-stage catch-up,
atomic generations, receipt and parent verification; `watchdog.sh` PID-file liveness, progress
deadline, endpoint refresh, confirmed stop with retries, latched cap, persisted spend, launches and
hold timer; `launch.sh` uploads `expected-source.txt` and `init-intent.json` and verifies the
commit). Step 2, one local PASS/FAIL acceptance test (four 64-row stages, production driver and
scripts, local pull, fake Vast API and clock): (1) kills during writing, replay, fusing, training
after a nonfinal checkpoint and right after export, each proven to hit a live phase; (2) resume
from an intact checkpoint, a forced refused resume, and a restart after several retired stages;
(3) host loss mid-stage and between stages restored from home only, and a backup two stages behind
reaching the lag limit; (4) pull killed during transfer, at promotion and before acknowledgment,
a one-byte corruption in a successor state and in an incoming backup, a wrong spec and a host
behind home; (5) watchdog: absent and live driver, self-matching probe, repeated crashes, stalled
worker, SSH loss, unreleased hold, cap, completion with a delayed final backup, restart across a
clock gap, refused stop and a stop that still reports running; (6) independent invariants after
every scenario: trainer launch counts, earlier export hashes, parent digests, exact cumulative
updates, input IDs against assigned rows and skips, home recovery bytes, peak rental and home disk
projected to production, and the full bank against all four protected indexes. Step 3, opening of
the paid run: numeric cap, credit and monitoring before billing; on-host commit, scripts, bank-v2,
spec and input hashes; initialization digest and the 16-record identity check; stage 0 held at
exactly 8,250 updates with a verified home backup and a projection of spend so far plus 99 stages
within the cap; one real stop and start with both watchdog and pull rediscovering the endpoint;
then release. Go rule: Steps 2 and 3 pass, the run proceeds; a failure is fixed and the full Step 2
rerun once.

Step 2 result (2026-11-23): PASS on the second full run (commit `d0648e67`, 27 minutes, 64/64
checks: 22 watchdog, 42 main). The first run found that every resume after an interruption was
refused: the trainer charges a killed attempt its whole reserved wall budget (10 days), so the
stage restarted from scratch each time (correct result, up to one stage of paid training lost
per interruption). Fixed by `run --attempt-seconds`, which staged training sets to 6 hours.
Also fixed while building the test: the final state is hash-checked before `result.json`
(`411c7895`); the watchdog alerts when the driver waits on home backups or the final backup
never verifies, charges its own downtime, refuses to run without a cap, and records the
driver's own PID. Measured: every kill hit its live phase; the intact-checkpoint resume and the
export-boundary kill both resumed; a corrupted checkpoint was refused and the stage restarted
from stage 1's state; a backup two stages behind held stage 3 until home caught up from 0 to 2;
both host losses restored byte-identically from home; every pull fault left the last good copy
verifying; a corrupted successor state kept its predecessor and blocked completion; bank-v2 has
0 protected matches (positive control: bank-v1 finds the 6). Projection to 100 × 1,000 rows:
rental peak about 164 GB of 450 GB, home about 74 GB (134 GB free). Scripts:
`artifacts/rental/acceptance/` (`run_all.sh`); report and logs in `artifacts/rental/acceptance-out/`.
Artifact disposition: the test's run, home and pull-copy trees (24 GB) were deleted after the
pass; logs and `main-report.json` are kept until the release closeout.

Owner decision (2026-11-23): all-in cap $70 (`watchdog.json` `cap_dollars`), with Vast credit
topped up as the run goes. The watchdog alerts the agent (who pings the owner) when credit
covers under 8 hours, and stops the instance cleanly under 1 hour; after a top-up the agent
restarts it and the driver resumes (a stopped instance does not hold its GPU, so a restart can
force a move to a new host restored from home). Watchdog scenarios: 23/23 pass with these
checks. Step 3 waits on the first top-up and the owner's go.

Owner go for Step 3 (2026-11-24): first top-up made (credit $22.67 at launch, about 31 hours at
$0.72/h); "start the instance from api and get to it".

Step 3 result (2026-11-24): PASS. Launch at commit `de2e9427` on the same host (154.7.90.1:47606);
setup identity 16 records, tokens exact, labels 1.67%, layer-2 drift 1.08%. Stage 0: 8,250
updates, about 35.5 minutes wall (training 886 s, 9.3 updates/s, 2.4x the 5080), held, verified
at home 11:45. Projection: 100 stages about 59 hours, about $43 at $0.72/h, within the $70 cap
(the watchdog counts spend from this launch; earlier upload and storage are outside it). Real
stop and start at 11:46-11:47: same address, stage-0 state hash intact, watchdog logged the stop
and relaunched the driver after the hold was released; pull verified again at 12:00. Hold
released 11:52. 4-bit control started on the 5080 (jobwatch `control-feed`, pueue 67;
`control-run`, pueue 68; run root `artifacts/rental/control-run`). Credit at launch covers about
31 hours; the remaining run needs about $21 more.
Control screens (2026-11-24 13:54): `artifacts/rental/control/screen_loop.sh` (jobwatch
`control-screens`, pueue task 69 in a new pueue group `screen`, to be removed when the control
ends) screens each control export at 8,250 x 2^n updates and applies `plateau.py` with the
candidate's 825,000 as ceiling, into `artifacts/rental/control-screens/`. Pace at 13:54: rental
4 stages complete (about 39 minutes per stage), control 2 stages (about 31 minutes per stage
after each feed).
Mid-run candidate check (2026-11-24 21:00, owner request): training events through stage 14
clean (no non-finite losses, median gradient norm 1.8-2.7, 0 skipped rows); writer answers at
stages 0, 7, 13: none empty, 2-4 looping tails per 1,000, 34-35% stopped at the length limit (local
long corpus 34%). DEV teacher-forced top-1 35.5% at stage 0, about 28-29% since (peak learning
rate; the 14,000-update local run dipped to about 30% mid-schedule and ended at 33.5%). Served DEV
screens (`artifacts/rental/candidate-screens/`, exports hard-linked there), acceptance at 8,250 /
66,000 / 115,500 updates: greedy 48.1 / 49.2 / 49.9%, sampled 46.0 / 46.4 / 47.3%, agentic 51.0 /
60.2 / 64.7%, tool-call 54.5 / 55.9 / 57.8%. Mismatch: the candidate is on a cosine schedule near
peak step size, not annealed; the 4-bit control runs a constant step on the same data (8,250 /
66,000: greedy 57.3 / 57.0%, sampled 53.7 / 53.7%, agentic 62.0 / 68.0%); released Baseline
sampled 55.2%. Local annealed one-bit references at 14,000 updates: 50.8% greedy, 48.4% sampled.
Disk: rehearsal outputs `representative-run`, `speedtest-local`, `first-check-local` and
`representative-run-alias-refused` deleted with owner approval (28 GB); control exports are kept
because the staged driver re-verifies every completed stage's export on relaunch.
Control paused for disk (2026-11-25 03:30): each control stage's raw replay peaks at 71-97 GB
before fusing, and permanent use grows about 1.35 GB per stage (home mirror 0.75, control 0.6), so
free space during a peak fell to 43 GB at stage 24. The feeder (`control-feed`) was stopped after
stage 24's streams; the control finishes stage 24 (264,000 = stage 31 is its next screen) and
waits for stage 25's streams. Free space without a peak: 111 GB; the rental's home mirror needs
about 56 GB more, so the paid run is safe. Resume the control (restart `feed.sh` under jobwatch)
only with at least about 110 GB free beyond the mirror's remaining growth; that needs an owner-
approved retirement. Candidates whose results already name a retirement condition (about 55 GB):
the release-candidate long run's 14K and 28K states (10.2 GB, kept until the owner chose between
continuing that run and a fresh release run); `one-bit-attribution-01`'s four plain-control states
and pilot exports (about 26 GB, kept until the owner decided on a longer control); the
trained-tables pilot and interrupted-attempt states (about 19 GB, "can be retired as superseded").
Not candidates: `datasets/train-final-01-fused` is a component of the current TRAIN set
`train-final-02-fused`.

Overnight plan (owner, 2026-11-24, while the card payment is blocked): build and test the 4-bit
control driver; fix the gate verdict calculator (incomplete evidence accepted, one-sided drift
check); NVFP4 sampled qualification on the release runtime; shortlist tooling with the
multilingual-light variant measured on existing labels (the final build needs the run's labels);
model card, data card, bank-v2 provenance supplement and SWE-rebench attribution drafts. The
control waits for the rental's streams (owner choice: no locally written streams). Nothing
touches the rental; the GPU is free by morning.

Overnight results (2026-11-24):
- 4-bit control ready (`artifacts/rental/control/`): `make_control.py` builds `control.json`
  from `production.json` (same writer, replay, fuse and stages; plain template from
  `plain-256-s1`: Baseline weights `dflash-4b-bf16.gguf`, plain feed-forward, constant step
  after a 1,650-update warm-up, the agent's choice, checkpoints every 4,125). `feed.sh` copies
  each completed candidate stage's streams from `production-home/current` and advances the
  control's `backup-ack.json` (`backup-lag` -1), so the unchanged staged driver waits for
  streams and never writes its own; `run_control.sh` relaunches the driver after failures
  (a checkpoint screen can take the GPU between steps); `screen.sh` runs the greedy and
  sampled DEV screens and the panel on an export; `plateau.py` applies the plateau rule
  (overall from the screen, sampled from the sampled screen, agentic and tool-call strata from
  the panel). End-to-end test (2-stage mini candidate on the 5080 with pull, feeder and
  control sharing the GPU): streams byte-identical to the candidate's, intents correct,
  200/400 updates, screens and plateau decision produced. The anneal after a plateau is a
  single `continue` intent built by hand when needed.
- Gate verdict v2 (`7b0b2b7b`): every arm must cover each primary and guardrail scenario's
  exact roster from the gate manifest (sampled seeds 9201–9203); unpaired rows fail instead of
  crashing; the drift margin is two-sided; the release passes only with both comparators, a
  drift run and a lead above the margin; held-out transfer completeness is reported. 11 tests.
- NVFP4 sampled qualification on runtime v10 passed
  (`artifacts/shortlist-only-table-01/release-final/nvfp4-verifier-v10/`); a served 24-request
  sampled slice on v10 ran distribution-qualified, all outputs correct.
- Shortlist tool (`artifacts/rental/shortlist/`): `label_mixture.py` turns the home copies of
  completed stages' fused manifests into a label mixture (rental paths remapped, every label
  file hash-checked); `sweep.py` builds shortlist sizes with `vocabulary-variants`. Tested on
  the 1,000 5090 speed-test answers: 25,731 distinct labels, 222 MB shortlist-only file. The
  multilingual-light comparison needs the trained candidate.
- Drafts: `tasks/release-retrain-drafts/` (model card, data card with source counts, licences
  and attribution including SWE-rebench); bank-v2 provenance supplement in
  `artifacts/bank/retrain-01/provenance/`.
- Source identity changed after Step 2 only by the verdict calculator, which the production
  path does not import. Test outputs (34 GB) deleted; compact results kept beside the scripts.

Owner amendment (2026-11-23, before training): the 4-bit control's plateau checkpoints move from
2K, 4K, 8K, 16K... to stage-aligned doublings of 8,250 updates (8,250, 16,500, 33,000, 66,000,
...), so same-prefix comparisons use candidate stage exports production already keeps. All pass
and fail rules are unchanged; `release-gate-v1.json` keeps its frozen bytes and this amendment
governs `control_plateau_rule.checkpoints`. Final Astra readiness check (2026-11-23): not ready;
confirmed blockers are being fixed (see below); the owner asked not to rerun the rehearsal yet
and has not set the cap.

Pre-launch fixes (2026-11-23): bank-v2 (`artifacts/bank/retrain-01/bank-v2`) admits by user-text
fingerprint too; exactly the 6 protected rows were replaced (by GSM8K and Orca-Math rows), 0
protected matches against all four indexes, no other row or role changed; `production.json` uses
it with checkpoints every 4,125 updates. The driver restarts a stage from the previous stage when
a resume is refused, retires the previous state only once the current stage's state exists,
retires set-aside states after success, and verifies the final state binary. `pull.sh` keeps a
verified last-good generation, compares specification content, refuses a host behind home,
pulls only completed stages' final states (latest two kept), uses timeouts and acknowledges after
verification; two bugs (byte comparison of the specification, an empty-glob error) were found by
the rehearsal and fixed. `setup.sh` enforces the pinned uv, library linkage, home-written source
identity, the initialization digest and the runbook's identity thresholds. `watchdog.sh` (stop
call verified against the API) handles restarts, holds, completion, cap and credit. Host-loss
rehearsal (`hostloss.sh`, 3 stages × 64 rows): hold after stage 0; host lost during stage 1's
replay and restored from home (stage 0 verified); host lost mid-training in stage 2 and restored
(stage 1 verified); everything killed mid-training on the same host; final 200/400/600 updates,
final state verified, home verified through stage 2: passed. The refused-resume fallback was not
triggered (the same-host resume succeeded), so it remains exercised only by inspection.

Protected-content finding (pre-launch Astra review, verified 2026-11-23): bank admission compared
row ids, groups, source keys and aliases against the exclusion indexes, but not prompt text, and
the GSM8K adapter mints new content-derived ids. Hashing each bank row's user text (raw and
whitespace-normalized, the receipts' scheme) against the opaque indexes finds 6 TRAIN rows: 3
heldout24 prompts and 3 more from the broader development/final/sealed index; 0 DEV rows; 0 gate
matches. No generated shard anywhere in `artifacts/` contains any of the 6, so nothing protected
was ever written, replayed or trained (the pass, long, spending, scaling and speed-test slices
are clean). Astra's count of 239 TRAIN, 9 DEV and 2 pass-test rows did not reproduce with its
own script; recorded as unverified. Required before the rental: text-fingerprint admission in
`retrain_bank` with regression tests, a bank without the 6 rows, regenerated specs and a re-upload
of the bank.

Pass test (2026-11-23, owner-requested; `artifacts/datasets/retrain-01/long/arms/pass*`): on the
same 1,000 fresh records, one pass (8,429 updates) against production's 1.55 passes (13,083),
each with its own cosine, seeds 8101 and 8102: greedy 51.3/50.5% against 51.0/50.3%, sampled
48.2/48.4% against 48.1/48.5%; pooled one pass minus 1.55 passes +0.22 (−0.32, +0.77) greedy and
+0.01 (−0.38, +0.41) sampled. By the preregistered rule (keep 1.55 only if it wins by at least 1
point) production uses one pass: 100 stages × 8,250 updates = 825,000 (`production.json`;
the 1.25M spec is kept as `production-1250k.json`). Saves about 12 rented hours.

5090 speed test (2026-11-23, Vast.ai m:140586, Minnesota, $0.724/h; `artifacts/rental/
speedtest-home/`): writer 839 generated tokens/s at 32 slots (823 at 64 slots and batch 1,024,
not adopted), replay 0.35 s/record, fusing about 0.14 s/record, training 10.17 updates/s (1.92×
the 5080), 4,138 update groups per pass over 500 rows. Projection per 1,000-row stage: about 38
minutes at one pass, 45 at 1.55 passes; 100 stages about 63 or 75 hours, about $49 or $58 at
about $0.78/h all-in. Setup lessons: the plain `ubuntu:26.04` image needs `libgomp1` (now in
`setup.sh`); the 5.8 GB initialization is rebuilt on the host by `prepare-initialization` in 35 s
and matched its SHA256 exactly, so it is not uploaded; the owner's uplink is about 3–4 MB/s.
The host holds every rental input (all manifest files verified). Cross-GPU check queued
(`artifacts/rental/crossgpu.sh`): the same 500 answers replayed and trained on the 5080 with the
same seed and updates, both exports screened greedy and sampled.

Speedup pilots (2026-11-23, `artifacts/datasets/retrain-01/speedups/`): telemetry off gives a
byte-identical export and losses with 3.0% faster training (adopted in `production.json`);
replay overlap reproduces all 128 identity-fixture tap files byte for byte at about 0.39 s per
record against 0.47 (adopted, in code); writer batch 1,024 against 512 at 32 slots: 666 against
652 tokens/s, within noise and changing answers (not adopted); 48 slots need 13.8 GB of KV
cache and do not fit the 5080, so slot counts are measured on the 5090 speed test
(`artifacts/rental/speedtest.sh`: 32 and 64 slots, then one 500-row stage at 1,500 updates, no
DEV diagnostics; `setup.sh` installs, bundles CUDA 13.1 libraries if needed and checks replay
identity; `manifest-speedtest.txt`, 13.7 GB). Pass test: one pass s1 51.3% greedy, 48.2% sampled
(against 50.8%/48.4% for the 1.66-pass control); the remaining three arms resume after the speed
test is set up.

Local rental rehearsals (2026-11-23, retained in `artifacts/rental/`): crash recovery
(`rehearsal.sh`) passed after the driver learned to wait for a busy GPU; the representative
1,000-row stage completed in 4,151 s (generation 1,035, replay 481, fusing 153, training 2,359 at
5.30 updates/s, setup about 120). Forecast: about 115 local hours for 100 stages; 72–115 rented
hours at an unmeasured 1.0–1.6× 5090 speed-up, about $30–70 at $0.40–0.60 per hour.

First Astra review (2026-11-22): its four blockers (stage order, disk, resume, runbook) and
should-fix items are fixed; details are in the commit history and the runbook above.
