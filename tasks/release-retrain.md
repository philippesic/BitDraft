# Release retrain plan

## Goal

Retrain the release drafter from scratch on licence-clean, broader and much larger TRAIN data.
Judge it against standard public benchmarks and against Baseline both as released and fine-tuned
on the same data. Then publish a release whose claims, data card and licences hold up to
outside scrutiny.

## Authorized

Planning only, drafted 2026-11-21 from the owner's request of that day. No data pulls, captures,
builds, training, rentals or benchmark runs are authorized yet. Each workstream below needs an
explicit go. A listed item is a proposal, not a decision.

## Done when

- The owner has decided the open questions under `State`.
- The data card, eval suite and release gate exist and are frozen before the final training run.
- The final run and every comparison arm are measured, closed through close-results and
  recorded in FINDINGS.
- The release ships with a licence, a model card, a data card and reproducible eval commands.

## State

Owner: unassigned (planning thread). Branch: `main`. No artifacts created.

### What the previous train taught us

- **Data is the limit, not the recipe.** S4 flattened after about two passes and then mostly
  memorized (F025, F029). The long fused run was still gaining about 0.3 points per segment at the
  end (F033). TRAIN is 13,394 records. DFlash-class drafters train on about 800K samples, and the
  DFlare follow-up trains on 2.4M. We are about 60 times smaller.
- **The data, not one bit, carried the result.** A 4-bit control trained for 66 minutes matched the
  leader's speed (F034). Its acceptance on ordinary requests was flat between 2,000 and 21,000
  updates. It kept gaining only on agentic sessions, from 57% to 69% first choice. That is the
  owner's observation "about 1% and then it stopped learning". It fits a narrow TRAIN set:
  Baseline already knows general text better than our 13K records can teach it.
- **One-bit kept gaining with length.** The gap to the control narrowed from 10.8 to 7.8 panel points
  between 2,000 and 21,000 updates (F034). Whether much more data closes it is the central
  scientific question of this retrain.
- **One tool-call convention is fragile.** Moving the DEV tool tasks to JSON calls dropped the
  leader's acceptance from 79.7% to 46.7%, while Baseline only fell from 61.8% to 59.9% (F040).
  The agentic advantage came from training on our harness's own format.
- **Our evals are home-grown, reused and partly broken.** Every panel is reused DEV selection data.
  The protected agentic suite completed 0/15 for every arm because of the repeated-assistant rule
  (F040). The owner-policy rule needed merged strata to produce a verdict at all.
- **Licence problems in TRAIN.** MATH (`EleutherAI/hendrycks_math`) is AoPS content. Hugging Face
  disabled it after an AoPS DMCA notice on 2025-01-02. The legacy prompt sources
  (`legacy-freeze-002`, `legacy-gsm8k` and others) have no recorded licence audit. UltraChat is
  MIT-licensed but generated with GPT-3.5, which raises a terms-of-use question to check.
- **Things that worked and carry over:** calibrated start (+7.4 points), 4-slot blocks for depth 3,
  cosine schedule, `baseline-k` frozen tables, fused F16 storage, the vocabulary shortlist and the
  shortlist-only file, the recovered tensor-core kernel, and exact runtime-v10/v11.
- **Comparators any claim must name:** stock Baseline, the 4-bit control, the target's built-in
  draft head (57.9% acceptance, no file, 0.964× Baseline; F037) and the stock 2-bit and 3-bit files.

### Workstream A: scope decisions (before anything else)

1. **Release form.** Choose between one-bit only, also allowing three-valued (1.58-bit) weights, or
   whichever format wins at equal data. This decides which features get serving kernels.
2. **Targets.** Qwen3.5-4B only, or also a larger target (9B or 27B class) where verification
   dominates even more and VRAM savings matter. Q4KM only, or also NVFP4 and BF16.
3. **Which target produces TRAIN features.** The deployed Q4KM in our runtime, as now, gives exact
   features but slow generation. Batched BF16 in vLLM generates far faster but gives slightly
   different tokens. Pilot the mismatch before choosing.
4. **Serving path for users.** Our llama.cpp fork, an upstream llama.cpp PR, or also vLLM/SGLang.
   A release nobody can run outside this repo is weak for the resume goal.
5. **Licence.** Still undecided (`README.md`). The candidate starts from a conversion of Baseline's
   weights, so Baseline's licence and the target's (Qwen, Apache-2.0) constrain it. Check both.
6. **Rental scope.** The current rule is to rent only for a final, already-promising run. The owner
   now offers rentals for speed and VRAM. Decide whether pilots and data generation may rent too.

### Workstream B: production TRAIN data

- **Licence audit first.** Every source gets a licence, revision, attribution duty and verdict in
  the bank definition. Drop MATH, any unaudited legacy source and anything whose terms forbid this
  use. Prompts carry the licence risk. The responses are regenerated by the target, as now.
- **Volume target.** Start at about 150–250K conversations, with the option to grow toward 800K
  if the scaling pilot keeps paying. The final size is set by the pilot curve, not guessed now.
- **Mix (draft shares, to tune by pilot).** Chat and instruction following 20%. Coding (generation,
  editing, explanation, several languages) 20%. Agentic and tool sessions 15%. Maths and reasoning
  15%. Long documents (RAG, summarization, QA up to 32K) 10%. Writing 8%. Multilingual 7%.
  Structured output (JSON, tables, extraction) 5%. Multilingual is absent today, even though Qwen
  is strongly multilingual.
- **Candidate permissive sources, each to verify.** Chat: Open-PerfectBlend (used by NVIDIA's DFlash
  recipe), Nemotron post-training sets, oasst2, WildChat (ODC-BY). Maths: GSM8K train (MIT),
  Orca-Math, and synthetic problems written by an open Apache model. Code: CommitPackFT,
  OSS-Instruct-style sets under permissive licences, plus our own repository tasks. Long documents:
  QASPER (CC-BY, needs attribution), GovReport-like public-domain text. Tools: Glaive, xLAM/APIGen,
  ToolACE, Hermes function calling, each checked for licence.
- **Tool-call diversity.** At least four conventions: Qwen native XML, JSON inside the native
  delimiters, OpenAI-style function JSON, and Markdown/diff-style agents (aider-like edits). Use
  several agent harnesses and system prompts, MCP-style tool schemas, and parallel and multi-turn
  calls. Hold one convention out entirely as an evaluation-only transfer test.
- **Generation settings.** Thinking on and off; greedy and sampled (temperature 0.6–1.0) responses,
  because sampled is our weakest scenario. Regenerate every assistant turn of multi-turn sessions
  with the target.
- **Soft targets (new).** Store or compute the target's top-k probabilities and add a KL term
  beside hard-label cross-entropy. This learns from the target, not from a full-precision drafter
  teacher, so it differs from the rejected teacher idea. It should help sampled decoding most.
  Screen it first.
- **Hygiene.** Exact and near-duplicate removal (MinHash), decontamination against every public eval
  set in workstream D, the existing heldout24 and confirmation exclusions, and per-source caps.
- **Data card.** Sources, licences, attributions, shares, filters and the generation recipe, shipped
  with the release.

### Workstream C: pipeline and compute

- **Storage blocks stored features.** Fused records average about 10.6 MB, so 200K records would be
  about 2 TB. Only 291 GB is free (529 GB in `artifacts/`). Options:
  (a) online capture, where the target runs beside the trainer and computes features on the fly.
  This needs about 10 GB of extra VRAM for a BF16 4B target, which is easy on a rented 48–80 GB
  card. It is the standard approach in NVIDIA's DFlash recipe and in SpecForge.
  (b) store token IDs only and rebuild features with the existing token replay;
  (c) rent storage. Recommended: (a) for the final run, plus (b) for reproducibility.
- **Generation throughput.** About 150M response tokens at 200K conversations. That is a few hours
  of batched generation on one rented H100, and impractical through single-stream llama.cpp on the 5080.
- **Trainer portability.** Check that the trainer, receipts and authentication run on a rented Linux
  GPU. Decide whether multi-GPU data-parallel training is worth building, after a one-GPU
  throughput measurement on the rented card.
- **Free disk first.** Close and retire obsolete states and captures within authorized scope
  (`train-final-01-fused` duplicates content held in `train-final-02-fused`, for example). Do this
  only after the owner approves the list.

### Workstream D: evals and the release gate

Three separate tiers, each with its own data role:

1. **Training screens (private DEV, rebuilt).** New DEV drawn from the new mix's own sources but
   held out by group, with every scenario present, including multilingual and each tool convention.
   This tier is for selection only.
2. **Standard public suite (release gate).** The sets the DFlash and EAGLE-3 papers report:
   GSM8K, MATH-500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca. Spec-Bench adds
   translation, summarization, QA and RAG. Add BFCL prompts for tool calls under several
   conventions, a long-context set (LongBench subset), and SWE-bench Verified or Terminal-Bench
   trajectories for agentic work. Report mean accepted length per round (τ, the field's headline
   number) beside acceptance and speed-up, at temperature 0 and 1. Results then compare directly
   with published numbers. Decontaminate TRAIN against all of them. Run the gate only on frozen
   finalists, never during selection.
3. **Final reserve.** heldout24 stays untouched, as always. A fresh source-disjoint confirmation set
   is optional once the public suite is the gate.

Gate design (proposal):

- Keep our rigor and drop our custom prompt sets. Use paired source-group 95% intervals per
  benchmark, exact greedy outputs against target-only, and sampler validity.
- Pass rule: speed-up against Baseline has a lower bound of at least 1.00 on most categories, and no
  category falls below a stated tolerance. The owner fixes the numbers before the first run.
- **Agentic speed without a broken harness.** Greedy outputs are exact, so record target-only
  agent trajectories once with a standard harness and replay them as multi-turn requests for every
  arm. Task success becomes irrelevant to drafter speed. Only time and acceptance matter.
- **Deployment breadth.** Measure batch sizes 1, 4 and 16 (speculative gains shrink with batch),
  several context lengths, and at least one other GPU (a 4090 and an H100 on rental; Apple silicon
  if the Mac work opens).

### Workstream E: comparison arms

Every claim reports these arms on the same suite, runtime and depth:

- target-only, the speed-up denominator;
- stock Baseline;
- **4-bit control at matched data and length**: Baseline fine-tuned on the new TRAIN for the same
  updates as the candidate. Also record its learning curve to test the "stops after 1%" hypothesis.
  If it keeps gaining on broad data, the old plateau was a data effect.
- 4-bit control with the Q4_K shortlist table (open since F034);
- the target's built-in draft head;
- stock 2-bit and 3-bit Baseline files (size claims);
- the old leader `ef068eba…`, to show what the retrain bought.

Split every gain into the part only low precision earns and the part Baseline would also get
(differentiator rule).

### Workstream F: model features to re-screen at the new scale

Screen in short matched runs on a pilot slice of the new data. Promote only what survives:

- three-valued weights (+4.2 DEV survival; needs a serving kernel);
- a second one-bit plane, or higher precision for salient weights only (+3.3 for the full plane);
- an online Hadamard rotation before 4-bit input rounding, aimed at the largest loss at
  initialization (60.8% to 41.3%);
- trained attention and fusion tables with a Q4_K export (item 10);
- vocabulary-table training (item 7);
- soft-target KL (workstream B);
- the two-channel bypass, re-tested at scale (+0.58 at 21,000 updates, below the threshold);
- a deeper or wider one-bit body, which spends the size headroom on capacity;
- deeper serving depth if acceptance rises (recalibrate with `calibrate-depth`);
- 32K training context for long documents and agent sessions.

Previously dropped, not proposed again: zero-skipping and slot-reuse kernels, pre-summed lookup
tables on GPU, activations below 4 bits, Q2_K/Q3_K vocabulary tables on this runtime, and the
full-precision drafter teacher (owner, 2026-11-16).

### Workstream G: release packaging

Licence, model card with the claim wording, data card, both file forms, runtime build or upstream
PR, reproducible eval commands, and the compact evidence behind every number.

### Preflight for the costly steps

- **Scaling pilot (under 10 GPU minutes per arm, local).** Train one-bit and the 4-bit control on
  1× and 4× slices of TRAIN at equal updates. It must show that more distinct data raises DEV
  acceptance at fixed updates, and whether the one-bit gap narrows. If neither moves, a bigger
  corpus is not justified and the plan stops there.
- **Generation-mismatch pilot.** Compare token agreement between BF16 vLLM and Q4KM runtime
  responses on about 200 prompts. It decides question A3.
- **Rental throughput pilot (about 1 hour rented).** Measure updates per second for online capture
  on one H100 against the 5080. It sets the final run's cost.
- Rough final cost, to firm up after the pilots: 200K conversations at two passes is about 15 times
  the last run's 23.6 GPU hours on the 5080. Divide by the measured rental speed-up. Add data
  generation (a few H100 hours) and both matched arms.

## Next

The owner answers A1, A2, A6 and the gate question. With a go, the licence audit and the scaling
pilot can start first: both are cheap and run locally.
