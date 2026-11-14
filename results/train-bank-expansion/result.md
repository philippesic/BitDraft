# A 2,633-record TRAIN expansion adds 1.72M supervised positions, led by agentic sessions

The frozen 6K TRAIN bank had 200 short tool-harness prompts and no multi-turn tool
sessions, while the drafter's main deployment is long agentic coding. Tranche 1 adds 874
complete target-generated tool sessions, each captured once with every assistant turn
supervised, plus 1,759 coding, writing, conversation, maths and paper prompts with
answers up to 2,048 tokens. It used 194.9 GB of new FP16 captures and 4.9 GPU hours, and
combines with the 6K corpus through one admitted mixture manifest. This establishes
training data readiness, not learning or serving gains.

## Outcome

| Category | Records | Supervised positions | Mean answer | Input bands (<512 / 512–2K / 2K–8K / 8K+) | Record bytes |
| --- | ---: | ---: | ---: | --- | ---: |
| Agentic sessions | 874 | 271,345 | 310 labels per session | 0 / 45 / 829 / 0 | 90.2 GB |
| Other code (CommitPack edits) | 460 | 128,575 | 279 | 460 / 0 / 0 / 0 | 10.4 GB |
| Writing | 500 | 594,130 | 1,187 | 489 / 11 / 0 / 0 | 26.6 GB |
| Conversation | 420 | 377,598 | 898 | 223 / 136 / 61 / 0 | 31.2 GB |
| Reasoning | 300 | 331,005 | 1,102 | 299 / 1 / 0 / 0 | 14.6 GB |
| Documents (QASPER papers) | 79 | 18,332 | 231 | 0 / 0 / 70 / 9 | 22.0 GB |
| **Total** | **2,633** | **1,720,985** | | 1,471 / 193 / 960 / 9 | **194.9 GB** |

Supervised positions are distinct labelled positions; with `up-to-15` each is also one
window anchor. The admitted mixture of the 6K corpus and the three expansion shards has
8,633 records and 4,004,074 windows (the 6K corpus alone: 2,283,089). Writing grows from
8% of the bank's records to 12% and from about 0.33M to 0.92M supervised positions; answers
average 1,187 tokens against 213–1,126 in the 6K bank's capped writing.

**Sessions.** 1,255 TRAIN coding projects were built from CommitPackFT repository groups
that share no repository with the curated pool. The target alone ran each in the live
harness (`tools` system prompt, five file/test tools, greedy, 1,024-token calls, 16-call
limit) with runtime-v6: 856 of 1,255 passed their checks, 854 ended with a final answer
and 401 hit the call limit. Session records average 2,517 tokens (range 1,645–6,801; none
reach 8K because the projects are small) with 12% of positions supervised.

**Multi-span supervision was built and is exact.** One capture per session supervises
every assistant turn instead of one capture per call with repeated prefixes; per-call
capture would have stored about five times the tokens for the same labels. The record
is the template's final rendering, so each turn's context equals what the harness sent
for that call; in the 874 records only 5 of 4,620 turns were dropped because a later message
rewrote their context. 78% of the 4,615 span turns re-render to exactly the generated
tokens; the rest retokenize the same text differently. Across 271,345 span positions, 98.5% of labels equal the next
history token, and every turn's last label is its end-of-turn token. Windows end at each
disagreement, so every window is the capture's own greedy chain. Records stop at the first
assistant turn that repeats the previous one exactly (370 of 1,255 sessions looped,
mostly on `run_tests`).

## Scope and limitations

Labels come from the pinned Q4KM target through a runtime-v6 helper; the 6K corpus was
captured with runtime-v5 helpers. Runtime-v6 matched runtime-v5 outputs on every checked
request, but feature equality between the helpers was not measured. Every record passed
the product loader at capture, and the mixture admitted all records with an 8 GiB CPU
budget; feature arrays were not re-hashed at closeout.

Sessions come from one task family: commit-message change requests in two- to
eight-file Python projects, with compile-and-changed checks rather than real tests, and
only the 4B target's behavior (including its loops). They cover 1.6K–6.8K-token contexts,
not the 8K–16K range. Non-session sources are partly synthetic (UltraChat) or human
(OpenAssistant), with unknown public exposure. Documents are still QASPER papers. The
460 CommitPack edit prompts follow the 6K bank's per-row source licenses, including 42
copyleft-style or unknown ones; sessions and new sources are permissive (MIT, Apache-2.0, BSD,
ISC). Legacy Dolly/Magicoder rows were excluded because they come from the old corpus that
the S4 data-curve mixture samples; names cannot collide with that subset, and mixture
admission would refuse any shared tokenized prompt if both were combined.

## Decision

Retain tranche 1 for training (decided). Train on the expanded bank by pointing a schema-2
intent's `corpus` at `train-expansion-01/mixture.json` with `corpus_format:
corpus-mixture-v1`, as the S4 data-curve intent does for its mixture; the intent, budget
and campaign remain separate allocations. Whether the agentic records improve live
acceptance is unmeasured.

Tranche 2 is proposed, not authorized. After the 314 GB FP32 corpus is retired, about
420 GB would be free; a 300 GB tranche fits with headroom. At tranche 1's measured
per-record sizes and capture rate (132 generated tokens/s):

| Tranche 2 part | Records | Storage | Supervised positions | GPU |
| --- | ---: | ---: | ---: | ---: |
| Already generated sessions (`sessions-2`, unused) | 381 | 39 GB | 0.12M | 0.3 h |
| New long sessions from larger repositories (about 8K tokens, assumed 450 labels each) | 400 | 131 GB | 0.18M | 1.5 h |
| Writing | 800 | 43 GB | 0.95M | 2.0 h |
| Conversation | 600 | 45 GB | 0.54M | 1.1 h |
| Reasoning | 400 | 20 GB | 0.44M | 0.9 h |
| CommitPack edits | 600 | 14 GB | 0.17M | 0.4 h |
| **Total** | **3,181** | **about 290 GB** | **2.4M** | **6.2 h** |

Long sessions need a new multi-file project source: CommitPackFT has no permissive
multi-file groups left, and its single-file groups give shorter sessions. Candidate pools
remain ample (33K UltraChat writing, 97K multi-turn conversations, 7.2K MATH, 3K
CommitPack edits, 350 more QASPER papers).

## Evidence and reproduction

[Structured facts](result.json) authenticate the mixture, admission receipt, banks,
session results, shard corpora and summary under
`/home/philip/BitDraft/artifacts/datasets/train-expansion-01/`. Commands:
`generate-sessions` (`specs/sessions.json`, `specs/sessions-2.json`), `select-expansion`
(`specs/expansion.json`, `specs/expansion-2.json`), `capture` (`specs/capture-{a,b,c}.json`)
and `native-helpers` (`specs/helpers.json`), run from frozen snapshots in
`executed-sources/` by `specs/chain.py`, `chain4.py` and `launch.py`; `chain3.py` was
stopped after its session step and replaced by `chain4.py`, which sized the last session
shard from actual rather than maximum bytes. `specs/summarize.py` recomputes the table and
`specs/mixture.py` the admission. Sources: CommitPackFT Python (`bigcode/commitpackft`
fc56fe33, MIT card), MATH train (`EleutherAI/hendrycks_math` 21a56338, MIT), OpenAssistant
oasst2 train (179dd21f, Apache-2.0), and the authenticated curation-09 pool.

## Artifact disposition

Retain everything under `train-expansion-01/` as current TRAIN inputs: read-only sources,
helpers, specifications, logs, executed sources, both session generations (the 381
unused sessions are tranche-2 input), both banks, the three capture shards, the mixture
and its admission. Nothing was discarded; the 6K corpus, old FP32 corpus and DEV
references were not modified. Exact recapture needs the recorded helpers and runtime-v6.
