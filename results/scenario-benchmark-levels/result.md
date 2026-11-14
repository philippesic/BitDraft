# New scenarios tighten the faster claim

The final S4 data-curve HE4 drafter accelerates thinking and sampled requests against
target-only, but trails Baseline on both formats. The three-level protocol exposes these
regressions and requires fresh confirmation before a “faster” claim. No such claim is
established here.

## Outcome

| Target / scenario | HE4 vs target (95%) | HE4 vs Baseline (95%) | HE4 acceptance |
| --- | --- | --- | ---: |
| nvfp4-sampled | 1.241 (1.211, 1.282) | 0.954 (0.942, 0.965) | 42.6% |
| nvfp4-thinking | 1.516 (1.460, 1.579) | 0.881 (0.869, 0.891) | 52.4% |
| q4km-sampled | 1.266 (1.233, 1.310) | 0.954 (0.945, 0.963) | 43.9% |
| q4km-thinking | 1.521 (1.473, 1.576) | 0.875 (0.864, 0.887) | 53.8% |

Ratios above 1 favor HE4; intervals are paired 95% source-group bootstrap intervals.
[Measurements](measurements.md) gives all three arms, acceptance, payload/resident memory, Level
2/3 costs and earlier S4 standings. [Compact evidence](result.json) retains all arms and links
the three compact CSV measurement tables. HE4’s payload is 516.0 MiB; the retained Baseline is
363.8 MiB.

Both thinking sets matched target-only exactly for both drafters. Q4KM reached the 2,048-token
cap on 25/40 requests; NVFP4 on 16/40, identically across arms. Both native sampler
verifications passed. All 120 request/seed pairs in each sampled format also matched target
token/text/stopping signatures for both drafters; equality was observed, not required.

The original 120 diagnostic windows retain exactly the same metrics. Twenty agentic windows from
seven DEV histories and twenty separate thinking windows add 10.6 seconds (2.9 agentic, 7.7
thinking). The combined pass took 20.6 seconds, including 10.0 for the original windows; a
separate original-only pass took 12.1 seconds. The isolated two-update resident pilot measured a
complete interruption of 73.4 s, within the 120-second budget, with both supplementary groups
present. Their added resident cost was 4.5–4.6 seconds per pass.

Earlier protocol-v3 records remain unchanged: their original S4 beats target-only but trails
Baseline in coding, chat/writing, reasoning and long documents on both formats. Its
structured/tool-request gain remains visible. The final `405e326b…` export now has routine
references alongside S4; the campaign screen is unchanged.

## Scope and limitations

Qwen3.5-4B Q4KM/NVFP4, one RTX 5080, runtime-v7, depth 3, cache-ram 0, consistent32 exact
verification, 32768-token context and F16 KV. Thinking selection sorts broader120 coding/
reasoning groups by SHA256(`scenario-levels-v1:` + group), taking 20 per category and the first
six per category for the slice. Thinking is enabled through the chat template; the 2,048 cap
includes reasoning. Capped speed does not establish completed-answer quality.

Sampling uses those requests with thinking disabled, temperature 0.7, top-p 0.9, top-k 0, min-p
0, no penalties, and seeds 8101/8102 (+8103 for Level 3). Greedy proposals with target
sample-and-match preserve the target distribution; fixed-seed equality is not required. The
verifier also checks rejection, which is not this serving policy. All seeds stay inside their
source group in the 2000-draw bootstrap. Intervals cover request variation, not process drift;
candidate-only historical timing receives no new paired interval.

Level 2 costs project fixed slices from full records, without independent startup timing.
Resident memory is whole-device after load, not peak or incremental drafter allocation. The
agentic suite is supplied separately. Missing scenarios, sparse intervals, invalid outputs or an
agentic completion regression prevent a passing verdict.

## Decision

Retain the three levels and additive references. A Level 2 regression beyond noise or direction
disagreement requires Level 3. Every primary scenario needs a speed-interval lower bound >1
against target-only and >=1 against Baseline, with agentic coding decisive, then one successful
fresh confirmation. No escalation trigger is not speed qualification.

The 120-request confirmation bank remains frozen and unexecuted: 20 each coding, chat/writing,
reasoning, documents, thinking and sampled. MBPP sanitized, BIG-Bench Hard, GovReport/LongBench
and authored requests were checked against TRAIN, its expansion, existing DEV and final-state
TRAIN ancestry. This is not a target-pretraining or semantic deduplication claim. Future
admission includes `confirmation/protection-v2.json`. heldout24 and its aliases remain untouched
as final reserve.

## Evidence and reproduction

Artifact root: `/home/philip/BitDraft/artifacts/benchmarks/scenario-levels-01/`. Result JSON
binds measurements, execution/extraction source, inputs and reproduction scripts. The original
final state is `769e184e…`, its serving export `405e326b…`. Maintained entries are `compare`,
`verify-sampler`, `capture`, `record-reference`, `scenario-report` and the training campaign;
run retained scripts from their frozen source.

Shared references remain in `artifacts/benchmarks/references/`; new specs are in
`artifacts/benchmarks/runtime-v7-depth3/specs/scenario-levels-01/`. Old specifications and
canonical first observations remain unchanged. The full local gate passed 138 tests and 92 rule
proofs. Confirmation inputs were validated on CPU and never executed.

## Artifact disposition

Retain workloads, confirmation sources/protection/access receipts and diagnostic captures as
current inputs. Retain reusable-reference panels, verifier/generation receipts, native helpers
and executed source with their dependency closures. Compact measurements and scripts regenerate
reports; initial CPU report revisions are validation history superseded by v2. Historical
records retain their existing disposition. Retired 2 closed, unselected probe states through
Store retention after compact extraction, preserving the original final state and native probe
exports. Exact probe optimizer/RNG resume is unavailable; the original parent and recipe remain
sufficient to repeat qualification. No shared model, bank, reference or historical state was
deleted. New task artifacts occupy 4.27 GB after retirement; observed peak was 14.47 GB.
Conservative compute accounting, including CPU admission, totals 6168.3 seconds (102.8 minutes).
