# Issue and failure analysis

This records what went wrong in the old Binary Eagle repo and its threads, with evidence, root cause where known, and the rule, check, or plan item that now addresses it. Unaddressed items are marked as such. It is an analysis document, not a status diary and not permission to start a later port stage.

## Sources

- Garden Repair problem map, 9 November 2026: https://cursor.com/canvas/store-2a326036-ee6e-460b-a22f-a5bffbf66171/044aff89-ce08-4dfb-90dd-58568b4be2c4
- `/workspace/research/bot-interviews.md`
- `/home/philip/.t3/scratch/2026-11-10-read-only-binary-eagle-porting-bc9c678e/PORTING_PLAN.md` and `SUMMARY.md`
- This repository's `FINDINGS.md` and `docs/decisions/`
- Hook and thread observations from 9 November 2026 in America/Los_Angeles

## Training, data, and selection failures

| Issue | Evidence | Root cause | Addressed by |
| --- | --- | --- | --- |
| Overnight continuation overfitted a tiny dataset while reporting "healthy" | HE4 continuation, 7–8 November 2026: 24 sequences, 1,762 unique labels, about 118 passes in 6 hours. Development loss 7.12 to 8.55 (already 7.53 at 5 minutes). Draft acceptance 22.61% to 17.85%. The agent saw the worsening and kept reporting operational health. The 1,439.532-exposure continuation would fail the proposed exposure guard. | No pre-launch sufficiency print; no scientific stop rule; monitors checked liveness, not learning quality. | Plan C7 and data/exposure validators in stage 0; source-stage trainer stop rules. A real training run still needs Philip's budget and go. |
| Best serving checkpoint was not selected | HE4broad: DEV top-1 peaked at 17.7778% at 14,400 seconds (hour 4) and ended 13.3333% at 28,800 seconds; CE worsened to 8.8991. Formal serving was scored only at 0 and 8 hours. Hourly full states were retained, so the failure is the missing selection policy, not a missing hour-4 archive. The 896.889-exposure HE4broad would fail the proposed exposure guard. Latest consumed-DEV panel: HE4broad 24.57% acceptance, 174.11 tok/s; target-only 185.44; Baseline 65.33%, 249.66 tok/s. This does not show HE4 beating Baseline. | Serving was not the periodic selection objective; smoke used a different shorter workload. | Plan section 9: 30-minute DEV, formal serving at frozen mid-run points, atomic keep-best by valid complete-request throughput. Implemented as contracts in stage 0; behavior lands in the source and hardware stages. |
| A new trainer was copied for each campaign | 659 tracked script files (53 `check_*`, 31 `prepare_*`). Continuation copied the previous route and inherited its dataset assumptions. | Fixes did not carry forward. | One trainer and one benchmark entry in the porting plan. Not ported yet. |
| Data capture silently dropped prompts | Overnight capture watched by Binary Bot reported "still on 4,492" all night against about 10k briefed. A dedup bug on empty `source_topic` dropped 5,505 prompts. Thread self-reports also disagreed with disk (4,120 versus 4,316). 2080 Ti recapture gross counts, reported by Philip and not remotely audited in the plan: 10,150 sequences, 970,121 windows, 1,111,811 labels. The 5,505 versus about 5,497 difference is still an open reconciliation. | No expected-versus-actual gate; the bot accepted the number. | Plan section 5 and P1/P2: exclusion counts, two-null-row proofs, expected-versus-actual as a failing gate. Ledger import is a source-stage item. Admission of the recapture is still an open question. |
| Safety checks broke saved state | DFlash2 smoke attempts: a missing export version plus an optimizer-state aliasing bug rejected 2,100 update-seconds of training. | Qualification and export paths could mutate or reject valid state. | P5: CPU resume without tensor alias mutation; failed write preserves old complete state; later GPU next-update recovery is mandatory. Not yet ported. |

## Repo, docs, and comparison failures

| Issue | Evidence | Root cause | Addressed by |
| --- | --- | --- | --- |
| Run artifacts and stale instructions lived in the repo | 905 GB and 1.355 million files; `runs/` 839–840.967 GiB across about 1,320 folders; 16,670 markdown files on disk, 370 tracked; 299 nested AGENTS, CLAUDE, or SKILL files. One still named Q4_0 EAGLE as primary. | Artifacts and instructions shared the checkout. | Stage 0 repo policy; no tracked runs; one root AGENTS.md. Old artifacts stay in place; the plan authorizes no deletion. |
| Front-door docs were out of date | README described the W1A1 scaffold and an SM75 (2080 Ti) build. `CANDIDATE_STAGES.md` said there was no precision winner yet. `docs/STATUS.md` still quoted November 6 free space near 66 GiB after the disk was far smaller. | Narrative docs were not generated from current evidence. | One FINDINGS.md; STATUS/HISTORY/PLAN/SCRATCH path components are rejected. README in this repo states the names and that HE4 has not been shown to beat Baseline. |
| Mixed, unlabeled comparisons cost about 50 messages | Binary Bot mixed DSpark with DFlash, old host-traced with GPU-only timing, and took a 0.74% training-check match as acceptance. | Results had no required identity (era, clock, workload, metric). | Benchmark comparison key; typed clocks; no `--force`; DSpark versus modern DFlash may appear only as labeled history, with no speed ratio. Runner not built yet. |
| Disk was already too full for the next run | At plan time: 5.3 GiB free in WSL of about 1,007 GB; C: 8.6 GiB then 6.4 GiB on recheck, unexplained. Proposed reserves failed. | Peak storage was not estimated and reserved before launch. | Preflight disk and exposure budgets in the plan. Capacity and backup placement remain an open question. Unaddressed as a machine-ops decision. |

## Agent and infra failures

| Issue | Evidence | Root cause | Addressed by |
| --- | --- | --- | --- |
| Completion webhook returned 401 | The porting-plan thread's completion POST returned HTTP 401 (missing Authorization header). Binary Bot's webhook never fired. | Requests lacked the sender key; the chain was untested. | Later hook install on the four machines; a real Codex Stop later reached EM bots with HTTP 200. Directory-rename path routes are not updated in this change. Unaddressed: any remaining self-POST in thread prompts. |
| A later HE4 thread was cut by a restart | On 9 November 2026 in America/Los_Angeles, the 2080 Ti HE4 capture thread's first run ended failed at a T3 restart; a second run was started. | Infra restart, not a scientific stop. | Unaddressed as a T3/runtime issue. This repo cannot fix the restart. The remaining-stages plan still requires recoverable keep-best and retained failed attempts once training is authorized. |
| Briefs were prohibition lists without a scientific stop | HE4 brief had hashes and "do not" lists, and "do the work; do not wait for more permission," but no learning-quality stop rule. | Purpose lived outside the harness. | Planned training state machine and checkable stop rules. Skills stage is out of scope and has no plan. |
