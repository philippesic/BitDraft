# Recover historical HE4 TRAIN captures

Authorized by Philip November 9, 2026 at 12:50 PM PT. Capture only on DESKTOP-T4JKGJB RTX 2080 Ti; no training or 5080 use.

The overnight baseline remains immutable at `~/he4-capture-2080ti/manifest.json`: 10,150 sequences, 970,121 windows, 1,111,811 unique label positions (176 modern reserved, 4,316 historical, 5,658 broader authored).

`prepare.py` now drops null identity values as well as empty strings. A fresh replay admits 9,813 historical TRAIN rows and the same 176 modern reserved rows. The finite supplement contains exactly 5,497 newly admitted historical rows absent from the original pool and captured IDs. Exclusions record 12 protected identity matches, 31 real duplicate identity matches in fixed replay, 144 non-TRAIN roles, 16 previously used modern prompts, and 4,492 original-pool rows. The fixed replay's extra 8 duplicate matches are encountered after newly admitted rows; the original audit's 23 real collisions remain excluded.

Run `venv/bin/python project/scripts/he4_capture/run_supplement.py` from `~/he4-capture-2080ti`. The runner reuses the original `run_resume.py` command environment and verification code, with a finite pool, per-sequence retry, durable receipt adoption, a process lock, and no deadline. It writes only under `supplemental/`, including live status, supplemental/combined manifests, final audit, and terminal webhook receipt. A failed length/tokenizer check is explicit; capture or verification failure after three attempts terminates as failed. An external supervisor restarts a process that exits without completion. No original captures are removed or reordered.

Preflight: 509 GiB available, GPU idle. Teacher SHA-256 is checked at launch. Contract: taps [2,6,10,14,18,22,26,30], zero_based_pre_block_input_all_tokens, greedy max 128, consistent 32, KV F16, full-logit argmax verification then scratch deletion, stop at first EOG. Final counts come from verified receipts. The final audit checks sample tensor hashes and arrays, all receipt hashes/counts, no duplicate IDs, and no full-logit scratch.

## November 9, 6:10 PM PT checkpoint

The chat interruption stopped the initial execution after 3,952 verified sequences, 430,682 windows, and 485,994 unique label positions. No length/tokenizer failures. Restarted from receipts without losing completed sequences, under detached local tmux socket `he4-supplement`, session `historical-supplement`. This session runs `supervise_supplement.py`, with each attempt isolated/cleaned by `remote_job.py`. Progress resumed past 3,952; 449 GiB remained free. An earlier injected runner SIGKILL also demonstrated automatic restart after 27 sequences.

The running supervisor preserves the finite 5,497-row supplement and performs terminal audits and webhook POST itself. `supplemental/status.json` is the live status. The original webhook had returned HTTP 401 Unauthorized; terminal delivery for this run must be checked explicitly. Original `run_resume.py` is included unchanged as the reused overnight verifier/environment dependency. All 5,497 supplemental rendered prompt hashes were independently checked against all 10,150 baseline prompt files, with zero overlap; details are in `supplemental/admission-audit.json`.

## Final result — November 9, 8:18 PM PT

Capture complete: all 5,497 supplemental rows verified; zero length/tokenizer failures. Added 598,314 windows and 675,243 unique label positions. Combined totals are 15,647 sequences, 1,568,435 windows, and 1,787,054 unique label positions. Source counts: modern reserved 176 sequences / 19,283 windows / 21,744 labels; historical 9,813 / 1,029,982 / 1,166,928; broader authored 5,658 / 519,170 / 598,382. Labels mean distinct sequence/position labels, not distinct vocabulary token IDs.

All receipt hashes and count totals passed. Sampled feature tensor hashes passed. All 5,497 token arrays were checked for metadata equality and no post-EOG inputs or labels: 722 sequences stopped at first EOG; 4,775 hit the 128-token cap. The baseline manifest is byte-identical, supplemental/baseline IDs and rendered prompt hashes are disjoint, and no supplemental full-logit scratch remains. Model pin and 2080 Ti identity passed preflight on each launch. The supervisor exited with code 0, its tmux server exited, and no capture GPU process remains. Disk free: approximately 426 GiB.

Final outputs under `~/he4-capture-2080ti/supplemental/`: `captures/`, `manifest.json`, `combined-manifest.json`, both manifest SHA-256 files, `prompt-pool.json` with exclusions, `admission-audit.json`, `audit.json`, `final-audit.json`, `all-token-array-audit.json`, `status.json`, and `completion.json`. Compact computed evidence is tracked in `experiments/he4-historical-supplement.json`.

Notification blocker: the requested terminal webhook POST was attempted six times and rejected with HTTP 401 Unauthorized, matching the overnight run's failure. Capture remains complete. Delivery cannot be claimed; the exact error is in `supplemental/webhook-error.json`. No credentials were supplied or invented.
