# Recover historical HE4 TRAIN captures

Authorized by Philip November 9, 2026 at 12:50 PM PT. Capture only on DESKTOP-T4JKGJB RTX 2080 Ti; no training or 5080 use.

The overnight baseline remains immutable at `~/he4-capture-2080ti/manifest.json`: 10,150 sequences, 970,121 windows, 1,111,811 unique label positions (176 modern reserved, 4,316 historical, 5,658 broader authored).

`prepare.py` now drops null identity values as well as empty strings. A fresh replay admits 9,813 historical TRAIN rows and the same 176 modern reserved rows. The finite supplement contains exactly 5,497 newly admitted historical rows absent from the original pool and captured IDs. Exclusions record 12 protected identity matches, 31 real duplicate identity matches in fixed replay, 144 non-TRAIN roles, 16 previously used modern prompts, and 4,492 original-pool rows. The fixed replay's extra 8 duplicate matches are encountered after newly admitted rows; the original audit's 23 real collisions remain excluded.

Run `venv/bin/python project/scripts/he4_capture/run_supplement.py` from `~/he4-capture-2080ti`. The runner reuses the original `run_resume.py` command environment and verification code, with a finite pool, per-sequence retry, durable receipt adoption, a process lock, and no deadline. It writes only under `supplemental/`, including live status, supplemental/combined manifests, final audit, and terminal webhook receipt. A failed length/tokenizer check is explicit; capture or verification failure after three attempts terminates as failed. An external supervisor restarts a process that exits without completion. No original captures are removed or reordered.

Preflight: 509 GiB available, GPU idle. Teacher SHA-256 is checked at launch. Contract: taps [2,6,10,14,18,22,26,30], zero_based_pre_block_input_all_tokens, greedy max 128, consistent 32, KV F16, full-logit argmax verification then scratch deletion, stop at first EOG. Final counts come from verified receipts. The final audit checks sample tensor hashes and arrays, all receipt hashes/counts, no duplicate IDs, and no full-logit scratch.
