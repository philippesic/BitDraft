# Five-to-six update-hour plateau investigation

Philip asked whether the acceptance plateau and approximately 1.2 tokens/s drop
indicate a training failure, overfitting, or poor returns from continued training.
This investigation reads existing metadata, updates, recovery receipts, requests
and native results. No GPU evaluation, heldout access, trainer stop or training
recipe change was made. The existing eight-hour allocation remains in effect.

## Findings

- **A real plateau on this consumed DEV3 panel.** Acceptance rises only 46.0064%
  to 46.3141%. Across one pass over the three prompts, accepted tokens rise
  288 to 289 and rounds fall 313 to 312. Code gains seven accepted tokens;
  prose loses four and reasoning loses two. Five greedy timing repetitions
  repeat these exact quality counts. This is mixed per-prompt movement, not
  five independent observations of quality or broad generalization.
- **The small throughput decrease is not a resolved model regression.** HE4broad
  falls 216.667 to 215.472 tokens/s (0.552%). The unchanged external Q4 falls
  248.696 to 246.117 (1.037%), including one slower repetition, which remains
  in all reported results. Target-only rises 184.004 to 184.409 (0.220%).
  Independent bootstrap resampling of the five full repetition bundles per
  hour gives a candidate six/five throughput ratio of 0.99448 with a conditional
  repeat-only 95% percentile interval 0.98587–1.00351. This does not include
  between-hour drift or new-prompt uncertainty, and is not a substitute for a
  randomized back-to-back checkpoint comparison. Control normalization alone
  cannot prove improvement; the Q4 slow repetition also affects that ratio.
- **Aggregate acceptance is not a complete throughput model.** Rounds saved on
  code are offset by extra rounds on prose and reasoning. Costs depend on the
  prompt/context and processing phase; a tiny aggregate acceptance increase
  need not produce a throughput increase. Existing timings cannot separate
  all causal kernel/host effects, so the entire dip is not assigned to noise.
- **No observed training or evaluation discontinuity.** The complete audited
  178,426-update prefix follows the exact seed-8101 global shuffle, has no
  repeated windows, and reaches 15,480 of the 15,647 capture sequences. It uses
  11.38% of the 1,568,435-window pool. All audited losses/gradients/timers are
  finite; constant-rate and exact parameter/optimizer restore receipts pass.
  Mean synchronized update cost is 129.984 ms in hour five and 129.431 ms in
  hour six; the trainer itself has not slowed. The dataset does not revert to
  the old 40-sequence run. Window overlap and semantic correlations remain;
  less than one window epoch does not rule out overfitting.
- **Native controls and saved outputs validate.** Byte identities for target,
  Q4 comparator, native source/parser/runtime/build, cohort inputs, sampling,
  precision, KV policy and warmups match across all twelve formal panels.
  Frozen-source paths differ after localization; comparison uses their identical
  byte hashes and sizes. All 900 saved responses match the fixed target
  token/content/termination signature. All 180 server cells reconcile their
  native counters, request boundaries and no-prompt-cache policy. No new QAT
  versus native numerical parity replay was conducted.
- **No affirmative diagnosis of overfitting.** Fixed teacher-forced DEV loss
  improves 4.68009 to 4.60307, while greedy agreement falls from 42/135 to 37/135
  positions. That tiny, mixed readout is insufficient to diagnose broad
  overfitting. The 15-position decay-weighted training objective and depth-two
  native acceptance are different metrics. Current evidence supports
  diminishing returns on this narrow panel, not an established capacity ceiling.

## Recommendation and limits

Finish the already-bounded eight-hour allocation to obtain the remaining curve
and final recoverable state, then stop unchanged training. This is confirmation,
not an expectation of a sudden throughput win: the sixth-hour candidate remains
about 12.45% below the external Q4 comparator on these prompts. An extension
beyond eight update hours is not supported by these results.

Before a new training allocation or architecture/objective change, compare saved
checkpoints on a broader already-consumed development cohort with TRAIN overlap
checks, keeping target/runtime/control and exposure semantics fixed. The existing
consumed `natural24` route is a candidate for that follow-up, not an acquisition
launched here. Do not open heldout prompts or change the live recipe based on
this three-prompt panel alone. The cause of the precise sub-percent timing change
remains unresolved without a controlled rerun.

[Comparison and timing/quality details](update-plateau-comparison.html) ·
[Machine-readable comparison](update-plateau-comparison.json) ·
[Full exposure/control audit](update-plateau-audit.json).

Reproduce with the CPU-only commands:

```sh
.venv/bin/python -m scripts.he4fresh.inspect_plateau --output update-plateau-audit
.venv/bin/python -m scripts.he4fresh.compare_updates --older 5 --newer 6
```

The earlier [wall-time/input-pipeline investigation](setup-audit.md) remains a
separate historical record; its files were preserved.
