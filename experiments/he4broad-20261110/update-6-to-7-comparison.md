# Seven-hour follow-up

The on-demand comparison waited for the already-scheduled seven-update-hour
panel, preserving the running job and the original eight-hour cap. No extra GPU
benchmark or new automation was launched. Training then restored the complete
seven-hour own-state checkpoint and continued into its final segment.

| Metric | Six update hours | Seven update hours | Change |
| --- | ---: | ---: | ---: |
| Draft acceptance | 46.3141% | 48.8487% | +2.5346 percentage points |
| HE4broad complete-request tokens/s | 215.4717 | 222.0654 | +3.0601% |
| External Q4 tokens/s | 246.1168 | 248.6768 | +1.0402% |
| Target-only tokens/s | 184.4094 | 184.3509 | −0.0317% |
| Fixed teacher-forced DEV loss | 4.6031 | 4.4508 | lower |

All 75 new requests match target output. Across one deterministic three-prompt
pass, accepted draft tokens rise 289→297: prose +7, reasoning +1, code unchanged.
The seven-hour point shows renewed progress after the small five-to-six-hour
change; it does not establish a hard plateau or a broad overfitting diagnosis.
The candidate remains approximately 10.7% below external Q4 on this panel.

Existing repetition bundles give a conditional repeat-only candidate throughput
ratio interval of 1.0225–1.0382. After normalization to Q4, the analogous interval
is 0.9980–1.0384; do not interpret the raw ratio interval as proof against all
between-hour timing drift. Keep every saved repetition, including the slower
six-hour Q4 repetition. This is still only three consumed DEV prompts, not a
new heldout or broad generalization test. Recommendation remains: finish the
fixed eight-hour allocation, then evaluate retained checkpoints more broadly
before considering a further training allocation.

The fresh CPU audit verifies the exact seeded order of 192,372 distinct windows
with no repeats, constant learning rates and restore receipts, all 975 saved
formal responses, and counters/timing/controls for all 195 formal server cells.
No model, optimizer, training data or active pinned source was edited.

[Interactive comparison](update-6-to-7-comparison.html) ·
[Comparison data](update-6-to-7-comparison.json) ·
[Fresh integrity audit](update-hour7-audit.json).

```sh
.venv/bin/python -m scripts.he4fresh.inspect_plateau --output update-hour7-audit
.venv/bin/python -m scripts.he4fresh.compare_updates --older 6 --newer 7 --output update-6-to-7-comparison --audit update-hour7-audit
```
