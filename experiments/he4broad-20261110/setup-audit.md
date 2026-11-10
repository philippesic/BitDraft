# Early learning-curve audit

Read-only investigation of the apparent flattening after the first training hour.
The trainer, pinned source, recipe, target and evaluation schedule were left
unchanged. No new GPU evaluation or held-out access was performed.

## Findings

- **No small-dataset cycling:** the authenticated index contains 15,647 distinct
  prompt token sequences and 1,568,435 windows. The audited 19,601-update prefix
  follows the exact seed-8101 shuffle, with 19,601 distinct windows, no repeats,
  and 10,879 distinct sequences. At the first hour, 3,662 windows from 3,229
  sequences had been exposed: 0.23% of the window pool and 3.03% of label
  positions. Windows overlap within a sequence; exact prompt uniqueness does
  not establish semantic independence or rule out distribution-specific fitting.
- **No observed optimizer discontinuity:** the constant learning rates and
  complete Adam restoration receipts agree across hourly segments. Immutable
  checkpoints at zero, one and five hours have the expected update counts and
  cursor/order metadata. All five-hour Adam step counters are 18,187. All
  inspected parameters are finite; 99.997% of the 424,673,280 latent binary
  weights remain inside the straight-through gradient support. None of the
  125,952 row scales is zero. This excludes broad clipping-induced freezing,
  not every possible numerical or objective defect.
- **A real input-pipeline inefficiency:** globally shuffled windows use a
  two-sequence memory cache. Almost every next window therefore requires
  reading and hashing another complete capture over SSHFS. Mean input
  materialization/verification cost is about 0.85 seconds per update, versus
  about 0.14 seconds for synchronized model work. This is steady from the
  first hour, not a new execution slowdown after hour one. At the five-hour
  checkpoint, 18,000 training-wall seconds include only 2,521 synchronized
  update seconds (42 minutes). The authorized wall cap includes this input
  time; it is not five hours of optimizer work. The bounded loader solved
  storage capacity but has poor locality and no overlapping prefetch.
- **The evaluation is internally consistent but too narrow for a general
  plateau claim:** six formal panels (zero through five hours) use the same
  three prompts, target, Q4 comparator, source/build, sampling, cache policy,
  two warmups, five repetitions and 256-token cap. All 450 saved responses
  match fixed target token/content/termination signatures across arms and
  checkpoints. All 90 server-cell counter sets independently reconcile;
  measured requests report no prompt-cache reuse and saved timing bounds
  agree. Controls remain stable. Repetitions provide timing evidence, not
  additional prompt diversity. One prompt per domain also makes the reported
  stratified bootstrap intervals unsuitable as generalization intervals.
- **Loss and acceptance measure different behavior:** training optimizes a
  decay-weighted 15-position hard-label objective; native panels propose at
  depth two. Training-loss medians decline, but successive medians use different
  sampled examples. The teacher-forced development readout contains only
  135 positions. These cannot diagnose broad overfitting or demand proportional
  gains in native acceptance. No fresh full QAT/native numerical parity replay
  was performed by this CPU audit.

Rapid initial recovery from quantization damage followed by diminishing returns
is consistent with the curve, but remains an interpretation. The evidence does
not establish an architectural ceiling or justify changing the model recipe.

## Recommended next decision

Before attributing the curve to capacity, qualify a bounded authenticated
prefetch/cache path that preserves the exact window order and recovery cursor;
report both input-inclusive wall time and actual update time. Do not silently
extend this run's allocation. Compare saved checkpoints on a broader development
cohort with TRAIN-exclusion checks (the existing consumed `natural24` cohort is
an available option) rather than opening a new held-out set. These are proposed
follow-ups, not changes made to the live experiment during this investigation.

## Evidence and reproduction

- [Data, exposure, controls and timing audit](setup-audit.json), produced by
  [inspect_plateau.py](../../scripts/he4fresh/inspect_plateau.py) with
  `.venv/bin/python -m scripts.he4fresh.inspect_plateau`.
- [Zero/one/five-hour immutable checkpoint inspection](ste-audit.json) and
  [five-hour scalar/scale inspection](scalar-audit.json), produced on the archive
  host by [inspect_checkpoint_cpu.py](../../scripts/he4fresh/inspect_checkpoint_cpu.py).
  Both compressed and decoded checkpoint hashes match their receipts. The
  helper imports neither Torch nor CUDA; temporary decoded scratch was removed.
- [Diagnostic charts](setup-audit.html), [current campaign report](report.md),
  and [operating procedure](operations.md).
