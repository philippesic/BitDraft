# Adopt the live baseline run; do not restart

User explicitly requires the training task to remain with a subagent. Root
implemented the missing path after three partial returns, but user corrected
that workflow. Adopt this healthy run and finish native evaluation.

- Canonical resource root: /home/philip/binary-eagle-decoding.
- Live run: runs/dflash-modern-train-20261104-14/artifacts.
- Ownership: runs/dflash-modern-train-20261104-14/ownership.json records boot,
  process startticks, supervisor/worker identities, source pins and stop steps.
- Local tmux socket binary-eagle-runtime; session dflash-modern-train-20261104-14.
- Supervisor scripts/remote_job.py, stop grace120seconds. Authenticate before
  adoption or any signal. Preserve training unless user pause/failure.
- Live source: /home/philip/.codex/worktrees/dflash-baseline-completion; do not
  remove or alter training scripts in that checkout while job is active.
- Worker command: scripts/dflash_baseline/train.py --seconds1800 (actual argv
  in state.json). Snapshot0/300/900/1800 actual update seconds. Full budget is
  required. Do not count capture/export/eval or checkpoint I/O against budget.
- Source Model/trainer/exporter/replay/build/smoke scripts are now in Git.
- Frozen zero: runs/dflash-modern-20261104-02/zero.gguf.
- Actual modern runtime: runs/dflash-modern-native-20261104-03/bin.
  Isolated relink uses the modern packed EAGLE base runtime with only DFlash
  admission and FFN graph changes; target/verifier unchanged. Old CPU patch
  on5eb221ec was insufficient; matched-modern.patch is relevant now.
- Native zero smoke: runs/dflash-modern-native-20261104-03/smoke-result.json;
  updated two-step smoke: smoke-updated-two-result.json in same directory.
  Each exact32-token equality vs target-only and Q4 on one TRAIN prompt.
- Actual update/resume qualification: runs/dflash-modern-resume-20261104-12/
  artifacts/result.json. Second update byte-exact after resume;10.53GB peak
  torch allocation. movement.json records2162468 flips and18 changed scales.
- Data: runs/dflash-modern-capture-20261104-04/data/result.json;24 existing
  TRAIN plus3 consumed DEV sequences, eight fresh taps,1426 training windows.
  Do not relabel DEV as sealed/unseen. All ancestry recorded.
- Numerical limitation: F32 QAT reference differs from native BF16/F16 execution.
  Three contexts38/45 argmax agreement; BF16 input-rounding probe42/45.
  Provisional14/15 per-context gate did NOT pass. Native layerwise diagnostic
  gives1.24e-7 relative FFN error with identical inputs; small floating input
  differences amplify through quantization. Root timeboxed this and proceeded
  provisionally after exact native correctness/update/resume checks. Actual
  native checkpoint eval is required; do not claim numerical parity.

Remaining work: monitor actual progress through1800 update seconds, export
checkpoint300/900/1800 using export_trained.py, native correctness plus matched
Q4/zero/trained full-request throughput, final report/registry/status/goal,
resource release and bounded storage cleanup. Use2warmups/5alternating reps on
consumed DEV, frozen depth2 and target/context/verifier settings. Keep baseline
original manifest unchanged. Actual native metrics, not simulated acceptance,
determine outcomes. A simple benchmark can adapt smoke.py and modern_requests
patterns; generic modern_requests admission is EAGLE-specific, not directly
usable as-is. Native log contains per-request speculative durations and counts.
No concurrent GPU eval while training. No feature/precision sweeps.

Root has not created benchmark script yet. No result is claimed beyond the
qualified steps above. Training completion is not task completion: evaluate it.
Git author absent; command-local Philip Pesic / philippesic06@gmail.com only.
No push and no unpublished submodule gitlink. Preserve all shared resources.
