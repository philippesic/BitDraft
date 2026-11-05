# Bounded binary-method screen

Results and interpretation: `experiments/binary-methods-20261104/README.md`.
These are experiment-specific scripts, not a production backend. The retained
raw run is `runs/binary-methods-20261104-01/`; source snapshots there preserve
exact experiment inputs even after this temporary worktree is removed.

Do not rerun into the frozen artifact paths. For a new screen, choose a unique
run directory and update the `R` constants and supervised run names first.
Check the device pause control and GPU ownership before execution.

- `build.py --sharedlut`: isolated CUDA-TU compile/relink using pinned historical
  source/objects. No model/verifier/target code changes. The default without
  this switch reconstructs the earlier seven-method pilot, not the final runtime.
- `native.patch`, `bmma.cuh`, `lutgemm_exact.cuh`: exact native screening changes.
  `GGML_SCREEN_A1` overrides activation precision inside CUDA only; no deployable
  A1 export is implied. The shared-LUT adapter includes per-call weight transpose.
- `kernels.cu`, `bench.py`: resident synthetic actual-shape phase measurements,
  sampled exact scaled-output checks, three32-operation graph repetitions.
- `lutgemm_build.py`, `lutgemm_bench.py`: published FP16 LUT-GEMM kernel check;
  downloaded source and hash live in the run. Different arithmetic from A8/A1.
- `native.py`: short plain/traced native panel; frozen target tokens and
  within-precision draft-counter checks. Use `--build native-lut` and the nine
  arms from the final freeze receipt. Requests include one warmup per server.
- `analyze.py`: Nsight request-window attribution and trace/plain reconciliation.
- `charts.py`, `report.py`, `plot.py`: report, interactive HTML and matplotlib
  standalone charts. Matplotlib ran in a temporary uv environment, not the
  project's Python environment.

Standalone compile used CUDA13.1 nvcc, `-std=c++17 -O3 -arch=sm_120
--ftz=false --fmad=false -Xcompiler -fPIC -shared`, the existing
`results/cuda-glibc-compat/include` compatibility header directory, and the pinned
native source's `ggml/src/ggml-cuda` include directory. Native compile uses the
retained flags in `native-lut/build.json` including SM120a. Disassembly is retained
for the exact native binary-MMA function as well as the initial standalone probe.

LUT-GEMM derivative copyright/license notices are retained in the header and
`LUT-GEMM-LICENSE`. No third-party training recipes or models were downloaded.
