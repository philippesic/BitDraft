# Training requirements

The CPU worker implements explicit fresh, resume and branch modes. See the
[execution contract](execution.md) for supported schema and command boundaries. Fresh must prove original authenticated initialization and no Adam
history. Resume requires exact parameters, optimizer counters/moments, RNG/sampler,
exposure and lineage; branch retains parent cost/exposure and records changed intent.
Tiny tensor alias and next-update recovery tests pass; actual retained-artifact and
native parity remain separate gates. GGUF export has CPU producer/reader conformance coverage.

The following selection requirements are historical design inputs to the deferred
benchmark redesign; no selection implementation or accepted methodology exists here.
They proposed using only valid complete matched complete-request scores attached to
a durable full state/export. Higher score wins; exact ties retain the earlier state.
DEV/CE/acceptance are diagnostics. A final decline, interrupted panel, invalid output,
nonfinite score or undurable state cannot replace the best. The store alone will own
latest/best publication; the pure foundation selection function publishes nothing.

The historical hour-four DEV peak does not establish a formal serving peak: formal
HE4broad panels ran at zero/eight hours. Larger DEV and periodic serving protocols,
resource/exposure limits and stopping tolerances need separately approved intent.
heldout24 never supplies a training or selection signal. CPU fixtures exercise actual optimizer updates on synthetic tensors. They certify no
full-size allocation, native runtime fidelity, useful learning or serving improvement.
