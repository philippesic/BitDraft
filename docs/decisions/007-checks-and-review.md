# 007: Checks and review

Date: 2026-11-09. Status: foundation decision; later gates remain pending.

One exact public command runs ten enforcing stages and reports unavailable runtime gates separately. Every one of 125 atomic rules needs its own bad/control proof and exact diagnostic/exit. Missing tools/cases/tests and checker-disable mutations fail. Root AGENTS stays under 80 lines and 4,000 UTF-8 bytes; CLAUDE is exactly @AGENTS.md. A content-bound local verification receipt is unapproved; authenticated Philip review/branch protection stays unwired without a remote.

November 10 source increment: domain validators now live in the standard-library
product contracts module. The tools module is a compatibility re-export; product
code imports no tools. Negative next-token positions now reject before indexing.
The data core adds immutable prompt roles, alias/group separation, explicit training
count/provenance admission, window boundaries and streamed artifact authentication.
The exact pytest dependency exception covers only its new rejection-test module.

The CPU store uses destination-filesystem temporary writes, data/directory fsync,
content authentication and atomic latest-pointer replacement under a nonblocking
writer lock. Injection before pointer publication preserves the parent; after a
successful rename a crash may expose the complete parent or complete child.
This is process/storage-protocol coverage, not proof of power-loss durability.
The worker control loop has injected operational time and explicit update/wall/
attempt caps; failed attempts consume cumulative cost. Its tests use a fake worker
and qualify control behavior only, not HE4 numerical updates or source parity.
All current named CPU scenarios are now in the mandatory roster.

The numerical increment retains strict production typing and isolates tensor-library
calling/type gaps to its test profile. A pure scalar projection oracle caught and
corrected the inclusive sign-gradient boundary at +/-1. CPU subprocesses hide CUDA
because the reused Torch environment otherwise initialized CUDA during an optimizer
test. GGUF fixtures authenticate BF16 initial tensors and an explicit Q4_K vocabulary
basis, check shapes and memory estimates, and quantize frozen projections to Q4_0.
The capture reader preserves the last native EOG label and rejects array/metadata
mismatch, corruption and over-budget materialization. It retains one capture rather
than an eager corpus. These witnesses do not claim full-size source/native parity.

The tensor worker now uses the original grouped AdamW rates, weighted fifteen-label
loss and gradient clipping, with shared gains/clips kept aliased. A streamed,
byte-bounded checkpoint keeps parameters, Adam, RNG, sampler cursor and cumulative
budget counters together. CPU tests prove identical next-update parameters after
resume and preserve the parent archive. An explicit branch resets Adam/cursor while
retaining parent updates/cost and lineage. Same-process rollback charges discarded
updates; abrupt-process accounting still needs the durable run journal integration.
CUDA state handling is implemented but not qualified by these CPU witnesses.
