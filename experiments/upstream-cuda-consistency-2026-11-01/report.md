# CUDA speculative consistency: upstream changes and recovered local workaround

The recovered October 31 workaround is the relevant validated reference for
Qwen3.5 4B/9B Q4_K_M and NVFP4 with F16 KV on RTX 5080. The upstream narrow
changes are possible ways to reduce its overhead, not a missing fix that should
replace it immediately. Neither upstream change establishes equivalence for all
four local model/quantization combinations.

This investigation ran CPU checks and read source and saved evidence only. It
did not execute CUDA, load models, run training, start servers, change the main
checkout or its submodule, push, or spawn agents. The machine is local RTX 5080
WSL, hostname `DESKTOP-E78LJSI`, as corrected by the human. Root retains sole GPU
execution ownership. Historical Qwen3 F16 training is a separate track.

## Recovered evidence changes the recommendation

The preserved [conservative patch](/home/philip/binary-eagle-decoding/runs/spec-cuda-fix-20261031/cuda-decode-consistent.patch)
is 4,488 bytes, SHA-256
`efc28b921fbdf27bfb072567b078785ef70e1496b74f24658b480106ddcb2c37`.
Its environment switch is `GGML_CUDA_DECODE_CONSISTENT=32` in the recorded
validation. Read it alongside the
[existing audit](/home/philip/binary-eagle-decoding/runs/5080-broader-training-20261031/existing-cuda-fix-audit.json),
[library proof](/home/philip/binary-eagle-decoding/runs/spec-cuda-fix-20261031/live-library-proof.json),
and [validation source](/home/philip/binary-eagle-decoding/runs/spec-cuda-fix-20261031/validate-patched.py).

I independently re-read the underlying request records and regression logs:

- All 128 saved speculative-versus-no-spec request comparisons match token IDs,
  text, stopping reason and generated-token count. The two no-spec repetitions
  agree for every prompt. All saved `timings.cache_n` values are zero.
- The saved native regression logs contain 2,040 comparisons, zero greedy flips,
  and reported maximum absolute logit difference zero. This is a recomputation
  of historical logs, not a new GPU run or a bitwise comparison of saved arrays.
- Thirteen evidence-file hashes match: eight request files, four regression
  logs, and the CUDA shared library named by the historical process-map proof.
  This verifies the retained library bytes, not a currently running process.
- Applying the retained patch to the unpatched source directory identified by
  its build cache reproduces all five touched files in `patched-source/` exactly.

The four request arms are 4B Q4_K_M + MTP, 4B NVFP4 + MTP, 9B Q4_K_M +
DFlash Q4_K_M, and 9B NVFP4 + DFlash Q4_K_M. All use F16 K/V, context 4,096,
batch/ubatch 512, concurrency one and greedy sampling. MTP uses draft maximum
three; DFlash uses sixteen. There are sixteen prompts and two repetitions per
arm. The regression source covers prefixes 24/255/1264, verification widths
2/4/8/16, partial/full retention, and three continuation steps after rejection.
Its assertion permits drift up to 1e-5; the recorded outcome was zero.

The request screening uses two repetitions and no alternating arm order, so it
does not establish a performance acceptance result. Both compared arms use the
patched backend. Their agreement does not prove equality to an unmodified
backend's normal decoding. Limit 32 is configuration, not proof that every
width through 32 or arbitrary context/precision/concurrency has passed.

The later [session closure](/home/philip/binary-eagle-decoding/runs/5080-broader-training-20261031/SESSION-CLOSED.json)
records `throughput_goal_met=false`; the
[analysis addendum](/home/philip/binary-eagle-decoding/runs/5080-broader-training-20261031/native-analysis-addendum02.json)
preserves its existing target/verifier, precision and numerical source. Those
records do not authorize modifying the old experiment or restarting training.
This report makes no training or throughput-success claim.

## Source identities and what is already present

The requested baseline is llama.cpp
`5eb221ec53cb4317aef963e3bdec3cc69401eca4`. I read its Git objects from
`/home/philip/binary-eagle-decoding/third_party/llama.cpp`; no checkout was changed.
The assigned worktree's gitlink is pinned there, but its initialized submodule
reported another HEAD and pre-existing staged differences. It was left alone.

The pinned baseline descends from upstream
`6e60f35608ec6918b44a9839c0c433687165f086`. The FA dispatch/common/vector files
and delta-net graph file are unchanged from that upstream revision. The pinned
MMVF/MMVQ/MMQ files add local audit calls but leave the dispatch logic unchanged.
Consequently the immutable upstream links below also identify the relevant
baseline algorithms. Neither Apoze commit nor danbev's overrides are present
in that pinned source.

The recovered October 31 runtime is a different source snapshot, not this pin.
For example its `ggml_cuda_should_use_mmvf` takes an additional `warp_size`
argument and handles shapes unsupported by MMF before the Ampere/Ada ordering.
The recovered patch cannot be treated as a drop-in patch for the pinned source.
I authenticated its five touched source files against its own original source,
but did not authenticate a Git commit for the full recovered runtime snapshot.

The issue-comments snapshot used for filtered discovery is
`/home/philip/binary-eagle-decoding/runs/upstream-consistency-20261101/issue-comments.json`,
SHA-256 `4c7d54d7b26fa6fd428f57259a4571851d31a496ef1019eb4d583844d137a772`.
I fetched the exact referenced commits from GitHub rather than relying on
comments' descriptions or mutable branch names.

## Comparison by mechanism

| Mechanism | Existing conservative workaround | Apoze narrow branch | danbev diagnostic commit |
| --- | --- | --- | --- |
| Dense quantized projections | Remaps each token to an independent channel with `ne[1]=1`, preserving single-column MMVQ arithmetic within the configured limit. Covers Q4_K/Q6_K/NVFP4 when that path is supported. | No MMVQ changes. | Eight warps for multi-column Q4_K/Q6_K in the GB10 table only. |
| Dense F16/F32 projections | Same channel remap selects the one-token MMVF predicate and specialization. | Earlier commit makes the intended Ada+ F16 threshold reachable through width four; F32 remains width three. | No MMVF/F32 changes. |
| F16 KV flash attention | For head size 256, sequence count one and eligible layout, forces VEC within the limit, one query column per block and one KV partition. | Quantized-KV change does not repair the F16-KV VEC/MMA switch. | Forces MMA and a common wide MMA tile for supported small widths. |
| Quantized KV flash attention | Does not extend stock VEC selection beyond its normal quantized-KV threshold. Not validated here. | Extends Ada+ VEC to width four, forces one column, fixes partition count for eligible directly supported quantized K/V. | Overrides the unquantized-KV selection branch; not a general quantized-KV solution. |
| Fusion | Disables CUDA graph fusions globally while the switch is positive. | No fusion change. | No fusion change. |
| Recurrent state / rollback | No new GDN arithmetic; historical native tests exercise state after rejection. Fusion disable also changes which fused paths can run. | No state changes. | Qwen3 full-attention test, no Qwen3.5 recurrent validation. |

The conservative remap runs token channels in parallel in one kernel; it is not
sequential target decoding. It avoids the generic MMVQ width-dependent warp
choice and the MMVQ-to-MMQ crossover. For NVFP4 this means small target
projections use Q8_1 activation staging in MMVQ rather than native FP4 MMQ.
Global fusion disable and one-partition attention can impose substantial costs,
but those costs require measurements on the actual frozen workload.

The matrix guard requires ordinary two-dimensional tensors. It does not cover
MUL_MAT_ID, arbitrary batched/permuted attention multiplications with FA disabled,
or every backend operation. Its attention override has a specific F16-KV,
head-size-256 shape. Generalizing it to MoE, quantized KV or other architectures
would exceed the evidence.

## Exact upstream changes and applicability

### Apoze: two commits, not one

[8767ef1ca8d644080a14e78378704989df16764c](https://github.com/Apoze/llama.cpp/commit/8767ef1ca8d644080a14e78378704989df16764c)
moves the Ada+ F16 MMVF condition ahead of the Ampere-MMA condition. In the
pinned baseline, Ampere capability includes Blackwell, so it returns at width
one before reaching the intended Ada+ width-four case. The reordering has a
source-level justification. It can affect F16 linear tensors and FA-off
attention; it does not make F32 or quantized matrix operations invariant.
BF16's analogous ordering is unchanged by the upstream patch.

[6263445513416ef93fd5d05391299f33c64702fa](https://github.com/Apoze/llama.cpp/commit/6263445513416ef93fd5d05391299f33c64702fa)
then changes three FA files: quantized-KV VEC selection from width two to four;
one query column per block for Ada+ quantized template types; and a fixed
partition count equal to that template's occupancy limit for widths through
four. The partition condition requires neither K nor V to need F16 conversion.
Thus mixed quantized/F16 templates and a missing compiled quantized-FA instance
can escape that condition. `GGML_CUDA_FA_QUANTS` and actual selected templates
matter. Target Q4_K_M/NVFP4 weights alone do not activate these changes.

The [RTX 5080 laptop report](https://github.com/ggml-org/llama.cpp/issues/25618#issuecomment-6082491096)
uses different 27B dense and 35B MoE models and Q4_0 KV. Its faster partition
search, two-column vector choice, MoE token loop and F32 threshold adjustment
are described in a comment, not included in the cited three-file commit. Those
claims are leads; no exact source diff for those additional changes was supplied
there. The report's own width-three draft limit is an applicability boundary,
not a limit on the present project's authorized research effort.

### danbev: useful diagnostics on different Blackwell hardware

The branch resolved to
[0fb12b9b702e1546b1444acf033090401e9ea4d7](https://github.com/danbev/llama.cpp/commit/0fb12b9b702e1546b1444acf033090401e9ea4d7),
parent `276378515ceebee5d14bcc8536603dd5c87491a2`. Its commit explicitly calls
the change diagnostic. It fixes multi-column Q4_K/Q6_K to eight warps only in
the GB10 table and forces an MMA attention instance for supported widths through
eight while bypassing the normal unquantized one-token VEC preference.

[The experiment](https://github.com/ggml-org/llama.cpp/issues/25618#issuecomment-6441719728)
used Qwen3-8B Q4_K_M, DSpark Q8_0, F16 KV and GB10. NVIDIA identifies
[RTX 5080 as compute capability 12.0 and GB10 as 12.1](https://developer.nvidia.com/cuda/gpus).
The pinned source distinguishes those constants and selects separate generic
and GB10 MMVQ tables. The GB10 warp hunk therefore does not act on a normal
SM120 build. For generic SM120 MMVQ the warp count is four for widths 1-4 and
two for 5-8. In addition, current Q4_K switches to MMQ above width five, Q5_K
above six, Q6_K above seven; NVFP4 stays MMVQ through eight.

For Qwen3.5's usual GQA ratio four, the diagnostic wide-tile condition can select
an eight-query-by-four-head MMA tile. This still needs source/shape checks,
including sparse/layout fallbacks and stream-K behavior. Forcing one template
does not itself prove complete model invariance or a performance gain. The
commit's warp change has no NVFP4 case.

### Other evidence followed and objections

[Issue 28111](https://github.com/ggml-org/llama.cpp/issues/28111) reports SM120 IQ
matrix non-invariance within MMVQ at larger small widths. It is not direct
evidence about Q4_K_M or NVFP4.
[PR 28108](https://github.com/ggml-org/llama.cpp/pull/28108), exact head
`1e2f6954442e6369eda1c4c5c30561109e5f3d8a`, is closed and unmerged. It uses a
different four-column FA specialization and an environment override, not Apoze's
fixed one-column/occupancy partition strategy. Its Q8-KV SM120 results also do
not validate the local F16-KV models.

[Issue 27407](https://github.com/ggml-org/llama.cpp/issues/27407) and
[the RTX 3090 warp intervention](https://github.com/ggml-org/llama.cpp/issues/25618#issuecomment-5424885684)
show why a dispatch boundary alone cannot identify the causal operator. The
latter's unchanged output after changing warps does not refute danbev's
two-intervention result on another architecture, nor establish that attention
is always the only remaining site.

[The corrected recurrent-isolation comment](https://github.com/ggml-org/llama.cpp/issues/25618#issuecomment-5392171333)
withdraws its earlier claim that GDN/rollback were the earliest causal operators.
[Issue 31472](https://github.com/ggml-org/llama.cpp/issues/31472) adds CPU/MoE
reports but does not establish a single backend-independent code defect.
Different backends can independently change reduction order with batch shape.
None of these reports supersedes the local source and retained native checks.

## Qwen3.5-specific gaps

The official pinned [4B configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
and [9B configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
both specify 32 layers: 24 recurrent linear-attention layers and eight full
attention layers, head dimension 256, sixteen query heads and four KV heads.
Their hidden widths are 2,560 and 4,096. The actual GGUF tensor types and model
hashes still come from the local experiment, not the official configuration.

The [baseline Qwen3.5 graph](https://github.com/ggml-org/llama.cpp/blob/6e60f35608ec6918b44a9839c0c433687165f086/src/models/qwen35.cpp)
has dense FFNs plus SSM alpha/beta projections and convolution/state handling.
Q4_K_M is a mixed tensor policy, not a promise every weight is Q4_K; native
conversion can retain higher precision for some groups. F32 projections still
leave MMVF at width four without the conservative remap. A whole-model tensor
inventory is needed before omitting that path from an optimized replacement.

The [delta-net graph](https://github.com/ggml-org/llama.cpp/blob/6e60f35608ec6918b44a9839c0c433687165f086/src/models/delta-net-base.cpp)
routes snapshot-preserving speculation through `ggml_gated_delta_net`, records
newest-first snapshot slots, and copies only the written slots to rollback
groups. The [CUDA kernel](https://github.com/ggml-org/llama.cpp/blob/6e60f35608ec6918b44a9839c0c433687165f086/ggml/src/ggml-cuda/gated_delta_net.cu)
already loops sequentially over tokens inside each state shard, with the same
launch shape for a given state size. AR/CH graph labels alone do not prove two
different CUDA algorithms here. State copies, convolution views and partially
rejected verification still require native checks when changing dispatch or
fusion. No source-justified new GDN patch emerged from this investigation.

## Delivered changes and focused checks

The two files in [reference-ports](reference-ports/) are reviewable ports of
the two Apoze commits to the requested pin. They are comparison material, not a
recommendation to layer duplicate dispatch fixes onto the recovered workaround.
Neither reference port is enabled or integrated.

The only newly proposed implementation is
[0003-modern-quant-dispatch-audit.patch](patches/0003-modern-quant-dispatch-audit.patch),
against the pinned baseline. Its existing opt-in logger excluded Q4_K, Q5_K,
Q6_K, MXFP4 and NVFP4. Simply allowing those types would expose another bug:
the MMQ call unconditionally labels activation packing Q8_1, although
[MMQ selects native FP4 packing on Blackwell](https://github.com/ggml-org/llama.cpp/blob/6e60f35608ec6918b44a9839c0c433687165f086/ggml/src/ggml-cuda/mmq.cu).
The patch includes those weight types and derives labels from the same
`use_native_fp4` predicate used by execution. It changes logging only. The
logger still samples selected names (block zero, FC/output), deduplicates records,
and supplies a dispatch inventory rather than execution counts. It does not
instrument FA/GDN or the full model. The recovered runtime has no such local
audit function, so this patch is not a drop-in patch for that newer snapshot.

[check_upstream_cuda_consistency.py](../../scripts/check_upstream_cuda_consistency.py)
reads immutable Git objects, applies the reference ports and audit patch in a
disposable source fixture, and compiles the actual host selection/partition/audit
functions with capability stubs using GCC 15.2 and `-Wall -Wextra -Werror`.
It passed 6,750 dispatch cases (36 expected F16 changes), 1,536 FA partition cases,
twelve native/fallback MMQ label cases, and audit filtering/deduplication checks.
These are host-control-flow tests, not CUDA compilation or numerical kernel tests.

[check_recovered_cuda_evidence.py](../../scripts/check_recovered_cuda_evidence.py)
recomputed the historical checks and source identity described above without
executing any recovered validator, build, server or model.

```sh
python3 scripts/check_upstream_cuda_consistency.py \
  --source-repo /home/philip/binary-eagle-decoding/third_party/llama.cpp
python3 scripts/check_recovered_cuda_evidence.py \
  --run /home/philip/binary-eagle-decoding/runs/spec-cuda-fix-20261031 \
  --audit /home/philip/binary-eagle-decoding/runs/5080-broader-training-20261031/existing-cuda-fix-audit.json
```

All scripts clean up their temporary source fixtures and compiled host binaries.
Small fetched primary-source snapshots remain ignored under this worktree's
`runs/upstream-cuda-source-audit/` for root review; no large artifact was generated.

## Handoff to root

Preserve the recovered conservative build as a separately identified reference.
Authenticate its full source/build/model lineage before porting or relabeling it.
The next useful GPU work is a measured removal of one conservative intervention
at a time, with the same four targets, F16 KV, verifier, prompts and target/draft
precision. Quantized-KV Apoze tests would be a distinct experiment. A GB10-only
warp edit is not a meaningful RTX 5080 candidate.

Check the first differing intermediate tensors and decision changes, plus
snapshot/rejection continuations, before accepting a narrower variant. Compare
its patched no-spec and speculative arms, and separately disclose any change
from the original unmodified no-spec reference. Preserve held-out inputs and
data ancestry. Any speed claim still needs actual dispatch, five or more measured
repetitions initially, alternating comparison order, accepted/proposed counts,
and full-request/decode timing against the frozen references. Q4_0 EAGLE remains
the project's primary drafter comparison; Q4_K_M/NVFP4 here identify target
tracks, not permission to change that acceptance denominator.

No arbitrary GPU allocation or candidate cutoff is proposed. GPU validation of
the narrow reference ports, the new audit patch, larger contexts, other KV
formats, concurrency, stochastic verification, and total throughput remains
root-owned and unperformed in this investigation.
