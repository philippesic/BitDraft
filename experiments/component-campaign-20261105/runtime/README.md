# Qualified component runtime

`component.patch` applies to the preserved optimized execution runtime's
`src/models/dflash.cpp`. It adds private/tied embedding admission, one output
allocation for a tied table, F16 scalar tensor conversion to F32 operands and
IEEE half clip metadata decoding. The optimized CUDA library is reused unchanged.

`scripts/component_campaign/build.py::build(new_unique_path, half_clip=True)`
reconstructs this source and links against the pinned original objects. The
immutable original source/objects and exact commands remain under canonical
`runs/component-campaign-20261105-01/runtime04/build.json`. The existing runtime
was not rebuilt during restart recovery. Pure source reconstruction matches its
full source byte for byte; proof is
`runs/component-campaign-20261105-01/qualification/runtime-source-reconstruction.json`.

Native target/verifier uses `GGML_CUDA_DECODE_CONSISTENT=32`; target F16 cache
and external Q4 comparator remain fixed. Half scalar storage does not lower
normalization, softmax, residual or accumulation arithmetic. Cache conversion,
head arithmetic bounds and physical tying have separate qualification receipts.
