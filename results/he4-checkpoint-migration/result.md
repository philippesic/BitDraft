# The migrated HE4broad checkpoint reproduces its original next update exactly

## Outcome

The final original HE4broad checkpoint completed 220,095 updates. Import into the new
pipeline preserves trainable parameters, Adam state, Torch/CUDA random state and the
original shuffled sampler position. Running its next original-data update independently
through each pipeline produces identical loss, gradient norm, every trainable parameter
and every optimizer tensor. Loss is 0.12160855531692505; gradient norm is 0.7907398343086243.

The comparison found and fixed an attention-cache derivative error: converting directly
through FP16 also rounded gradients. The retained arithmetic rounds cached values while
passing gradients through unchanged. A CPU regression test distinguishes these behaviors. The older arithmetic fixture used
the generic BlockLayer cache cast, which differs from the actual HE4 ComponentLayer.
Its projection characterization is retained; cache gradients now compare with an
[independently executed HE4 layer](cache-layer-reference.json) using the
[retained driver](executed-cache-reference.txt). The older fixture stays immutable.

## Scope and limitations

Measured on RTX 5080 with Torch 2.14.0+cu130, one original-data update at the final
checkpoint boundary. This establishes migration parity at that boundary, not long-run
training quality, FP16 capture suitability, native export parity or hybrid-data quality.
Python/NumPy legacy random state remains in the original checkpoint; current sampling uses
its own seeded generator and model execution uses Torch state. Historical timing fields
remain in the migration receipt; newly measured clocks must identify their own boundaries.

## Decision

Retain the importer and corrected cache arithmetic. Require separate FP16 capture and
hybrid-dataset qualification before adopting either for continued training.

## Evidence and reproduction

[Compact measured comparison](result.json). Raw checkpoint, original and new drivers,
original next-update state and migration receipts remain under
`/home/philip/binary-eagle-artifacts/migration/` and
`/home/philip/binary-eagle-artifacts/qualification/`. The original frozen source is
`/home/philip/binary-eagle-decoding/runs/he4broad-20261110-01/source-update8h-20261110-01`.
Drivers and exact input identities are retained in the external qualification directory.
Reproduction runs the legacy driver first, then the new driver against the imported state;
GPU ownership must be exclusive. The earlier mismatching attempt remains in its log.
