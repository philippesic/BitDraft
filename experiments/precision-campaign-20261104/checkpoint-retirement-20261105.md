# Superseded checkpoint retirement

The user explicitly authorized freeing space from earlier precision checkpoints
on November 5. Six completed exploratory/intermediate optimizer states were
retired; their exact state can no longer be resumed. This narrowly supersedes
earlier blanket preservation for the paths listed below, not other paid states.

Retired WSL states: `no-gain/checkpoint.pt.zst`,
`adapted/G2D8-fusion/checkpoint.pt.bsz`, and
`adapted/G8D8-attention-vo/checkpoint.pt.bsz`, all under
`runs/precision-campaign-20261104-01/`. Total: 12,968,172,047 bytes.
Their native exports and dictionary bases passed hash and full reconstructed
stream verification before deletion. Measurements, configurations and source
history remain.

Retired Windows states under
`C:/Users/philip/binary-eagle-decoding-artifacts/`:
`precision-campaign-20261104-01-pilot-archives/checkpoint-32.pt.bsz`,
`precision-campaign-20261104-01-pilot-archives/checkpoint-64.pt.bsz`, and
`precision-campaign-20261104-01-control-archives/no-midpoint.pt.zst`.
Total: 13,070,420,636 bytes. The retained final pilot checkpoint passed its full
archive hash check; the no-midpoint native export passed full reconstructed
stream verification. Canonical symlink aliases for retired files were removed.

All selected finalists, current B0 and required parents, vanilla comparison,
current component campaign checkpoint, native exports, dictionary bases,
captures, results and source history remain protected. No host configuration,
unrelated archive transfer, VHD compaction or filesystem trim was changed.

After deletion, both Windows C and WSL reported approximately 14 GiB free.
WSL deletion alone did not release host space; direct Windows retirement did.
The active component checkpoint's earlier I/O error still requires revalidation
before training; freeing space alone does not establish resumability.

Exact identity/deletion receipts:
[WSL retirement](../../runs/precision-campaign-20261104-01/retired-exploratory-checkpoints-20261105.json)
and [Windows retirement](../../runs/precision-campaign-20261104-01/retired-windows-checkpoints-20261105.json).
