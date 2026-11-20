# The leader's file without vocabulary rows is 185 MB, half of Baseline's, with identical drafts

The leader's small serving file ([shortlist-only table](../shortlist-only-table/result.md))
still shipped the 68,728 shortlisted Q4_K vocabulary rows, 99 MB of its 284 MB. Runtime-v10
already rebuilds any other row exactly from the target's Q6_K table, so test-list item 12
asked whether the file can ship no rows at all.

It can:
- the file is 185.1 MB, 51.5% smaller than Baseline's 381.5 MB and 34.8% smaller than the
  small file;
- the rows the server builds at start-up equal the shipped rows byte for byte;
- every draft is identical to the small file's on the 60-request screen and 8 pilot tasks;
- start-up takes 6.5 seconds against 0.85, and GPU memory and speed do not change.

The saving is file size only. The rows still occupy GPU memory.

## Outcome

### File

| File | Bytes |
| --- | ---: |
| Baseline | 381,456,064 |
| Leader, shortlist rows only (item 11) | 284,037,024 |
| Leader, no vocabulary rows | 185,068,704 |

The new file is the leader without `output.weight` and without `token_embd.shortlist`. It
keeps the shortlist token IDs, the rows-only marker and the provenance keys, and adds
`dflash.vocab_shortlist.derived_type` (Q4_K). The same all-rows proof admits it.

### Row identity and parity

Runtime-v11 is runtime-v10 plus the edit `vocabulary_shortlist_derived_rows_v1`. At start-up
it converts each shortlisted row from the target's Q6_K `token_embd` with the conversion
runtime-v10 uses for other rows, and logs a hash of the result.

- The hash of the 68,728 built rows equals the hash of the rows shipped in item 11's file,
  in both runs.
- 60-request screen, greedy, depth 3: the new file on runtime-v11 gives the same tokens,
  drafts proposed (6,974) and drafts accepted (4,167) per request as the small file on
  runtime-v10. Target and the small file are identical on v11 and v10.
- 8 pilot tasks: target, Baseline, Baseline with the shortlist and the small file are
  identical on v11 and v10; the new file equals the small file.

### Start-up, memory and speed on the screen

| | Small file | No vocabulary rows |
| --- | ---: | ---: |
| Start-up to "model loaded" | 0.85 s | 6.48 s |
| Logged row conversion | none | 5.56 s |
| GPU memory after start | 5,977 MiB | 5,977 MiB |
| Tokens/s | 244.0 | 244.1 |
| Acceptance | 59.75% | 59.75% |

The conversion runs on one thread at every start; nothing is cached to disk.

## Scope and limitations

- **Screen and pilot only.** The 120-request panel and the five release scenarios were not
  run. Drafts are computed from byte-identical rows, so no difference is expected there.
- **One arm order.** Start-up, memory and speed are descriptive, without intervals.
- **File size only.** GPU memory is the same as the small file's.
- **Start-up cost is paid at every start.** A thread pool or a disk cache was not built.
- **Reused DEV data**, as for the leader. The protected confirmation sets and heldout24
  were not touched.
- **Setup:** Qwen3.5-4B Q4KM, one RTX 5080, batch 1, native Linux, runtime-v11, depth 3.
  NVFP4 was not run. The file needs runtime-v11 and the target's table at start-up.
- The first runtime-v11 build logged the row hash at a level the benchmark server hides.
  Parity passed in that run, but it could not show row identity. The builder raised the
  log level, deleted that build and its runs, and reran everything. Only the rerun is
  recorded.

## Decision

- **Retain** the `rows_only.derived` output of `vocabulary-variants` and the runtime edit
  `vocabulary_shortlist_derived_rows_v1`
  ([benchmarks](../../docs/benchmarks.md#shortlist-only-vocabulary-table)).
- **No promotion.** `ef068eba…` stays the provisional acceptance leader; this is another
  serving form of it. Saved references stay on runtime-v8.
- **Proposed, not decided:** which form is released, the 284 MB file on runtime-v10 with a
  0.85-second start or the 185 MB file on runtime-v11 with a 6.5-second start.

## Evidence and reproduction

All paths are under `artifacts/shortlist-derived-rows-01/`.

**Build:** commits `0460117e` and `df334f80` (merged as `3d0ba920`), frozen source
`88f09679…` in `builds/executed-sources/`. Runtime-v11 `builds/runtime-v11/` (`result.json`
`ceef8fb9…`). New file `builds/variants/drafter-shortlist-derived.gguf` (`a68c297c…`). The
regenerated small file (`156b4f72…`), embedded-shortlist file (`5820b573…`) and sidecar
(`7577676d…`) equal item 11's.

**Checks:** `runs/v11-check` and `runs/v11-screen`, compared per request with item 11's
`checks/runs/v10-check` and `runs/v10-screen`. `builds/logs/summary.txt` holds the row
hashes, the parity counts and the table above; the lead reread it and the file size on
2026-11-20.

**Commands:** `builds/README.md` and `builds/run.sh`; GPU steps ran as pueue task
`item12-check`.

## Artifact disposition

**Retained** (1.5 GB in all), owner Philip, until the release form is decided: the new
file, runtime-v11, the frozen source, both runs and the logs. The regenerated small file and
embedded-shortlist file (827 MB) duplicate item 11's byte for byte and can be retired at
that decision through the retirement command.

**Dependencies:** item 11's runtime-v10, quantizer, small file and v10 runs under
`artifacts/shortlist-only-table-01/`; the leader export `ef068eba…`; the target. None was
changed.

**Retired:** the builder's first runtime-v11 build and its runs, deleted before the recorded
rerun. Their parity result is not preserved.
