# CPU execution contract

The public source entry is `python -m w1a1_eagle`, with `preflight INTENT`,
`run INTENT OUTPUT --scope TASK`, and `report OUTPUT`. `--json` before or after the subcommand emits JSON records;
a run emits its admission before allocation and a terminal report afterward.
Human output describes the same records. Reporting reads existing records only.
`python -m tools.doctor` belongs to development tooling, alongside the full check.
The bootstrap and numerical worker are private controller entries declared in
[commands](../policy/commands.toml). The benchmark command remains deferred.

An intent has exact schema 2, validated by [Intent](../w1a1_eagle/intent.py).
It names its objective, hypothesis, decision, risk and externally authorized scope;
source, Q4_K vocabulary and corpus artifact references; explicit fresh/resume/branch
mode and parent; geometry, seed and CPU device; update, attempt and cumulative wall
limits; memory, maximum state bytes and disk reserve; a backing-capacity upper bound;
code identity; explicit update/time checkpoint cadence; and a supplied minimum-distinct-data/maximum-reuse policy. These policy
values are inputs, not recommended scientific thresholds. The complete current field
set is executable in Intent.load and exercised by the numerical integration fixture.
Unknown fields or schema versions refuse. No legacy checkpoint reader is implied.

Artifact references contain `path`, `sha256`, and `bytes`. Relative paths resolve
against their containing document. Corpus schema 1 contains `target_sha256`,
`protected`, `expected_train`, and `records`. Each row names id/source/group/topic,
aliases, role, ancestry, metadata and files. Roles and aliases are reconciled before
opening training metadata. heldout24 is always protected. The ancestry field records
upstream admission provenance; an editable manifest cannot independently establish
that provenance or authorize a run. A real corpus must be produced and checked against
its retained source/role receipts before a scientific run. The modern index adapter now reconciles the selected retained metadata/source chain;
this remains distinct from raw tensor authentication on exposure and native qualification.

The reader supports the selected modern eight-tap F32 capture contract and optional
pinned int32 NPY copies. It authenticates files before exposure, rejects disagreement,
keeps bounded feature materialization, and reconstructs the original single seeded
permutation from the saved cursor, repeating it across passes. The final native label can lie at the last input position;
future mask slots require labels, not observed future inputs. Exposure counts identify
trajectory/native-logit positions. Admission reports planned successful exposure and
the worst allowed exposure including failed attempts. Parent consumption is retained
in checkpoint lineage; branch changes do not establish fresh-data eligibility.

The store owns immutable tensor states and manifests; lifecycle owns the chained event
journal and controller identity. Scratch and read-only diagnostics do not require those
owners. The controller uses nonblocking local ownership to prevent two controllers
writing one output root; this makes no claim about other projects or GPU exclusion.
CPU workers hide CUDA. GPU requests refuse; no project-local lock claims GPU ownership.
State writes recheck observed headroom. Windows backing availability is queried through
the mounted /mnt/c adapter and bounded by the supplied observation. These checks do
not reserve capacity against other writers.

Each attempt binds dirty executable source, its intent and loaded Python/native library
identities. A separate bootstrap arms an OS wall timer before importing Torch, so the
limit remains active without an agent or notification service. The controller also
bounds the subprocess and targets an isolated process group with exact boot/process
birth identity. A failed attempt retains its cost. A lost controller refuses live owned
work; once its owner is gone, recovery conservatively charges its entire remaining
wall budget rather than inventing elapsed time. Automatic retry is unsupported; an
explicit recovery remains subject to cumulative bounds. Before any state or update
attempt exists, an exact-recipe fresh retry can reuse the same root and retain failed
preflight cost. After initialization, recovery requires an explicit parent and resume.

Training records an exposure before its update. Complete states are published through
checksummed atomic manifests. A resume restores tensors, shared aliases, AdamW state,
CPU RNG, sampler cursor and counters. A branch resets optimizer/cursor while retaining
parent lineage and cumulative work. Neither mode silently loads an old trained state
for a fresh run. Unknown or corrupt complete journal evidence refuses. A torn final append can be
reconciled only after live ownership checks: preserve its bytes in an incident receipt,
validate the intact prefix, and discard no complete event. It cannot become selected.
Process-crash recovery is distinct from power-loss durability. Native, hardware,
scientific adequacy and serving performance remain unqualified by CPU fixtures.


Supported readers: intent 2, canonical corpus 1, tensor payload 3, store manifest 1,
event envelope 1, and the emitted native-format GGUF W1Ax version 2. Earlier CPU
tensor schemas 1/2 and intent schema 1 have no implicit reader or migration. Preserve its
originals; migration needs an explicit new artifact and verified source lineage.
Tensor payload 3 binds source/vocabulary identities and validates exact Adam moment
coverage, types and shapes before restoration. Artifact relocation does not change
content identity; exact recipe identity additionally binds executable source.

A CPU run now also exports its final complete state. The pinned GGUF adapter streams
headers and bounded tensor chunks through the store rather than letting NumPy bypass
the byte budget. It preserves original Q4_K vocabulary codes, checks frozen execution
operands against their export operands, and verifies every tensor through the GGUF
reader before publication. Sign bits use little-endian 32-bit words, nonnegative zero
is one, and unused tail bits are zero. This is format conformance, not native loading.

Checkpoint cadence is explicit. The prospective byte bound includes retained intermediate
states, initial/final state, export and a bounded torn-journal incident. The store never
automatically deletes historical states to fit. Every entered update is journaled before
sampling; actual exposure is a separate event. Recovery charges uncommitted attempts,
including failed reads and updates. Whole-attempt wall cost includes preflight, loading,
serialization and failure in the controller; the tensor's operational update timer is
separate. A new task/root does not grant a new allocation merely by changing fields.
Parent lineage remains explicit; full parent allocation reports must be retained rather
than inferred from tensor update time.

Loaded identity records actual mapped executables and imported package code files.
Bytecode caches are bypassed in CPU subprocesses. Terminal identity adds late imports
and rejects drift in previously loaded files. File hashes may be reused within that same
attempt only when device/inode/size/mtime/ctime remain equal; the report labels reuse.
This caches file identity work, never execution proof or hardware measurements.

An optional `input_adapter` selects `he4fresh-index-v1` with exact manifest/target
identities, expected sequence/window/position counts, exclusions, logical root and
physical metadata root. Optional data_root maps tensor files separately. Unsupported
forms refuse; canonical corpus 1 remains supported. Intent payloads are internally
immutable: reading raw returns a copy rather than a mutable admission record.
Admission names prior-use counts explicitly. Exposure estimates describe additional
visits, never unknown lifetime consumption or an assertion of scientific adequacy.

Each admitted run retains a bounded source closure under sources/<identity> and an
exact schema-1 intent receipt under intents/<identity>. Workers and their preflight
execute from that closure, with bytecode reuse disabled, rather than a mutable
checkout. Relative input references retain their original containing directory.
Changing the caller's intent after admission cannot change its child. Snapshot files
are read-only and checked by content; this is cooperative worktree isolation, not
protection against an owner who deliberately rewrites retained files or dependencies.
Actual loaded dependency identities are still checked separately.

Tensor payload 3 retains learned parameters, Adam/RNG/counters and the streamed
fingerprint of every frozen buffer. The authenticated initializer and vocabulary
remain required dependencies; they reconstruct frozen data before restore. A mismatch
refuses. Runtime buffer object/version guards detect ordinary in-place/replacement
mutations before publication. They are not a hostile Python memory sandbox.
The exact next-update CPU witness covers this format; old payloads are preserved
without implicit acceptance.

The supported event journal is bounded to 64 MiB. CPU execution stops at 56 MiB to
retain room for complete state/export/terminal evidence; it never deletes events to
continue. This is a declared storage stop, not goal completion. The prospective disk
upper bound includes the journal, source/intent snapshots and torn-tail receipt in
addition to states/exports. Longer execution needs an explicitly qualified journal
migration or larger supported format. No OS-reserved disk capacity is claimed.
Human terminal output names controller outcome, worker stop, confirmed activation,
starting state and risk; missing evidence remains unknown.
