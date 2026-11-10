# CPU execution contract

The public source entry is `python -m w1a1_eagle`, with `preflight INTENT`,
`run INTENT OUTPUT --scope TASK`, and `report OUTPUT`. `--json` emits JSON records;
a run emits its admission before allocation and a terminal report afterward.
Human output describes the same records. Reporting reads existing records only.
`python -m tools.doctor` belongs to development tooling, alongside the full check.
The bootstrap and numerical worker are private controller entries declared in
[commands](../policy/commands.toml). The benchmark command remains deferred.

An intent has exact schema 1, validated by [Intent](../w1a1_eagle/intent.py).
It names its objective, hypothesis, decision, risk and externally authorized scope;
source, Q4_K vocabulary and corpus artifact references; explicit fresh/resume/branch
mode and parent; geometry, seed and CPU device; update, attempt and cumulative wall
limits; memory, maximum state bytes and disk reserve; a backing-capacity upper bound;
code identity; and a supplied minimum-distinct-data/maximum-reuse policy. These policy
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
its retained source/role receipts before a scientific run. The present fixture is a
synthetic schema/conformance proof, not that real-corpus qualification.

The reader supports the selected modern eight-tap F32 capture contract and optional
pinned int32 NPY copies. It authenticates files before exposure, rejects disagreement,
keeps bounded feature materialization, and reconstructs a seeded epoch permutation
from the saved cursor. The final native label can lie at the last input position;
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
explicit exact-recipe resume remains subject to cumulative bounds.

Training records an exposure before its update. Complete states are published through
checksummed atomic manifests. A resume restores tensors, shared aliases, AdamW state,
CPU RNG, sampler cursor and counters. A branch resets optimizer/cursor while retaining
parent lineage and cumulative work. Neither mode silently loads an old trained state
for a fresh run. Unknown/partial journal evidence refuses instead of becoming selected.
Process-crash recovery is distinct from power-loss durability. Native, hardware,
scientific adequacy and serving performance remain unqualified by CPU fixtures.
