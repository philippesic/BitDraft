# Continuing HE4 on the same data reduced request throughput

## Outcome

In this investigation, continuing training made the measured outcomes worse. Acceptance fell from 22.61% to 17.85%, and complete-request
throughput fell from 170.56 to 160.51 tok/s (paired ratio 0.9411 [0.9338, 0.9470]).
The run added 169,097 updates on only 1,762 unique labels, a mean exposure of 1,439.532
per position. Development loss rose from 7.12 to 8.55 and had already reached 7.53 after
5 minutes, while the run kept reporting healthy because monitoring checked liveness only.


| Complete-request throughput | Tokens/second |
| --- | ---: |
| Before continuation | 170.56 |
| After continuation | 160.51 |

[Compact figure data and regeneration recipe](result.json). The paired-ratio interval
does not supply per-bar error bars.

## Scope and limitations

one fixed-data trajectory, DEV3, 5 process repetitions. The smoke cap of 64 differs
from the formal 256. No general conclusion about capacity or convergence.

## Decision

Revise the continuation approach; retain the negative result. This remains a proposed disposition; it does not authorize new training.

## Evidence and reproduction

[experiments/he4-continuation-20261107/report.md](../sources/experiments-he4-continuation-20261107-report-f38d371a.txt) [experiments/he4-continuation-20261107/audit.json](../sources/experiments-he4-continuation-20261107-audit-d5c79541.json) [experiments/he4-continuation-20261107/trajectory.csv](../sources/experiments-he4-continuation-20261107-trajectory-c8057511.csv) [experiments/he4-continuation-20261107/native.csv](../sources/experiments-he4-continuation-20261107-native-0c26d1d0.csv)

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. [Evidence access](../../docs/evidence.md) describes archive recovery and remaining
raw-input limits; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.
