# Owned modern CUDA cache duplicate audit

Read-only inventory of the requested supervisor CUDA cache directories. No cache paths/content were changed, deleted or hardlinked. The inventory excludes native-replay02 unconditionally and excludes any potentially live recorded process or process group. Process evidence is in the JSON; recorded PIDs were checked against current boot/birth data when available and /proc absence/group absence.

| Cache supervisor | Files | Logical MiB | Eligibility |
|---|---:|---:|---|
| local-w1a8-modern-cadence-20261103-01 | 26 | 150.77 | finished, no matching live owner/group |
| local-w1a8-modern-cadence-20261103-02 | 27 | 151.19 | finished, no matching live owner/group |
| local-w1a8-modern-capture-20261103-01 | 0 | 0.00 | finished, no matching live owner/group |
| local-w1a8-modern-capture-20261103-02 | 26 | 150.77 | finished, no matching live owner/group |
| local-w1a8-modern-generalization-capture-20261103-01 | 0 | 0.00 | finished, no matching live owner/group |
| local-w1a8-modern-gpu-training-qualification-20261103-01 | 36 | 422.27 | finished, no matching live owner/group |
| local-w1a8-modern-native-admission-20261103-01 | 48 | 206.70 | finished, no matching live owner/group |
| local-w1a8-modern-native-replay-20261103-01 | 9 | 9.92 | finished, no matching live owner/group |
| local-w1a8-modern-ops-20261103-01 | 2 | 0.14 | finished, no matching live owner/group |
| local-w1a8-modern-ops-20261103-02 | 2 | 0.14 | finished, no matching live owner/group |
| local-w1a8-modern-output-cap-20261103-01 | 27 | 151.19 | finished, no matching live owner/group |
| local-w1a8-modern-server-diagnostic-20261103-01 | 27 | 151.19 | finished, no matching live owner/group |
| local-w1a8-modern-serving-trace-20261103-01 | 31 | 151.62 | finished, no matching live owner/group |
| local-w1a8-modern-native-replay-20261103-02 | — | — | excluded_unconditionally_by_assignment |

Eligible caches contain261files, 261existing unique inodes, 1545.89MiB logical /1546.75MiB allocated. Hashed219unique candidate inodes (1122.39MiB read) in0.94s. Files were unchanged across hashing. Singleton sizes cannot match any other inventoried file and were not hashed.

Found42SHA256-identical content groups on separate inodes. Preserving all paths while consolidating only identical non-index blobs has a theoretical allocated-block saving of982,876,160bytes (937.34MiB /0.915GiB). Including identical mutable index files would yield937.35MiB; indexes should remain independent. This is potential duplicate storage, not measured reclamation and not authorization to mutate caches.

WSL free space before/after this audit: 3,022,929,920/3,022,929,920bytes (2.815/2.815GiB). Any delta reflects concurrent activity, not cleanup. No Windows volume reclaimed-space claim is made.

The JSON enumerates every exact duplicate path, same-relative-path relationship, size, SHA256, device/inode/link count, allocated blocks and source cache. Retain all unique cache blobs, including training qualification additions; their recompilation cost is unmeasured. Cached binary blobs are useful development output, not disposable evidence. Future review must confirm the GPU lane is idle, authenticate current owners/process groups again, recheck exact file identities/content and establish safe driver update/hardlink behavior before any action. Existing useful paths must survive. No paid models, checkpoints or original run artifacts were inspected.

All42duplicate groups have the same relative cache path as well as matching size/SHA256. The training qualification cache has 421954704bytes (402.41MiB) of content not duplicated in another eligible cache; preserve those additions. All existing261files are separate inodes, so these repeats are not already hardlinked.

Owner update: root briefly designated output-cap/cuda as shared by a new collection, then reported that collection failed its storage guard before opening the cache or starting any server/capture. The failed supervisor is terminal and output-cap cache quiescent again according to the owner; the empty failed-collection cache is included in this inventory. Any future action still needs immediate live-owner/identity revalidation.
