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
