# Benchmark requirements

No benchmark implementation or usable benchmark entry exists in foundation. The future
sole benchmark entry is reserved by policy; source/native/hardware approval comes first.
HE4 is W1A4/A8-down/Q4K-head. W1A8 means historical DSpark W1A8; DSpark Q4KM is the old
baseline. Baseline means current external DFlash Q4KM. Do not mix eras or relabel families.

Complete-request throughput is sum emitted tokens divided by sum full HTTP request wall.
Acceptance is sum accepted divided by sum proposed. Neither is a mean of per-request
percentages/rates. Host-traced draft wall, GPU event time, server decode wall and full
HTTP wall measure different scopes and cannot be pooled. Diagnostic instrumentation
is distinct from clean timing. Repetitions estimate timing variation, not unseen tasks.

Comparison requires matching machine/software, era, target/tokenizer/template, workload/
prompt IDs/roles/consumption, output cap/input premise, complete process blocks/attempts,
binary/runtime pins, instrumentation, depth, KV/cache and generation controls. Declared
candidate differences retain exact identities. Never waive clock/workload/correctness,
select favorable retries, invent missing fields or add a force bypass. Raw token outputs,
activation evidence and recomputable aggregates are mandatory. Cross-machine results
stay stratified; historical DSpark versus modern DFlash has no speed ratio.
