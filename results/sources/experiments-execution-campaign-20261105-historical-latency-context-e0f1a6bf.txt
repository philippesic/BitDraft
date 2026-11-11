# Historical head precision and draft-latency interpretation

November 5, 2026. Root inspected the actual two retained GGUF files named by
runs/dspark-w1a8-dflash-20261104-01/config.json, using GGUFReader metadata only.
No GPU experiment or tensor mutation occurred.

| Model | Output head | Shape (vocabulary x hidden) | Matrix bytes | MiB |
| --- | --- | --- | ---: | ---: |
| Historical DSpark W1A8 parent3504 | Q4_K | 151936 x 2560 | 218787840 | 208.65 |
| Historical DSpark Q4_K_M | Q6_K | 151936 x 2560 | 319065600 | 304.28 |
| Current compact DFlash G4/D8 | Target-shared Q6_K | 248320 x 2560 | 521472000 | 497.31 |
| Current DFlash Q4_K_M | Same target-shared Q6_K | 248320 x 2560 | 521472000 | 497.31 |

These are stored head matrix bytes, not total deployment bytes, measured traffic,
or isolated head timings. Shared current head bytes are charged once in combined
model accounting. Q4_K_M is a mixed GGUF weight recipe, not a promise that output
heads use four bits.

Historical complete draft host spans: W1A8 2.712ms versus DSpark Q4 3.031ms,
about10.5% lower. This is an artifact comparison including both FFN and head
precision changes, NOT an isolated causal binary-FFN speedup. Head matrix bytes
are31.4% lower, which supports a bandwidth-saving hypothesis but does not quantify
its latency contribution. A matched head-only ablation is required to apportion
those savings. Markov matrices, embedding, fusion and attention formats match
between those two historical files; FFN coverage differs as previously recorded.

The historical DSpark binary runtime also uses a seven-position Tensor Core path;
current DFlash depth2 ordinarily uses three positions, and current qualified
DP4A beat the tested MMA adapter. Architecture/batch/dispatch and additional
compensation arithmetic also differ; do not promise the historical kernel win
on current shapes. Different source vocabularies further change the head's share
of total draft work. Current profiled shared-head cost is about0.57ms per round,
the largest measured kernel group in the short diagnostic trace.

A draft-only Q4 head may lower its compute/read cost while adding private runtime
storage because the target still needs its unchanged shared Q6 matrix. Historical
DSpark already owns a separate output tensor. Our modern head screen tested
Q4_0, not the historical Q4_K, and did not test it on the final optimized compact
G4 candidate. A controlled current Q6_K versus private Q4_K test would answer a
different question than that earlier Q4_0 screen. This is a proposed next isolation,
not a new implementation or training authorization, nor a guaranteed speedup.

Sources: [historical latency report](../dspark-w1a8-dflash-20261104/README.md),
[original pins](../../runs/dspark-w1a8-dflash-20261104-01/config.json),
[current component metadata](../precision-campaign-20261104/component-inventory.json),
[current execution report](report.md). Raw data and frozen historical results
remain unchanged.
