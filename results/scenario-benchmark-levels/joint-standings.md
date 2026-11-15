# Joint final-candidate standings

Export `405e326b…`, runtime-v7, depth 3, Qwen3.5-4B, RTX 5080. All seven primary scenarios appear together. This is recomputation of retained evidence, with no new GPU runs.

Ratios above 1 favor final HE4. Brackets are paired 95% source-group intervals; ordinary request scenarios compare separate executions and have no new timing interval. Agentic ratios use assigned tasks per second including tools and failures; request scenarios use emitted tokens per second. HE4 payload 516.0 MiB, Baseline 363.8 MiB. Resident memory is device-wide after load, not peak or incremental allocation; unavailable observations remain unavailable.

## Level 3

| Format / scenario | HE4 vs target | HE4 vs Baseline | HE4 acceptance | HE4 resident MiB |
| --- | --- | --- | ---: | ---: |
| q4km / agentic-coding | 1.3188 [1.2702, 1.3673] | 0.9952 [0.9605, 1.0226] | 66.3% | unavailable |
| q4km / coding-requests | 1.859 historical | 0.988 historical | 86.1% | 6248.0 |
| q4km / chat-writing | 1.215 historical | 0.910 historical | 36.6% | 6248.0 |
| q4km / reasoning | 1.321 historical | 0.936 historical | 43.0% | 6248.0 |
| q4km / long-documents | 1.263 historical | 0.954 historical | 51.8% | 6248.0 |
| q4km / thinking-on | 1.521 [1.473, 1.576] | 0.875 [0.864, 0.887] | 53.8% | 6215.0 |
| q4km / sampled | 1.266 [1.233, 1.310] | 0.954 [0.945, 0.963] | 43.9% | 6222.0 |
| q4km / structured-tools | 1.698 historical | 1.151 historical | 96.2% | 6248.0 |
| nvfp4 / agentic-coding | 1.4359 [1.3784, 1.4796] | 1.0090 [1.0003, 1.0196] | 69.9% | unavailable |
| nvfp4 / coding-requests | 1.816 historical | 0.997 historical | 80.7% | 6048.0 |
| nvfp4 / chat-writing | 1.217 historical | 0.907 historical | 35.7% | 6048.0 |
| nvfp4 / reasoning | 1.341 historical | 0.949 historical | 43.3% | 6048.0 |
| nvfp4 / long-documents | 1.275 historical | 0.966 historical | 49.6% | 6048.0 |
| nvfp4 / thinking-on | 1.516 [1.460, 1.579] | 0.881 [0.869, 0.891] | 52.4% | 6015.0 |
| nvfp4 / sampled | 1.241 [1.211, 1.282] | 0.954 [0.942, 0.965] | 42.6% | 6048.0 |
| nvfp4 / structured-tools | 1.632 historical | 1.163 historical | 92.1% | 6048.0 |

Agentic completion is identical across all three arms: 37/60 for each format.

q4km: **not-established**. agentic-coding: fail; chat-writing: insufficient; coding-requests: insufficient; long-documents: insufficient; reasoning: insufficient; sampled: fail; thinking-on: fail. Agentic coding is the deciding scenario; fresh confirmation remains unexecuted.

nvfp4: **not-established**. agentic-coding: pass; chat-writing: insufficient; coding-requests: insufficient; long-documents: insufficient; reasoning: insufficient; sampled: fail; thinking-on: fail. Agentic coding is the deciding scenario; fresh confirmation remains unexecuted.

## Level 2

| Format / scenario | HE4 vs target | HE4 vs Baseline | HE4 acceptance | HE4 resident MiB |
| --- | --- | --- | ---: | ---: |
| q4km / agentic-coding | 1.3215 [1.2651, 1.3670] | 1.0098 [0.9854, 1.0355] | 64.4% | unavailable |
| q4km / coding-requests | 1.859 historical | 0.988 historical | 86.1% | 6248.0 |
| q4km / chat-writing | 1.215 historical | 0.910 historical | 36.6% | 6248.0 |
| q4km / reasoning | 1.321 historical | 0.936 historical | 43.0% | 6248.0 |
| q4km / long-documents | 1.263 historical | 0.954 historical | 51.8% | 6248.0 |
| q4km / thinking-on | 1.519 [1.425, 1.644] | 0.886 [0.863, 0.913] | 52.8% | 6215.0 |
| q4km / sampled | 1.264 [1.203, 1.387] | 0.956 [0.945, 0.976] | 44.8% | 6222.0 |
| q4km / structured-tools | 1.698 historical | 1.151 historical | 96.2% | 6248.0 |
| nvfp4 / agentic-coding | 1.4497 [1.3269, 1.5662] | 1.0016 [0.9915, 1.0117] | 68.9% | unavailable |
| nvfp4 / coding-requests | 1.816 historical | 0.997 historical | 80.7% | 6048.0 |
| nvfp4 / chat-writing | 1.217 historical | 0.907 historical | 35.7% | 6048.0 |
| nvfp4 / reasoning | 1.341 historical | 0.949 historical | 43.3% | 6048.0 |
| nvfp4 / long-documents | 1.275 historical | 0.966 historical | 49.6% | 6048.0 |
| nvfp4 / thinking-on | 1.483 [1.408, 1.587] | 0.892 [0.885, 0.901] | 52.4% | 6015.0 |
| nvfp4 / sampled | 1.228 [1.170, 1.348] | 0.935 [0.919, 0.966] | 43.2% | 6048.0 |
| nvfp4 / structured-tools | 1.632 historical | 1.163 historical | 92.1% | 6048.0 |

Agentic completion is identical across all three arms: 5/12 Q4KM and 7/12 NVFP4. These small slices have lower completion than the 37/60 full suites and cannot substitute for them.

q4km: **Level 3 required: True**. agentic-coding: direction disagrees with main panel against baseline; chat-writing: acceptance below saved baseline beyond interval; long-documents: acceptance below saved baseline beyond interval; reasoning: acceptance below saved baseline beyond interval; structured-tools: direction disagrees with main panel against baseline; sampled: slower than baseline beyond interval; thinking-on: slower than baseline beyond interval.

nvfp4: **Level 3 required: True**. agentic-coding: direction disagrees with main panel against baseline; chat-writing: acceptance below saved baseline beyond interval; coding-requests: acceptance below saved baseline beyond interval; long-documents: acceptance below saved baseline beyond interval; reasoning: acceptance below saved baseline beyond interval; structured-tools: direction disagrees with main panel against baseline; sampled: slower than baseline beyond interval; thinking-on: slower than baseline beyond interval.

## Routine cohort costs

Summed request/task seconds for target-only and Baseline are historical saved controls; final HE4 seconds are the new candidate-only run. Final panel wall additionally includes loading, warmup and bookkeeping. These cohorts are shared by Level 2 and Level 3; do not add duplicated coverage twice.

| Format / cohort | Saved target s | Saved Baseline s | Final HE4 s | Final panel wall s |
| --- | ---: | ---: | ---: | ---: |
| q4km / screen | 45.9 | 34.1 | 33.5 | 38.5 |
| q4km / requests | 261.3 | 183.8 | 195.5 | 202.7 |
| q4km / prefix | 23.5 | 13.9 | 14.4 | 21.4 |
| q4km / live | 81.3 | 58.8 | 53.5 | 60.4 |
| nvfp4 / screen | 41.5 | 30.8 | 30.9 | 35.3 |
| nvfp4 / requests | 243.9 | 172.2 | 182.3 | 188.2 |
| nvfp4 / prefix | 22.3 | 13.0 | 13.5 | 17.1 |
| nvfp4 / live | 54.8 | 42.1 | 39.7 | 48.8 |

## Reproduction

Frozen reporting source: `/home/philip/BitDraft/artifacts/benchmarks/scenario-levels-01/joint-01/executed-sources/6531097afa92e7257d44d581358fb2f5ed97e2da8c469cd424a60ebaeea39f1e`.
Run `/home/philip/BitDraft/artifacts/benchmarks/scenario-levels-01/joint-01/publish.py DESTINATION` with the destination result directory, from that source with the project Python and `PYTHONPATH` set to the same directory. Fresh report specifications, registered references and source results are authenticated before projection. The complete source is checked even when only a slice is selected. Each slice retains source allocation time separately; no standalone slice allocation time is invented.

The JSON retains every arm, failed/completed strata, supplementary panel evidence and registry proofs. All original raw runs remain unchanged and both result pages remain available. Confirmation was not run.
