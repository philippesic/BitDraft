# Scenario measurements

Per-unit measurements: [new requests](new-requests-measurements.csv), [final routine](final-routine-measurements.csv), [earlier protocol](previous-protocol-measurements.csv). The JSON index binds each file and the complete retained extraction.

Measured request seconds below exclude startup. Full panel wall includes load/warmup/teardown; Level 2 slices are projections, without independent startup measurements.

| Precision / cohort | Target request s | Baseline request s | HE4 request s | Full panel wall s |
| --- | ---: | ---: | ---: | ---: |
| q4km-thinking-slice | 100.9 | 58.9 | 66.4 | projected slice |
| q4km-thinking-suite | 342.2 | 196.8 | 225.0 | 797.6 |
| q4km-sampled-slice | 103.4 | 78.2 | 81.8 | projected slice |
| q4km-sampled-suite | 635.0 | 478.5 | 501.5 | 1658.5 |
| nvfp4-thinking-slice | 75.5 | 45.5 | 50.9 | projected slice |
| nvfp4-thinking-suite | 272.6 | 158.4 | 179.8 | 630.2 |
| nvfp4-sampled-slice | 105.9 | 80.7 | 86.2 | projected slice |
| nvfp4-sampled-suite | 619.7 | 476.4 | 499.5 | 1636.2 |

## Three-arm comparisons

Throughput is emitted tokens per request second. Resident MiB is whole-device memory after load, not peak or incremental drafter allocation. All intervals are paired 95% source-group intervals.

| Precision / cohort | Arm | Drafter MiB | Resident MiB | Tokens/s | Acceptance | vs target (95%) | vs Baseline (95%) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| q4km-thinking-slice | baseline | 363.8 | 6033.0 | 314.3 | 66.2% | 1.714 (1.630, 1.817) | 1.000 |
| q4km-thinking-slice | candidate | 516.0 | 6215.0 | 278.5 | 52.8% | 1.519 (1.425, 1.644) | 0.886 (0.863, 0.913) |
| q4km-thinking-slice | target | 0.0 | 4335.0 | 183.4 | — | 1.000 | 0.584 (0.550, 0.613) |
| q4km-thinking-suite | baseline | 363.8 | 6033.0 | 318.8 | 66.6% | 1.738 (1.687, 1.793) | 1.000 |
| q4km-thinking-suite | candidate | 516.0 | 6215.0 | 278.9 | 53.8% | 1.521 (1.473, 1.576) | 0.875 (0.864, 0.887) |
| q4km-thinking-suite | target | 0.0 | 4335.0 | 183.4 | — | 1.000 | 0.575 (0.558, 0.593) |
| q4km-sampled-slice | baseline | 363.8 | 6040.0 | 203.7 | 51.5% | 1.322 (1.270, 1.424) | 1.000 |
| q4km-sampled-slice | candidate | 516.0 | 6222.0 | 194.8 | 44.8% | 1.264 (1.203, 1.387) | 0.956 (0.945, 0.976) |
| q4km-sampled-slice | target | 0.0 | 4342.0 | 154.1 | — | 1.000 | 0.756 (0.702, 0.787) |
| q4km-sampled-suite | baseline | 363.8 | 6040.0 | 203.4 | 50.0% | 1.327 (1.296, 1.370) | 1.000 |
| q4km-sampled-suite | candidate | 516.0 | 6222.0 | 194.1 | 43.9% | 1.266 (1.233, 1.310) | 0.954 (0.945, 0.963) |
| q4km-sampled-suite | target | 0.0 | 4342.0 | 153.3 | — | 1.000 | 0.754 (0.730, 0.772) |
| nvfp4-thinking-slice | baseline | 363.8 | 5833.0 | 325.5 | 64.9% | 1.661 (1.577, 1.773) | 1.000 |
| nvfp4-thinking-slice | candidate | 516.0 | 6015.0 | 290.5 | 52.4% | 1.483 (1.408, 1.587) | 0.892 (0.885, 0.901) |
| nvfp4-thinking-slice | target | 0.0 | 4135.0 | 196.0 | — | 1.000 | 0.602 (0.564, 0.634) |
| nvfp4-thinking-suite | baseline | 363.8 | 5833.0 | 327.2 | 65.2% | 1.721 (1.667, 1.782) | 1.000 |
| nvfp4-thinking-suite | candidate | 516.0 | 6015.0 | 288.2 | 52.4% | 1.516 (1.460, 1.579) | 0.881 (0.869, 0.891) |
| nvfp4-thinking-suite | target | 0.0 | 4135.0 | 190.1 | — | 1.000 | 0.581 (0.561, 0.600) |
| nvfp4-sampled-slice | baseline | 363.8 | 5866.0 | 208.8 | 50.3% | 1.313 (1.266, 1.400) | 1.000 |
| nvfp4-sampled-slice | candidate | 516.0 | 6048.0 | 195.4 | 43.2% | 1.228 (1.170, 1.348) | 0.935 (0.919, 0.966) |
| nvfp4-sampled-slice | target | 0.0 | 4168.0 | 159.0 | — | 1.000 | 0.762 (0.714, 0.790) |
| nvfp4-sampled-suite | baseline | 363.8 | 5866.0 | 206.6 | 48.7% | 1.301 (1.274, 1.341) | 1.000 |
| nvfp4-sampled-suite | candidate | 516.0 | 6048.0 | 197.0 | 42.6% | 1.241 (1.211, 1.282) | 0.954 (0.942, 0.965) |
| nvfp4-sampled-suite | target | 0.0 | 4168.0 | 158.8 | — | 1.000 | 0.769 (0.746, 0.785) |

## Full arm allocation seconds

These include load, warmup and teardown. The first Q4KM thinking run did not record per-arm allocation time.

| Panel | Target s | Baseline s | HE4 s |
| --- | ---: | ---: | ---: |
| nvfp4-sampled-suite | 633.0 | 490.1 | 512.8 |
| nvfp4-thinking-suite | 280.0 | 164.7 | 185.1 |
| q4km-sampled-suite | 650.8 | 492.3 | 515.0 |
| q4km-thinking-suite | unavailable | unavailable | unavailable |

## Earlier protocol-v3 standings

Recomputed from the original three-arm S4 records, which remain unchanged. Their payload/memory metadata and all per-unit values are retained in result.json; historical resident memory was not recorded.

| Target / scenario | S4 vs target (95%) | S4 vs Baseline (95%) | S4 acceptance |
| --- | --- | --- | ---: |
| q4km / chat-writing | 1.147 (1.103, 1.212) | 0.859 (0.848, 0.872) | 32.3% |
| q4km / coding-requests | 1.735 (1.647, 1.816) | 0.922 (0.892, 0.956) | 77.7% |
| q4km / long-documents | 1.221 (1.181, 1.272) | 0.922 (0.905, 0.940) | 48.0% |
| q4km / reasoning | 1.231 (1.175, 1.313) | 0.872 (0.849, 0.894) | 37.4% |
| q4km / structured-tools | 1.629 (1.548, 1.691) | 1.104 (1.030, 1.233) | 86.8% |
| nvfp4 / chat-writing | 1.157 (1.107, 1.228) | 0.862 (0.850, 0.875) | 32.1% |
| nvfp4 / coding-requests | 1.677 (1.586, 1.763) | 0.921 (0.898, 0.946) | 72.4% |
| nvfp4 / long-documents | 1.201 (1.165, 1.249) | 0.910 (0.889, 0.929) | 43.7% |
| nvfp4 / reasoning | 1.251 (1.212, 1.314) | 0.885 (0.873, 0.902) | 38.2% |
| nvfp4 / structured-tools | 1.566 (1.403, 1.681) | 1.116 (1.057, 1.168) | 88.3% |

## Final export on existing routine cohorts

Only the final candidate was rerun. Saved control speeds are descriptive historical ratios and have no new paired timing interval. Acceptance comparisons retain their source-group intervals in result.json.

| Target / cohort / scenario | Candidate s | HE4 vs saved target | HE4 vs saved Baseline | HE4 acceptance |
| --- | ---: | --- | --- | ---: |
| nvfp4-harness / coding-requests | 11.7 | 1.688 historical | 0.942 historical | 68.6% |
| nvfp4-harness / structured-tools | 1.7 | 1.425 historical | 1.127 historical | 90.5% |
| nvfp4-live / unclassified | 39.7 | 1.380 historical | 1.061 historical | 61.0% |
| nvfp4-requests / chat-writing | 72.3 | 1.217 historical | 0.907 historical | 35.7% |
| nvfp4-requests / coding-requests | 19.9 | 1.816 historical | 0.997 historical | 80.7% |
| nvfp4-requests / long-documents | 31.9 | 1.275 historical | 0.966 historical | 49.6% |
| nvfp4-requests / reasoning | 54.8 | 1.341 historical | 0.949 historical | 43.3% |
| nvfp4-requests / structured-tools | 3.4 | 1.632 historical | 1.163 historical | 92.1% |
| nvfp4-screen / chat-writing | 8.2 | 1.201 historical | 0.962 historical | 38.9% |
| nvfp4-screen / coding-requests | 6.6 | 1.608 historical | 1.020 historical | 75.2% |
| nvfp4-screen / long-documents | 9.7 | 1.245 historical | 0.995 historical | 54.7% |
| nvfp4-screen / reasoning | 4.9 | 1.369 historical | 0.972 historical | 46.6% |
| nvfp4-screen / structured-tools | 1.5 | 1.479 historical | 1.164 historical | 91.2% |
| q4km-harness / coding-requests | 12.4 | 1.671 historical | 0.943 historical | 66.8% |
| q4km-harness / structured-tools | 2.0 | 1.405 historical | 1.135 historical | 91.1% |
| q4km-live / unclassified | 53.5 | 1.519 historical | 1.099 historical | 74.2% |
| q4km-requests / chat-writing | 73.9 | 1.215 historical | 0.910 historical | 36.6% |
| q4km-requests / coding-requests | 20.8 | 1.859 historical | 0.988 historical | 86.1% |
| q4km-requests / long-documents | 35.5 | 1.263 historical | 0.954 historical | 51.8% |
| q4km-requests / reasoning | 60.6 | 1.321 historical | 0.936 historical | 43.0% |
| q4km-requests / structured-tools | 4.7 | 1.698 historical | 1.151 historical | 96.2% |
| q4km-screen / chat-writing | 8.7 | 1.209 historical | 1.027 historical | 39.2% |
| q4km-screen / coding-requests | 6.6 | 1.665 historical | 1.036 historical | 81.7% |
| q4km-screen / long-documents | 11.1 | 1.274 historical | 0.989 historical | 53.4% |
| q4km-screen / reasoning | 5.0 | 1.393 historical | 0.968 historical | 47.3% |
| q4km-screen / structured-tools | 2.0 | 1.553 historical | 1.180 historical | 95.1% |

## Diagnostics and interruption

Original 120 windows: all metrics unchanged. Supplementary agentic/thinking groups cost 2.9/7.7 seconds; combined 160-window pass 20.6 seconds. Resident supplementary cost was 4.5–4.6 seconds per pass. The complete resident pilot interruption took 73.4 s. The two probe updates used the original final state with a new optimizer and extended cosine end; this qualifies interruption mechanics, not improved model quality.
