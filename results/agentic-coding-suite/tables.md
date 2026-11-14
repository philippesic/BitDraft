| Format / panel | Arm | Acceptance, 95% interval | Task rate vs target, 95% interval |
| --- | --- | ---: | ---: |
| Q4KM / slice | Target-only | — | 1.00× (reference) |
| Q4KM / slice | Baseline | 61.78 [55.54, 66.86]% | 1.3087 [1.2303, 1.3781]× |
| Q4KM / slice | S4 final | 64.39 [58.11, 71.39]% | 1.3215 [1.2651, 1.3670]× |
| Q4KM / suite | Target-only | — | 1.00× (reference) |
| Q4KM / suite | Baseline | 63.40 [57.74, 67.89]% | 1.3252 [1.2524, 1.4036]× |
| Q4KM / suite | S4 final | 66.28 [61.61, 69.63]% | 1.3188 [1.2702, 1.3673]× |
| NVFP4 / slice | Target-only | — | 1.00× (reference) |
| NVFP4 / slice | Baseline | 69.34 [61.93, 75.90]% | 1.4474 [1.3187, 1.5705]× |
| NVFP4 / slice | S4 final | 68.93 [62.17, 73.74]% | 1.4497 [1.3269, 1.5662]× |
| NVFP4 / suite | Target-only | — | 1.00× (reference) |
| NVFP4 / suite | Baseline | 68.83 [65.07, 70.90]% | 1.4231 [1.3617, 1.4673]× |
| NVFP4 / suite | S4 final | 69.93 [67.41, 71.63]% | 1.4359 [1.3784, 1.4796]× |

| Format / panel | S4 / Baseline task rate, 95% interval | Acceptance difference, percentage points, 95% interval |
| --- | ---: | ---: |
| Q4KM / slice | 1.0098 [0.9854, 1.0355]× | 2.61 [-0.43, 7.34] |
| Q4KM / suite | 0.9952 [0.9605, 1.0226]× | 2.88 [0.80, 5.56] |
| NVFP4 / slice | 1.0016 [0.9915, 1.0117]× | -0.41 [-2.35, 1.37] |
| NVFP4 / suite | 1.0090 [1.0003, 1.0196]× | 1.10 [-0.32, 2.87] |

| Format / panel / outcome | Tasks | Seconds/task: target / Baseline / S4 | Acceptance: Baseline / S4 | S4 / Baseline task rate |
| --- | ---: | ---: | ---: | ---: |
| Q4KM / slice / completed | 5 | 10.27 / 8.24 / 8.12 | 62.40% / 66.67% | 1.015× |
| Q4KM / slice / failed | 7 | 15.19 / 11.32 / 11.24 | 61.53% / 63.50% | 1.007× |
| Q4KM / suite / completed | 37 | 10.33 / 7.84 / 7.77 | 65.10% / 68.39% | 1.009× |
| Q4KM / suite / failed | 23 | 14.09 / 10.56 / 10.78 | 61.52% / 63.97% | 0.980× |
| NVFP4 / slice / completed | 7 | 14.03 / 9.45 / 9.51 | 74.12% / 72.11% | 0.994× |
| NVFP4 / slice / failed | 5 | 16.66 / 11.86 / 11.74 | 64.42% / 65.56% | 1.010× |
| NVFP4 / suite / completed | 37 | 11.05 / 7.82 / 7.78 | 71.04% / 71.49% | 1.006× |
| NVFP4 / suite / failed | 23 | 15.83 / 11.03 / 10.89 | 66.52% / 68.28% | 1.013× |

| Format / arm | Full measured arm, min | Full raw task time, min | Slice projected raw task time, min | Warmup / other overhead, sec |
| --- | ---: | ---: | ---: | ---: |
| Q4KM / target | 12.30 | 12.09 | 2.64 | 7.18 / 5.23 |
| Q4KM / Baseline | 9.28 | 9.05 | 2.02 | 5.96 / 7.94 |
| Q4KM / S4 | 9.34 | 9.10 | 2.00 | 5.77 / 8.96 |
| NVFP4 / target | 13.17 | 12.95 | 3.03 | 7.56 / 5.41 |
| NVFP4 / Baseline | 9.29 | 9.11 | 2.10 | 5.79 / 5.60 |
| NVFP4 / S4 | 9.22 | 9.02 | 2.09 | 5.69 / 6.60 |
