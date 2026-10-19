## bfcl  (items run by every model: 1040)

| model | acc | 95% CI | runaway | tokens/item | reasoning tok | median latency s |
|---|---|---|---|---|---|---|
| swift15 | 79.0% (822/1040) | 76.5-81.4 | 0.1% (1) | 431 | 239 | 5.712 |
| intel | 82.0% (853/1040) | 79.6-84.2 | 0.1% (1) | 421 | 233 | 5.744 |

| pair (A - B) | diff | 95% CI | A-only / B-only | p |
|---|---|---|---|---|
| swift15 - intel | -3.0 pt | -5.1 .. -0.9 | 46 / 77 | 0.0066 |

| subset | swift15 | intel |
|---|---|---|
| irrelevance | 84.0% (n=100, run 0%) | 83.0% (n=100, run 0%) |
| live_irrelevance | 80.0% (n=150, run 1%) | 80.7% (n=150, run 1%) |
| live_multiple | 72.0% (n=150, run 0%) | 78.0% (n=150, run 0%) |
| live_parallel | 75.0% (n=16, run 0%) | 75.0% (n=16, run 0%) |
| live_parallel_multiple | 58.3% (n=24, run 0%) | 70.8% (n=24, run 0%) |
| live_simple | 77.0% (n=100, run 0%) | 88.0% (n=100, run 0%) |
| multi_turn_base | 64.0% (n=100, run 0%) | 62.0% (n=100, run 0%) |
| multiple | 89.0% (n=100, run 0%) | 91.0% (n=100, run 0%) |
| parallel | 86.0% (n=100, run 0%) | 87.0% (n=100, run 0%) |
| parallel_multiple | 78.0% (n=100, run 0%) | 80.0% (n=100, run 0%) |
| simple_python | 90.0% (n=100, run 0%) | 95.0% (n=100, run 0%) |

## evalplus  (items run by every model: 542)

| model | acc | 95% CI | runaway | tokens/item | reasoning tok | median latency s |
|---|---|---|---|---|---|---|
| swift15 | 86.0% (466/542) | 82.8-88.6 | 0.0% (0) | 879 | 522 | 9.793 |
| intel | 86.2% (467/542) | 83.0-88.8 | 0.0% (0) | 883 | 533 | 10.934 |

| pair (A - B) | diff | 95% CI | A-only / B-only | p |
|---|---|---|---|---|
| swift15 - intel | -0.2 pt | -2.2 .. +1.8 | 15 / 16 | 1.0 |

| subset | swift15 | intel |
|---|---|---|
| humaneval+ | 95.7% (n=164, run 0%) | 94.5% (n=164, run 0%) |
| mbpp+ | 81.8% (n=378, run 0%) | 82.5% (n=378, run 0%) |

## lcb  (items run by every model: 175)

| model | acc | 95% CI | runaway | tokens/item | reasoning tok | median latency s |
|---|---|---|---|---|---|---|
| swift15 | 58.3% (102/175) | 50.9-65.3 | 23.4% (41) | 10013 | 9596 | 79.954 |
| intel | 58.3% (102/175) | 50.9-65.3 | 31.4% (55) | 10832 | 10508 | 82.112 |

| pair (A - B) | diff | 95% CI | A-only / B-only | p |
|---|---|---|---|---|
| swift15 - intel | +0.0 pt | -6.9 .. +6.9 | 18 / 18 | 1.0 |

| subset | swift15 | intel |
|---|---|---|
| lcb_v6_easy | 97.7% (n=43, run 0%) | 100.0% (n=43, run 0%) |
| lcb_v6_hard | 31.2% (n=80, run 48%) | 26.2% (n=80, run 61%) |
| lcb_v6_medium | 67.3% (n=52, run 6%) | 73.1% (n=52, run 12%) |

## longctx  (items run by every model: 180)

| model | acc | 95% CI | runaway | tokens/item | reasoning tok | median latency s |
|---|---|---|---|---|---|---|
| swift15 | 96.1% (173/180) | 92.2-98.1 | 0.0% (0) | 542 | 382 | 6.696 |
| intel | 100.0% (180/180) | 97.9-100.0 | 0.0% (0) | 521 | 359 | 6.277 |

| pair (A - B) | diff | 95% CI | A-only / B-only | p |
|---|---|---|---|---|
| swift15 - intel | -3.9 pt | -6.7 .. -1.7 | 0 / 7 | 0.0156 |

| subset | swift15 | intel |
|---|---|---|
| longctx_110k | 93.3% (n=60, run 0%) | 100.0% (n=60, run 0%) |
| longctx_32k | 100.0% (n=60, run 0%) | 100.0% (n=60, run 0%) |
| longctx_64k | 95.0% (n=60, run 0%) | 100.0% (n=60, run 0%) |

## Macro average over benchmarks (equal weight)

| pair | diff | approx 95% CI |
|---|---|---|
| swift15 - intel | -1.8 pt | -3.7 .. +0.2 |
