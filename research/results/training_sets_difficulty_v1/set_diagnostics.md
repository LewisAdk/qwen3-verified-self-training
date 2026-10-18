# Prepared training-set diagnostics

No training has run. Unmatched sets are diagnostic only. Even matched sets await reasoning review and the remaining training decisions.

| Group | Examples | Supervised tokens | Solution tokens | Mean solution length | Max solution length | Gap vs hard | Match | Draw |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| hard | 300 | 32879 | 32579 | 108.60 | 342 | +0.00% | True | 1 |
| random | 300 | 29482 | 29182 | 97.27 | 342 | -10.33% | False | 330 |
| easy | 300 | 25430 | 25130 | 83.77 | 224 | -22.66% | False | 778 |

Bucket breakdowns (correct out of four):

- hard: {1: 58, 2: 95, 3: 147}; max full sequence 849.
- random: {1: 31, 2: 41, 3: 69, 4: 159}; max full sequence 849.
- easy: {4: 300}; max full sequence 743.

Feasible totals for random 300-problem subsets: {'minimum_possible': 15772, 'maximum_possible': 43351}.
Feasible totals for easy 300-problem subsets: {'minimum_possible': 17116, 'maximum_possible': 30995}.

Matching uses target-token counts including EOS; all groups contain 300 unique problems. With the original microbatch 1, accumulation 4 and two epochs, each model would take 150 optimizer updates. No shortest-solution selection, token padding, truncation or repeated examples were used.

Random is conditioned on token budget if matched. Group membership can overlap; review oversamples 1/4 and is not an unbiased pool error-rate estimate. The original 37 development / 63 reserved family split is preserved.
