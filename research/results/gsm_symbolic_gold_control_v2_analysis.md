# Supervised gold control — October 13, 2026

Primary gold minus base: **-6.33 pp**, 95% family-bootstrap interval **[-14.52, +2.28]**. Decision: **inconclusive**.
Secondary gold minus hard: -8.33 pp, marginal 95% interval [-16.50, -0.25].

| Model/group | P1 accuracy | P2 accuracy | Pooled accuracy |
|---|---:|---:|---:|
| base | 49.00% | 29.50% | 39.25% |
| gold_seed_mean | 41.33% | 24.50% | 32.92% |
| hard_seed_mean | 52.00% | 30.50% | 41.25% |
| gold_seed_2026 | 35.50% | 22.00% | 28.75% |
| gold_seed_2027 | 42.00% | 26.50% | 34.25% |
| gold_seed_2028 | 46.50% | 25.00% | 35.75% |

| Seed | Gold vs base fixes | Regressions | Net | Gold vs hard fixes | Regressions | Net |
|---|---:|---:|---:|---:|---:|---:|
| 2026 | 34 | 76 | -42 | 37 | 84 | -47 |
| 2027 | 42 | 62 | -20 | 38 | 69 | -31 |
| 2028 | 46 | 60 | -14 | 46 | 68 | -22 |

Secondary variant contrasts (marginal, no multiplicity adjustment):

| Variant | Contrast | pp | 95% interval |
|---|---|---:|---:|
| p1 | gold_minus_base | -7.67 | [-19.84, +5.24] |
| p1 | gold_minus_hard | -10.67 | [-23.17, +1.50] |
| p2 | gold_minus_base | -5.00 | [-13.91, +3.89] |
| p2 | gold_minus_hard | -6.00 | [-14.76, +2.00] |

Numeric-equality verdict changes: base: 0, gold_seed_2026: 0, gold_seed_2027: 0, gold_seed_2028: 0, hard_seed_2026: 0, hard_seed_2027: 0, hard_seed_2028: 0. Legacy scoring remains primary.

This is supervised learning, not self-training. Gold/hard use the same 300 problems, two epochs and 150 updates; gold has 32,206 supervised tokens/epoch versus hard 32,879 (−2.047%). Solution source, quality/style and length differ together.

Only 37 selected development families supply clusters; the questions were inspected in prior research. Intervals condition on these three adapters, without seed resampling. Secondary intervals are unadjusted; treat borderline results cautiously. The reserved 63 families remain untouched. Memory observations remain separate from the analysis. No further experiment was started.
