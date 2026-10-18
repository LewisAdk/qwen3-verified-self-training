# RQ2 evaluation — October 12, 2026

Primary: hard minus random = -0.92 percentage points; paired 95% family-bootstrap interval [-5.00, +2.87]. Decision: **inconclusive**.

| Group | P1 mean accuracy | P2 mean accuracy | Pooled mean accuracy |
|---|---:|---:|---:|
| base | 49.00% | 29.50% | 39.25% |
| hard | 52.00% | 30.50% | 41.25% |
| random | 53.67% | 30.67% | 42.17% |
| easy | 52.00% | 24.67% | 38.33% |

Secondary pooled comparisons; marginal intervals, no multiplicity adjustment:

| Comparison | Difference pp | 95% interval |
|---|---:|---:|
| hard_minus_easy | +2.92 | [-1.54, +7.50] |
| random_minus_easy | +3.83 | [-1.58, +10.09] |
| hard_minus_base | +2.00 | [-2.31, +6.34] |
| random_minus_base | +2.92 | [-1.11, +7.07] |
| easy_minus_base | -0.92 | [-5.95, +3.92] |

Per-seed results and fixes/regressions against base (400 questions each):

| Adapter | P1 correct / 200 | P2 correct / 200 | Pooled correct / 400 | Accuracy | Delta vs base pp | Fixes | Regressions | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_seed_2026 | 102 | 60 | 162 | 40.50% | +1.25 | 42 | 37 | 5 |
| hard_seed_2027 | 104 | 64 | 168 | 42.00% | +2.75 | 44 | 33 | 11 |
| hard_seed_2028 | 106 | 59 | 165 | 41.25% | +2.00 | 41 | 33 | 8 |
| random_seed_2026 | 108 | 63 | 171 | 42.75% | +3.50 | 42 | 28 | 14 |
| random_seed_2027 | 108 | 59 | 167 | 41.75% | +2.50 | 38 | 28 | 10 |
| random_seed_2028 | 106 | 62 | 168 | 42.00% | +2.75 | 41 | 30 | 11 |
| easy_seed_2026 | 109 | 50 | 159 | 39.75% | +0.50 | 39 | 37 | 2 |
| easy_seed_2027 | 96 | 48 | 144 | 36.00% | -3.25 | 32 | 45 | -13 |
| easy_seed_2028 | 107 | 50 | 157 | 39.25% | +0.00 | 39 | 39 | 0 |

Within-seed hard versus random (fix = random wrong, hard correct):

| Seed | Hard − random pp | Fixes | Regressions | Net |
|---|---:|---:|---:|---:|
| 2026 | -2.25 | 32 | 41 | -9 |
| 2027 | +0.25 | 36 | 35 | 1 |
| 2028 | -0.75 | 33 | 36 | -3 |

Numeric-equality diagnostic verdict changes: base: 0, hard_seed_2026: 0, hard_seed_2027: 0, hard_seed_2028: 0, random_seed_2026: 0, random_seed_2027: 0, random_seed_2028: 0, easy_seed_2026: 0, easy_seed_2027: 0, easy_seed_2028: 0. Legacy scoring remains primary.

Only 37 original families supply independent clusters; percentile intervals may be optimistic, and borderline exclusion of zero should be treated cautiously. Group-mean intervals condition on these three adapters, without seed resampling. Per-variant metrics, per-seed intervals, group-average flips and descriptive seed SD are in the JSON report.

Design: supervised-token matching within ±1%, with hard/random/easy counts 300/353/403 and updates 150/178/202. Easy received 34.67% more updates than hard. Preserve the recorded reasoning-quality limitation and unresolved memory observation in the dated plan.

The conditional 300-example experiment gate is not met. No additional training was started. The 63 reserved families remain untouched.
