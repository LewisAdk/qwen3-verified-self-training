# RQ2 evaluation — completed October 13, 2026

The dated analysis plan was saved October 12, before evaluation. The original report and JSON retain that plan date in their filenames; they are preserved unchanged.

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

## Independent validation and measured runtime

All 4,000 records (3,600 newly generated adapter outputs and 400 reused base outputs) have matching question, gold-answer and original-family identities. The reused base batches are byte-identical to the originals. All protected-file hashes match, and an independent direct family-difference calculation reproduces the primary estimate and interval. No further training or experiment was started.

The nine new evaluations used 118.11 minutes of measured generation time, excluding model loading and preflight. This exceeded the initial 60–90 minute estimate. No settings were changed to accelerate the run.

| Model | Generation min | No parsed answer | Tokens at 384 limit | PyTorch peak allocated GiB | PyTorch peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| base | reused | 12 | 20 | not recorded | 4.758 |
| hard_seed_2026 | 13.53 | 11 | 11 | 4.148 | 4.754 |
| hard_seed_2027 | 13.32 | 13 | 14 | 4.148 | 4.779 |
| hard_seed_2028 | 13.66 | 8 | 10 | 4.148 | 4.779 |
| random_seed_2026 | 12.67 | 6 | 5 | 4.092 | 4.779 |
| random_seed_2027 | 13.07 | 8 | 7 | 4.148 | 4.779 |
| random_seed_2028 | 12.52 | 4 | 3 | 4.086 | 4.779 |
| easy_seed_2026 | 12.79 | 5 | 5 | 4.086 | 4.801 |
| easy_seed_2027 | 13.60 | 8 | 7 | 4.135 | 4.801 |
| easy_seed_2028 | 12.94 | 5 | 5 | 4.148 | 4.801 |

These are descriptive diagnostics. An unparsed answer is scored incorrect under the fixed rule; no examples were excluded. A saved token count at the limit can include an answer completed exactly at that limit, and an unparsed response can end before the limit. No longer-budget evaluation was run. Evaluation memory counters do not resolve the earlier training-memory observation.

## Secondary variant comparisons

All five pooled secondary intervals include zero. The P1 random-minus-base interval just excludes zero (+4.67 pp, [0.14, 9.63]); this is a borderline secondary finding, without multiplicity adjustment. It does not satisfy or alter the primary hard-minus-random decision or the conditional experiment gate.

| Variant | Comparison | Difference pp | Marginal 95% interval |
|---|---|---:|---:|
| p1 | hard_minus_random | -1.67 | [-6.84, +3.49] |
| p1 | hard_minus_easy | +0.00 | [-5.97, +5.93] |
| p1 | random_minus_easy | +1.67 | [-5.26, +8.57] |
| p1 | hard_minus_base | +3.00 | [-2.63, +8.67] |
| p1 | random_minus_base | +4.67 | [+0.14, +9.63] |
| p1 | easy_minus_base | +3.00 | [-4.71, +10.79] |
| p2 | hard_minus_random | -0.17 | [-7.02, +5.56] |
| p2 | hard_minus_easy | +5.83 | [+0.00, +12.67] |
| p2 | random_minus_easy | +6.00 | [-1.52, +17.00] |
| p2 | hard_minus_base | +1.00 | [-5.37, +7.89] |
| p2 | random_minus_base | +1.17 | [-5.08, +8.04] |
| p2 | easy_minus_base | -4.83 | [-11.45, +0.73] |
