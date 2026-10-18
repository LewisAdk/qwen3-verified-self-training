# Scaled random self-training — final results, 2026-10-13

Decision: **not_success**. Equal three-seed mean scaled-random − saved base: **+1.5833 percentage points**.
Family-only 95% interval: **[-2.136905, +5.500000]**. Seed-and-family 95% interval: **[-3.076923, +6.500000]**.
Success requires both lower endpoints strictly above zero, as frozen. All three seeds and all 400 questions are retained; no best-seed selection. No new tests or outcome-guided method changes were introduced.

## Training and preservation

The frozen 2,000-example set, candidate choices and ordering were reused unchanged. Each seed trained freshly from the pinned base with the unchanged RQ2 numerical engine and settings. The same batch-8, FP16, greedy, 384-token evaluation and v1 scorer were used. Saved base outputs were reused. Accuracy was not aggregated, inspected or reported until all three evaluations and their integrity seals completed.

| Seed | Successful updates | Skipped | Training budget minutes used | Evaluation budget minutes used | Allocated GiB | Reserved GiB | Device-wide GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 1000 | 0 | 27.234/70 | 13.633/30 | 5.057 | 9.609 | 7.733 |
| 2027 | 1000 | 0 | 28.381/70 | 14.545/30 | 5.056 | 12.166 | 7.736 |
| 2028 | 1000 | 0 | 32.872/70 | 14.699/30 | 5.055 | 12.719 | 7.736 |

## v1 accuracies and family intervals

| Model/group | P1 accuracy | P2 accuracy | Pooled accuracy |
|---|---:|---:|---:|
| base | 49.0000% | 29.5000% | 39.2500% |
| scaled_random_seed_mean | 53.0000% | 28.6667% | 40.8333% |
| original_random_seed_mean | 53.6667% | 30.6667% | 42.1667% |
| scaled_random_seed_2026 | 57.5000% | 29.5000% | 43.5000% |
| scaled_random_seed_2027 | 51.5000% | 30.0000% | 40.7500% |
| scaled_random_seed_2028 | 50.0000% | 26.5000% | 38.2500% |
| original_random_seed_2026 | 54.0000% | 31.5000% | 42.7500% |
| original_random_seed_2027 | 54.0000% | 29.5000% | 41.7500% |
| original_random_seed_2028 | 53.0000% | 31.0000% | 42.0000% |

| Variant | Contrast | Difference pp | Family-only 95% interval |
|---|---|---:|---|
| p1 | scaled_random_minus_base | +4.000000 | [-1.052632, +10.000000] |
| p1 | scaled_random_minus_original_random | -0.666667 | [-4.696970, +4.445652] |
| p2 | scaled_random_minus_base | -0.833333 | [-6.078431, +4.166667] |
| p2 | scaled_random_minus_original_random | -2.000000 | [-8.550725, +4.259444] |
| pooled | scaled_random_minus_base | +1.583333 | [-2.136905, +5.500000] |
| pooled | scaled_random_minus_original_random | -1.333333 | [-5.121951, +2.719692] |

The optional paired seed-and-family interval against original random-353 is [-6.250000, +3.414634] pp. P1/P2 and original-random comparisons are secondary marginal analyses with no multiplicity-adjusted success claim. The random-353 comparison changes data, tokens and updates together and does not isolate their effects.

## Fixes, regressions and answer diagnostics

| Seed | Variant | Fixes vs base | Regressions | Net | Fixes vs matching random-353 | Regressions | Net | Parsed | Unparsed | Cap hits | Unparsed cap hits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | pooled | 41 | 24 | 17 | 38 | 35 | 3 | 392 | 8 | 8 | 7 |
| 2026 | p1 | 26 | 9 | 17 | 22 | 15 | 7 | 196 | 4 | 4 | 3 |
| 2026 | p2 | 15 | 15 | 0 | 16 | 20 | -4 | 196 | 4 | 4 | 4 |
| 2027 | pooled | 37 | 31 | 6 | 41 | 45 | -4 | 389 | 11 | 12 | 11 |
| 2027 | p1 | 21 | 16 | 5 | 21 | 26 | -5 | 198 | 2 | 3 | 2 |
| 2027 | p2 | 16 | 15 | 1 | 20 | 19 | 1 | 191 | 9 | 9 | 9 |
| 2028 | pooled | 25 | 29 | -4 | 31 | 46 | -15 | 390 | 10 | 10 | 10 |
| 2028 | p1 | 16 | 14 | 2 | 17 | 23 | -6 | 197 | 3 | 3 | 3 |
| 2028 | p2 | 9 | 15 | -6 | 14 | 23 | -9 | 193 | 7 | 7 | 7 |

## Secondary v2 scorer (new outputs only)

| Seed | Variant | v2 accuracy | v1→v2 verdict changes | v1 right→v2 wrong | v1 wrong→v2 right | Missing/invalid v2 | Continuation flags | Presentation flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2026 | pooled | 43.5000% | 0 | 0 | 0 | 10 | 0 | 1 |
| 2026 | p1 | 57.5000% | 0 | 0 | 0 | 3 | 0 | 1 |
| 2026 | p2 | 29.5000% | 0 | 0 | 0 | 7 | 0 | 0 |
| 2027 | pooled | 40.5000% | 1 | 1 | 0 | 16 | 0 | 0 |
| 2027 | p1 | 51.0000% | 1 | 1 | 0 | 3 | 0 | 0 |
| 2027 | p2 | 30.0000% | 0 | 0 | 0 | 13 | 0 | 0 |
| 2028 | pooled | 38.2500% | 0 | 0 | 0 | 17 | 1 | 0 |
| 2028 | p1 | 50.0000% | 0 | 0 | 0 | 3 | 0 | 0 |
| 2028 | p2 | 26.5000% | 0 | 0 | 0 | 14 | 1 | 0 |

The first clean numeric answer is retained while later continuation is flagged. No old base or original-random response was v2-scored. Therefore no mixed-scorer v2 gain over base is claimed. Every changed v2 verdict and its raw response is saved in the accompanying JSON.

## Clock and memory observations

Maximum wall-clock flag in the full training/evaluation phases: **0.000000 seconds**. Maximum across the entire scaled experiment, including earlier determinism/labeling/probe logs: **3.413894 seconds**.
Flag size is abs(successive wall delta − successive monotonic delta), not cumulative drift. Raw wall/monotonic deltas, timestamps and phase of the largest event are in the JSON. Budget use is the saved monotonic elapsed counter per supervised fresh process. Clock behavior remains unconfirmed; flags do not establish sleep or memory spill. Device-wide memory is sampled about once per second and may miss brief peaks. Allocated, reserved and whole-device counters are distinct.

| Phase | Wall flags | Suspend gaps | Sample gaps | Boot changes | Device peak GiB | Query errors |
|---|---:|---:|---:|---:|---:|---:|
| evaluate_2026 | 0 | 0 | 0 | 0 | 5.308 | 0 |
| evaluate_2027 | 0 | 0 | 0 | 0 | 5.308 | 0 |
| evaluate_2028 | 0 | 0 | 0 | 0 | 5.932 | 0 |
| train_2026 | 0 | 0 | 0 | 0 | 7.733 | 0 |
| train_2027 | 0 | 0 | 0 | 0 | 7.736 | 0 |
| train_2028 | 0 | 0 | 0 | 0 | 7.736 | 0 |

## Interpretation and stopped boundary

This is the prospectively specified scaling experiment motivated by the earlier post-hoc fixes/regressions pattern; that pattern was motivation, not supporting evidence. The 400 development questions cover only 37 previously inspected families. Neither the 63 reserved families nor GSM8K test was evaluated. Verified final answers do not certify every reasoning step. The observed batch-size sensitivity remains a reproducibility limitation; saved batch-8 base stayed the comparator.

The result applies to this combined increase in examples, supervised tokens and updates on this development set. Secondary results do not amend the two-interval success gate. All old protected files and the frozen training set are rechecked after analysis. No retry, setting change, extra seed, further training or evaluation was started. STOP.
