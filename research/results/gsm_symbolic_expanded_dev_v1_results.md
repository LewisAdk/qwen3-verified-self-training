# Expanded GSM-Symbolic development results — 2026-10-12

Inconclusive under the prespecified primary criterion.

Primary endpoint: pooled P1/P2 three-adapter mean minus base, instances 5–14 only.

Legacy scoring: +3.42 pp (95% CI -1.26 to +8.00).
Numeric scoring: +3.42 pp (95% CI -1.26 to +8.00).

| Model | New P1 accuracy | New P2 accuracy | New pooled accuracy |
|---|---:|---:|---:|
| base | 49.00% | 29.50% | 39.25% |
| 2026 | 51.00% | 32.00% | 41.50% |
| 2027 | 52.50% | 33.50% | 43.00% |
| 2028 | 53.00% | 34.00% | 43.50% |
| seed_mean | 52.17% | 33.17% | 42.67% |

## Numeric-formatting audit

| Model | Old changed / 200 | New changed / 400 | All changed / 600 |
|---|---:|---:|---:|
| base | 0 | 0 | 0 |
| 2026 | 0 | 0 | 0 |
| 2027 | 0 | 0 | 0 |
| 2028 | 0 | 0 | 0 |

Earlier GSM8K evaluation files (audited separately; repeated evaluations are not pooled):

Source inspection confirms numeric Decimal comparison in the generator and sequential GSM8K evaluators, but string comparison in the batched GSM8K and GSM-Symbolic evaluators. The audit keeps each original answer extractor unchanged.
- gsm8k_baseline_pilot.jsonl: 0/20 changed verdicts.
- gsm8k_dev_baseline.jsonl: 0/200 changed verdicts.
- gsm8k_dev_batch16.jsonl: 0/200 changed verdicts.
- gsm8k_dev_batch16_lora_pilot_v1.jsonl: 0/200 changed verdicts.
- Original base-model generated candidates: 0/100 changed verdicts (separate from evaluation).
- Accepted training solutions: 43/43 numerically correct; original training data retained.

Paired classification changes attributable to numeric scoring (new primary instances):

- Seed 2026: none.
- Seed 2027: none.
- Seed 2028: none.

## Observed precision

Legacy primary interval width: 9.26 pp; half-width: 4.63 pp.
Numeric primary interval width: 9.26 pp; half-width: 4.63 pp.
These describe observed precision; neither is a formal minimum detectable effect. Future power calculations need additional assumptions and a chosen power target.

## Run verification

- Base snapshot: ea980cb0a6c2ae4b936e82123acc929f1cec04c1; one cached folder was found before generation. Historical commit was absent from training metadata.
- Preflight passed for all four saved evaluator contracts, saved split and adapter hashes.
- All protected input hashes and the prespecified-rule hash still match after inference.
- Original scripts, checkpoints, pilot results and reserved families were preserved.
- Settings: batch 8, FP16, greedy, 384 new tokens; no automatic memory fallback.
- Each model completed 200 P1 and 200 P2 questions; complete batches validated before analysis.
- Eight synthetic tests and the installed-stack independent-stopping CPU check passed; an actual post-run resume check skipped all completed batches without new inference.
- The launch log wrapper failed; per-batch timing and GPU-memory measurements were saved by the evaluator. Post-run analyses have separate new logs.
Observed maximum reserved GPU memory: 4.80 GiB. Sum of timed generation batches: 51.93 minutes (excludes loading/preparation).

## Limits

Exploratory development results from 37 original families and three fixed trained adapters. The pooled primary rule was recorded before generation. P1/P2 and combined 0–14 reports are secondary. A borderline interval exclusion warrants caution; percentile bootstrap coverage is imperfect with a small number of families. Secondary intervals are not adjusted for multiplicity. More instances do not create additional independent families, and these results do not demonstrate RSI.
