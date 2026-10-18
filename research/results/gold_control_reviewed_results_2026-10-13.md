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

## Verified training and evaluation execution

All three gold models trained from the frozen base using the unchanged original training execution function and settings. Each saw all 300 references in each of two epochs: 600 exposures, 64,412 supervised tokens and 150 successful optimizer updates. No full training or evaluation hit an OOM, non-finite value, skipped update, 10.24-minute training limit or 26-minute adapter evaluation limit. No extra experiment followed this analysis.

| Seed | Epoch 1 loss | Epoch 2 loss | Attempted / successful updates | Skips | Training min | Adapter evaluation min | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 0.5619 | 0.4828 | 150 / 150 | 0 | 6.67 | 15.03 | 5.009 / 14.768 |
| 2027 | 0.5590 | 0.4836 | 150 / 150 | 0 | 5.46 | 13.76 | 5.009 / 14.688 |
| 2028 | 0.5624 | 0.4841 | 150 / 150 | 0 | 5.82 | 14.07 | 5.009 / 7.812 |

Training time follows the original timer, excluding loading and preflight. Adapter evaluation time includes model loading and generation, excluding the shared preflight. Reported reserved counters are retained as native PyTorch values, with the prior memory observation unresolved. Losses describe fitting the reference targets; they are not evaluation outcomes.

The memory probe used the 12 longest gold examples (maximum 847 tokens), achieved three successful updates with zero skips, peaked at approximately 4.992 GiB allocated and 5.160 GiB reserved, and saved no adapter.

## Independent checks and response diagnostics

CPU checks verified all 112 tensors in each final adapter are finite, every selected problem was used once per epoch, the original core trainer and approved execution-source hashes match, and all 2,800 records have matching question/gold/family identities. Base and hard batches are byte-identical to their saved originals. Independent direct cluster-difference calculations reproduced both bootstrap intervals. All protected input hashes match; the prepared data and analysis plan remain unchanged.

| Model | No parsed answer / 400 | Saved tokens at 384 limit |
|---|---:|---:|
| base | 12 | 20 |
| gold_seed_2026 | 20 | 20 |
| gold_seed_2027 | 13 | 14 |
| gold_seed_2028 | 16 | 18 |
| hard_seed_2026 | 11 | 11 |
| hard_seed_2027 | 13 | 14 |
| hard_seed_2028 | 8 | 10 |

All 400 questions are included per model. Unparsed responses count as incorrect under the fixed rule. These counts are descriptive diagnostics; no longer-budget evaluation, selective exclusions or alternate primary scoring was run.

## Operational correction and interpretation

The initial v1 memory-probe launch failed before any GPU training because importing the CPU preparation module disabled CUDA visibility. The corrected v2 entry point extracts only the reference-cleaning function and regexes, preserving CUDA visibility. Its new regression test and real-environment check passed before the fresh v2 probe and trainings. The failed v1 files are preserved; the original trainer, settings, gold references and plan were not changed.

The gold control did not meet the prespecified works criterion. Its negative point estimate does not by itself establish degradation versus base, since the primary interval includes zero. The secondary gold-minus-hard interval is wholly negative but only narrowly excludes zero; treat it as a borderline, unadjusted development finding conditional on these three adapters. This control does not diagnose a specific cause or demonstrate self-training improvement.
