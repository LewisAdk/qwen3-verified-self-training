# Supervised gold positive-control plan — October 13, 2026

This is a **supervised-learning control**, not self-training. The RQ2 hard-minus-random result is already known to be inconclusive. This new control is planned before any gold training or evaluation outcomes exist. It asks whether the unchanged training setup can improve development accuracy when given GSM8K reference solutions for the same hard problems.

## Frozen dataset and planned training

Use exactly the hard set's 300 problem IDs, in the same initial order, from pinned GSM8K training revision 740312add88f781978c0658806c59bc2815b9866, shuffled with seed 2026. Replace each selected model-generated solution with its GSM8K reference explanation. Remove only <<...>> calculator annotations and outer whitespace, and format its final line as #### answer, retaining the reference answer representation. Keep the references' wording and internal formatting. Do not filter, resample, shorten or token-match the reference solutions.

Prepared set: results/training_sets_gold_control_v1/gold.jsonl. Supervised tokens include EOS and use the exact existing prompt and target formatting. Gold has 32,206 supervised tokens per epoch versus hard's 32,879: -673 tokens (-2.047%). Two epochs give 64,412 supervised-token exposures. Maximum full sequence: 847 tokens; limit remains 1,024. Both gold and hard have 300 examples and 150 planned optimizer updates per model. This matches problems, example counts, epochs and steps, **not total supervised tokens**. Gold-minus-hard changes supervision source, style and length together; it does not isolate reasoning quality alone.

Planned seeds are 2026, 2027 and 2028, paired with the existing hard models. Reuse the immutable train_primary_lora_v1.py training operations and the identical pilot settings: LoRA rank 4, alpha 8, dropout 0, q_proj/v_proj, bias none; AdamW lr 1e-4, weight decay 0, constant learning rate with no scheduler; two epochs, microbatch 1, gradient accumulation 4 with the same end-of-epoch flush; FP16 with the same gradient scaler defaults; gradient clipping 1.0; gradient checkpointing with use_reentrant=False; prompt tokens masked; maximum length 1,024. Pin base snapshot ea980cb0a6c2ae4b936e82123acc929f1cec04c1. Preserve final-adapter-only saving after epoch two; count attempted, successful and skipped updates separately. Do not choose a best checkpoint or resume from an intermediate training checkpoint.

The existing runner accepts only its frozen hard/random/easy inputs. A future separate gold entry point must supply these validated gold inputs while reusing the immutable training operations, without editing the original runner or silently substituting the hard dataset. Confirm that integration and the matching memory check before training. No training entry point is executed by this preparation stage. Stop on failed safety checks; do not truncate data or alter settings.

## Primary estimand and decision

Primary: pooled **gold minus base**, equal-weight mean of the three gold adapter accuracies, on the same 400 GSM-Symbolic development questions: the saved 20 templates each for P1/P2, instances 5–14, covering 37 original families. Reuse the saved base results after question/gold/contract validation. The gold control "works" only if the entire paired 95% family-bootstrap interval is above zero. An interval wholly below zero indicates worse performance; otherwise the control is inconclusive. An endpoint exactly equal to zero does not pass. Treat a borderline interval cautiously.

Reuse the existing procedure: 10,000 percentile bootstrap replicates, seed 2026; resample 37 original_id families with replacement; keep a family's P1/P2 questions together; share draws across all gold, hard and base models; divide weighted correctness totals by weighted question totals before averaging the three seeds. Retain the existing rejection of draws missing a variant and linearly interpolated 2.5th/97.5th percentiles. Intervals condition on these trained adapters; do not bootstrap training seeds or present them as population training-seed uncertainty.

## Secondary reporting and evaluation

Secondary: **gold minus hard**, comparing three-seed means with the same shared family draws. Report per-seed accuracies and paired fixes/regressions versus base and versus the same-seed hard adapter; report P1/P2 separately as secondary. All secondary intervals are marginal and unadjusted for multiple comparisons. No majority-vote group verdict. The prior 300-example experiment gate is not changed by this separate control.

Evaluation remains batch 8, FP16, greedy decoding, 384 new tokens, evaluation seed 2026, exact existing prompt/parser/stopping and legacy string-Decimal scoring. Reuse the saved base and hard outputs only after their question identities, pinned revisions, evaluator contracts and source hashes match. Retain numeric-equality rescoring as a diagnostic without replacing the primary scoring rule. Generate all three gold adapters' outputs before analysis; do not select seeds or checkpoints using evaluation results.

## Preservation, limitations and stage boundary

The same 37 development families have already been inspected in prior research; this control is a prospective follow-up on development data, not a reserved-test claim. The **63 reserved original families remain untouched**. Reference solutions are dataset supervision and are not automatically proof of flawless reasoning. The original targeted 20-solution review remains a limitation of the hard comparison set; do not modify it in response to this control.

The earlier training-memory observation remains unresolved. Report native PyTorch counters separately from any sampled device-wide usage; do not assert that the earlier 9–15 GiB reserved counters establish RAM spill. Any further memory investigation is a separate documented check, without changing experimental settings mid-run.

Use only new preparation, training, evaluation and analysis output paths. Preserve all existing scripts, results, checkpoints and adapters. At this stage, **prepare the dataset and plan, then stop before training**. After later confirmation, train all three gold seeds with validated unchanged operations, evaluate the same 400 questions, run this analysis, then stop without another experiment.
