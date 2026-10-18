# RQ2 analysis plan — October 12, 2026

Saved before generating or inspecting RQ2 evaluation outcomes. This plan concerns the nine final primary adapters, three paired seeds (2026, 2027, 2028) in each of hard, random verified, and easy. No training settings or prepared sets will change.

## Primary comparison and decision

The primary estimand is pooled GSM-Symbolic accuracy for hard minus random verified, in percentage points, averaging the three trained adapters in each group with equal seed weights. Evaluate exactly the existing 400 development questions: 20 saved templates each for P1 and P2, instances 5–14, covering the same 37 original families. P1 and P2 questions contribute equally to pooled accuracy. The 63 reserved original families remain untouched.

Use the existing paired percentile original-family cluster bootstrap: 10,000 replicates, bootstrap seed 2026, sample 37 original_id families with replacement, keep a family's P1/P2 questions together, and use the same family draw for every adapter and the base. Each replicate uses weighted correct totals divided by weighted question totals, then averages the three seed accuracies within each group. Retain the existing procedure's rejection of draws lacking either variant, and its linearly interpolated 2.5th/97.5th percentiles.

Hard wins only if the entire primary 95% interval is above zero. An interval entirely below zero favors random. Otherwise the result is inconclusive; an endpoint exactly equal to zero does not exclude zero. A borderline interval remains borderline, given only 37 development families.

Either direction excluding zero meets the previously agreed gate for considering the secondary, existing 300-example training comparison. That experiment is not authorized by this evaluation request and will not be started automatically.

## Secondary reporting

Report hard minus easy, random minus easy, and each of hard, random and easy minus the reused base as secondary comparisons, using the same shared bootstrap draws. Also report P1/P2 separately, all nine individual adapter accuracies, their per-seed differences against base, and paired hard-minus-random differences for each seed. Report fixes/regressions/net changes against base for each adapter and within-seed hard versus random; provide group-average flip counts as descriptive summaries. The group mean has no single binary per-question verdict, so do not invent one by majority vote.

All intervals other than the prespecified primary interval are secondary, marginal and unadjusted for multiple comparisons. These intervals condition on the three trained adapters and assess variation across selected development families; they do not estimate population training-seed uncertainty. Additional instances do not add independent families. These development questions have previously been inspected for the earlier training-versus-base experiment. Do not present this as performance on the reserved final test set or interpret training losses as evaluation outcomes.

## Evaluation and preservation

Reuse the existing saved base responses only after verifying all 400 question/template/instance/family/gold identities, pinned dataset revisions, prompt, evaluator contract, base snapshot and generation configuration. Keep batch size 8, FP16, greedy decoding, 384 new tokens, evaluation seed 2026, the exact saved prompt, parser, per-row stopping and legacy string-Decimal equality. Pin the sole base snapshot ea980cb0a6c2ae4b936e82123acc929f1cec04c1. Verify final-adapter hashes against completed training records.

Write new scripts, manifests, batches and reports only. Resume completed batches only when their frozen contract and input hashes match; stop on OOM or evaluation failure without adjusting settings. Generate all nine adapters before running the analysis. Keep the legacy scoring result primary; include numeric-equality rescoring as a diagnostic of answer-format effects without replacing the saved rule.

## Recorded design differences and limitations

Hard: 300 examples, 32,879 supervised tokens per epoch, 150 optimizer updates over two epochs. Random verified: 353 examples, 32,619 supervised tokens per epoch, 178 updates. Easy: 403 examples, 32,592 supervised tokens per epoch, 202 updates. The primary sets match supervised-token totals within ±1%; example counts and optimizer steps differ. Easy received 34.67% (approximately 35%) more optimizer updates than hard; random received 18.67% more. This is a comparison under supervised-token matching, with these remaining differences recorded. Groups can overlap: hard/random 101 problems, random/easy 140, hard/easy zero.

Retain all verified solutions in all groups, with the earlier 20-solution reasoning review recorded as a limitation. Do not filter after seeing evaluation outcomes.

Memory remains an unexplained observation. Hard/2027's read-only, five-second logger peaked at 7,924 MiB (7.738281 GiB) of sampled device-wide usage, including other applications; brief peaks can be missed. PyTorch peak allocated counters were approximately 4.79–5.01 GiB, while some reported reserved counters were 9–15 GiB on the 8 GiB card. These readings do not establish system-RAM spill, its absence, or a physical resident-memory peak for every run. All nine runs completed with finite adapters and zero skipped updates. Preserve settings and investigate memory separately before larger future experiments.

After evaluation and analysis, stop. Do not train the conditional secondary models or begin any other experiment.
