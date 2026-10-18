# Scaled random self-training — approved amendment, 2026-10-13

This is a new frozen amendment to [the unchanged draft](scaled_random_self_training_plan_2026-10-13.md), SHA-256 `68408d1b22f6807a53ae45a53835ccedac11d48803b75e4f29eedec110da4a58`. The draft is incorporated by reference except for the explicit amendments and execution authorization below. Its prior “plan only,” undecided-target and approval-pending wording is superseded only within the authorized stages. All other settings, safeguards, seeds, scoring policies, paths and stop conditions remain in force.

## Frozen amendments

1. **N = 2,000**, with one frozen randomly selected verified example per distinct problem. Initial labeling pool is shuffled GSM8K train positions **1636–4035 inclusive (2,400 positions)**. If fewer than 2,000 eligible problems are found after completing the pool, append full contiguous **200-position blocks** using the draft’s global batch-index seed scheme. Complete each block before deciding whether another is needed; never exceed position 7472.
2. **Success requires BOTH** the pooled family-only 95% bootstrap interval and the pooled seed-and-family 95% bootstrap interval to have lower endpoints **strictly above zero**. Both use 10,000 replicates and random.Random(2026), exactly as specified in draft section 5. The three training seeds remain **2026, 2027, 2028**.
   - If only family-only passes: report **“family-only pass, not robust to seed variation”**, not success.
   - A seed-and-family-only pass is also not success.
   - Zero endpoints do not pass. The primary estimand, family-only method, seed-and-family sampling order, and remaining secondary analyses stay as drafted. The former “secondary results cannot decide success” wording does not apply to the seed-and-family criterion now explicitly included in the success gate.
3. The drafted v2 scorer and synthetic tests are approved. Retain the first clean answer and flag later continuation separately. Do not apply v2 to old outputs.
4. The drafted stop budgets and new paths are accepted: **300 minutes** for the initial 2,400-position labeling phase, **30 minutes** for the base determinism rerun and future evaluations, and **70 minutes per full training seed** if separately approved later. The initial phase budget is cumulative across pauses/resumes. Top-up blocks retain the drafted approximately 15.7-minute estimate; no arbitrary extra attempt or adaptive settings change is allowed.
5. The dataset, model, prompt, four-attempt labeling, candidate-selection RNGs, unchanged training engine, 1,024-token length guard, no silent replacements, longest-12-example probe and preservation requirements are unchanged.

## Authorized execution order and mandatory stop

1. Freeze this amendment and protected hashes. Implement/test new dated CPU wrappers and v2 scorer; prove unchanged training-engine identity and analyzer equivalence; reproduce original saved intervals exactly.
2. Run and report the **batch-4 base determinism check** on the same 400 development questions. Saved batch-8 base remains the primary comparator.
3. Label the initial fresh pool and any necessary full 200-position top-up blocks; validate all attempts, eligibility and yield.
4. Freeze exactly 2,000 problem IDs, candidate choices and source order. CPU-tokenize with the original training prompt, masking and EOS; stop on any sequence exceeding 1,024. If this passes, run the unchanged-engine memory probe on the 12 longest selected examples with three successful updates.

**STOP AFTER THE MEMORY PROBE. Full training runs and scaled-adapter evaluations are NOT authorized.** The executable entry point will refuse full training in this approval stage. Report determinism, verified yield, full/target token and length statistics, selected difficulty mix, and allocated/reserved/device-wide probe memory separately. Preserve and report failures; no settings are automatically altered to fit.

The six positive fixes-minus-regressions observations remain motivation only, not evidence for this new experiment. The reused 37 development families limit external validity; the **63 reserved families and GSM8K test stay untouched**. No reserved question content is inspected.

## Frozen execution locations

WSL new root: `/home/lewis/rsi-research/results/scaled_random_2026-10-13_n2000/`.

New source files and tests live under this root’s `src/`; labeling blocks under `labeling/`; exact set under `frozen_set/`; probe under `longest_sequence_probe/`; base check under `determinism_base_batch4/`. Windows user-facing plan, validation/report copies use new dated filenames under this chat’s `outputs/`. Existing files are never overwritten.

Before/after SHA-256 inventories include the existing research files and manifest-declared external protected inputs, all previous Windows output copies and the unchanged draft/diagnostic. New run artifacts are excluded from the old-file inventory but are individually frozen and hashed. The manifest key `split` resolves to `results/gsm_symbolic_pilot_split_v2.json`.

## Implementation validation requirements

CPU tests must cover malformed parser inputs, correct thousands grouping, currency/unit presentation, expressions, continuation, missing first answers, paired bootstrap aggregation, fixed versus resampled seeds, length-boundary rejection, deterministic candidate/set selection, output overwrite rejection, global labeling seed continuity and probe-only authorization. Existing saved v1 parser outputs may be verified for identity; v2 may run only on synthetic tests and new generated records.

Do not change the original engine/analyzer. Any narrow analyzer generalization must preserve its original family aggregation and percentile code and reproduce the saved original intervals exactly. A failure of an approved safeguard stops the stage; it does not authorize an amended method.

