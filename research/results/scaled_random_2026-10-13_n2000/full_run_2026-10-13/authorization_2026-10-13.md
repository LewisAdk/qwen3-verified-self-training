# Full training and evaluation authorization — 2026-10-13

The user approved all three frozen 2,000-example training runs, seeds 2026, 2027 and 2028, sequentially in fresh processes using the unchanged RQ2 engine/settings. Each must finish exactly 1,000 successful optimizer updates and zero skipped updates within 70 minutes. A safety-only UpdateCounter subclass invokes the original step once and immediately stops on any skipped update; the numerical execute function remains unchanged.

After all training runs succeed, evaluate each final adapter sequentially on the same 400 development questions at batch size 8, FP16 greedy decoding, 384 tokens, v1 primary scoring and the already-tested v2 secondary scorer. Each evaluation has a 30-minute budget. Reuse the saved batch-8 base. No aggregate accuracy is computed, inspected or reported until all three complete evaluation jobs and integrity seals exist.

Then execute the frozen analysis exactly: 10,000 family-only draws and 10,000 seed-first/shared-family draws with random.Random(2026), equal seed means, both lower endpoints strictly above zero for success, and all drafted secondary analyses. No old outputs are v2-scored. Include the maximum wall-clock flag size (absolute successive wall/monotonic delta discrepancy) and its raw deltas. Preserve all logs, original files, frozen choices and the 63 reserved families. GSM8K test stays untouched.

Stop on OOM, NaN/non-finite loss or gradients, any skipped update, budget overrun, crash or protected-file hash mismatch. No retry, resume or setting change is authorized. Inspect progress every 15–20 minutes and report only errors, stop conditions or completion. Stop after the final analysis. The draft and amended plan remain unchanged.

New approval plumbing: full_run_2026-10-13/src; immutable existing frozen set/probe; training_seed_2026/2027/2028; evaluation; analysis, under the existing dated scaled-random root. Fresh supervised job logs under full_run_2026-10-13/jobs. No original engine/evaluator/analyzer file is edited.
