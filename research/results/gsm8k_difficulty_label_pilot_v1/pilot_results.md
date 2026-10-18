# GSM8K difficulty-labeling pilot — October 12, 2026

Completed 200 unused training questions at shuffled indices 336–535, with four sampled base-model attempts each. No training or extra attempts were performed.

| Correct out of four | Problems | Percent | Interpretation |
|---|---:|---:|---|
| 0/4 | 16 | 8.0% | Unsolved in these four attempts |
| 1/4 | 18 | 9.0% | Difficult but solvable under this protocol |
| 2/4 | 26 | 13.0% | Intermediate |
| 3/4 | 34 | 17.0% | Intermediate |
| 4/4 | 106 | 53.0% | Easy under this protocol |

- At least one verified solution: 184/200.
- Strict hard category, 1/4: 18/200.
- Non-easy solvable category, 1–3/4: 78/200.
- Parsed answers: 782/800; numerically correct answers: 596/800.
- Timed generation and decoding: 14.74 minutes; peak reserved GPU memory: 4.38 GiB.
- Wall time across the GPU check, validation, full run and final validation: 15.37 minutes.

## Pool-size planning

The following are rough total source-pool sizes needed for 300 or 500 unique training questions if the observed fractions persist. They are not guarantees and do not estimate the training effect.

| Selection category | For 300 questions | For 500 questions |
|---|---:|---:|
| any_verified | 327 | 544 |
| non_easy_solvable | 770 | 1283 |
| strict_hard | 3334 | 5556 |

There are at most 7137 GSM8K training indices from 336 onward, including this pilot; exact duplicate or prior-use exclusions may reduce the usable pool.

## Verification and limitations

- A full outside-WSL archive was verified before this script was built or run. Backup details are in the separate receipt.
- Eight synthetic tests passed; actual WSL preflight, the first eight-attempt GPU check and both saved-record validations passed.
- The full run validated and skipped the first saved batch. All 800 candidate keys are unique, with four attempts per question.
- All protected old scripts, checkpoints and saved-result hashes still match. All 63 reserved families were preserved.
- Settings: frozen Qwen3-1.7B-Base snapshot, same two-shot prompt and original numeric verifier, FP16, batch 8, temperature 0.7, top-p 0.9, 256 new tokens.
- Four-attempt labels are noisy and specific to this prompt, sampler and token budget. Zero correct attempts does not mean impossible. Repeated sampled responses are legitimate and retained.
- Numerical outcome verification does not certify every intermediate reasoning step. No trained models were produced or evaluated in this phase.
- A 200-question pilot cannot itself supply 300–500 unique training questions. Decide selection categories, matched training budgets and the development/final-test family partition before observing any new training results.
