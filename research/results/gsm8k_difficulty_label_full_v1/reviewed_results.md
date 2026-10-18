# Full GSM8K difficulty labeling

Completed 1,100 new training problems at shuffled indices 536–1635, four attempts each, max_new_tokens 384. The 256-token pilot is excluded from training selection. No training or extra attempts were performed.

| Correct / 4 | Problems | Percent |
|---|---:|---:|
| 0/4 | 86 | 7.82% |
| 1/4 | 80 | 7.27% |
| 2/4 | 135 | 12.27% |
| 3/4 | 224 | 20.36% |
| 4/4 | 575 | 52.27% |

Verified eligible: 1014; hard (1–3/4): 439; easy (4/4): 575.
Enough eligible problems by count for all three 300-problem groups: yes. Constructing and checking the matched training-token budgets remains a separate step.
Parsed attempts: 4343/4400; numerically correct: 3322/4400.
Token-limit cut-offs by 0/4–4/4 bucket: [7, 5, 7, 7, 0].
Generation/decode: 86.42 minutes; peak reserved memory: 4.99 GiB.

Ten synthetic tests, WSL preflight, actual first-batch saved-record validation, resume and complete validation passed. All protected hashes still match, including pilot batches, original scripts, checkpoints, previous results, and the 63 reserved-family partition.

Each of the 1,014 eligible problems has at least one numerically verified candidate listed in correct_candidate_ids; the 86 unsolved problems have none. One solution per eligible problem still needs to be selected. Group selection and matched-token training have not been performed. Freeze their exact procedures and the three training seeds before training. Difficulty is a noisy four-attempt label, and numerical verification does not certify every reasoning step.
