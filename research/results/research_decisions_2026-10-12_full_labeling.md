# Labeling and training decisions — October 12, 2026

Recorded before the full labeling pass and before any new training.

- Full labeling: 1,100 unused GSM8K training problems, shuffled indices 536–1635 inclusive; frozen dataset revision 740312add88f781978c0658806c59bc2815b9866 and shuffle seed 2026.
- Four sampled base-model attempts per problem, temperature 0.7, top-p 0.9, FP16, batch 8; same two-shot prompt, parser, numeric verification and stopping behavior as the pilot. No extra attempts and no training in this pass.
- Deliberate change: max_new_tokens = 384 instead of the pilot's 256. Saved pilot diagnostics show 16 token-limit cut-offs, with 9/16 in 0/4 or 1/4 problems and all 16 in 0–3/4 problems. This meets the user's majority rule.
- Use only the new 384-token labels for training selection. The 256-token pilot remains a separate diagnostic cohort; do not pool its labels into the selection pool.
- “hard” = 1–3 correct answers out of four; “easy” = 4/4; 0/4 supplies no verified solution under this protocol.
- Three training groups: random verified (sample from all new problems with at least one verified attempt), hard (1–3/4), and easy (4/4).
- 300 unique problems per group; one numerically verified solution per problem. Random membership may naturally overlap the hard/easy arms; groups are interventions drawn from the same eligible source pool.
- Match total training tokens across the three groups. Before training, choose and freeze the exact token accounting and matching procedure, solution-selection rule, sampling seed and training schedule; no training is authorized by this labeling run. Do not silently replace examples, change labels or relabel 0/4 to fill a short arm.
- Three training seeds per group (nine trained adapters). Seed identities and all training settings must be frozen before training.
- Evaluate on the existing 37 development original families. All 63 reserved original families stay untouched; no reserved evaluation questions are added or inspected.
- If either selected category has fewer than 300 unique eligible problems, report the shortfall and settle any pool extension before training.
- Difficulty labels are noisy and specific to the prompt, sampling settings and 384-token budget. Numeric final-answer verification does not certify all reasoning steps. No new model-training success claim is made here.
