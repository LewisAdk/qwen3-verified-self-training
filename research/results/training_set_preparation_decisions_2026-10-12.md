# Training-set preparation decisions — October 12, 2026

User choices fixed before materializing the sets; CPU preflight has been inspected without changing the sampling policy. No training is authorized.

- Preparation seed 2026, with separate deterministic streams for candidate selection, hard sampling, random/easy draws and review sampling. Reuse identical sets across all three training seeds per group.
- Choose one correct candidate uniformly once per eligible problem using an index-specific seeded stream. Never select the shortest candidate or resample candidates while matching tokens.
- Use only the new 384-token cohort, indices 536–1635. Hard means 1–3/4; easy means 4/4; random is sampled from all 1,014 verified-eligible problems.
- Hard first: uniform random 300 of the 439 hard problems. Random and easy: uniform 300-problem draws, accept the first supervised-token total within ±3% of hard, at most 1,000 draws per group.
- On failure, export the closest draw as an explicitly unmatched diagnostic set. Do not approve it for training or tune the hard selection to force a match. Record both the chosen draw and all 1,000 attempted draws.
- Token accounting exactly follows the pilot trainer: tokenize a leading space, stripped solution, newline, and append one supervised EOS. Prompt tokens are masked from loss. Also report solution tokens without EOS and total full-sequence lengths. No truncation, padding to manufacture supervised tokens, or duplicated problems.
- Preserve rank 4, alpha 8, q/v projections, dropout 0, learning rate 1e-4, two epochs, microbatch 1, accumulation 4, AdamW with weight decay 0, gradient clipping 1, FP16, gradient checkpointing and the 1,024-token sequence guard. These settings imply 150 optimizer updates for 300 problems.
- Review 15 randomly sampled selected 1/4 solutions and five selected 2–3/4 solutions. Record sound reasoning, substantive errors and uncertainty. This targeted sample does not estimate the whole pool's flaw rate. Remove nothing automatically.
- Random membership may overlap hard/easy membership. The original 37 development / 63 reserved family partition stays unchanged.
- Preflight found no matching draw for either comparison group. Random's closest draw is 330 (−10.33%); easy's closest draw is 778 (−22.66%). Easy's maximum possible 300-problem total with fixed candidates is 30,995, below the permitted lower bound 31,892.63. These failures are reported, not repaired by changing the agreed protocol.
