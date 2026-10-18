# Primary token-matched sets — October 12, 2026

Prepared only; no training performed or authorized. Hard is an exact copy of the existing 300-problem set. Each comparison group uses unchanged fixed solution choices and a new seeded shuffled prefix. No reasoning-based filtering.

| Group | Examples | Supervised tokens / epoch | Gap vs hard | Mean solution tokens | Max solution tokens | Optimizer updates / model |
|---|---:|---:|---:|---:|---:|---:|
| hard | 300 | 32879 | +0.000% | 108.60 | 342 | 150 |
| random | 353 | 32619 | -0.791% | 91.41 | 342 | 178 |
| easy | 403 | 32592 | -0.873% | 79.87 | 238 | 202 |

Supervised totals include EOS; solution lengths include the original target space/newline formatting but exclude EOS. Two epochs double the listed totals. Optimizer updates are proposed, not performed, using two epochs and accumulation four with an end-of-epoch flush.

| Group | 1/4 | 2/4 | 3/4 | 4/4 | Max full sequence | Shuffle attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard | 58 | 95 | 147 | 0 | 849 | 0 |
| random | 32 | 44 | 77 | 200 | 849 | 1 |
| easy | 0 | 0 | 0 | 403 | 743 | 1 |

Overlap counts: {'hard_random': 101, 'hard_easy': 0, 'random_easy': 140}. No duplicates within a group. Hard/easy categories are disjoint; random can overlap either.

Quality limitation: retain the review findings without removing examples (15 apparently sound, two clear cancelling-error explanations, one borderline substantive explanation, two minor wording defects). The targeted assistant review is not a pool error-rate estimate or independent human annotation.

Training settings and original 37-development / 63-reserved family split are preserved. Example counts and optimizer updates differ by design while supervised token budgets match within 1%.

Primary gating contrast: the paired 95% original-family bootstrap interval for the three-adapter mean hard minus random excludes zero, in either direction. Only then run the existing 300-example secondary sets. Other pairwise contrasts are secondary. Stop for user confirmation before any training.
