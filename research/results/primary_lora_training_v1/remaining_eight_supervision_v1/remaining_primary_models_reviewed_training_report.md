# Remaining primary training models — 2026-10-12

All eight models completed two epochs with the frozen base and unchanged pilot settings. Evaluation has not started.

| Model | Epoch 1 loss | Epoch 2 loss | Attempted / successful updates | Skips | Training min | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_seed_2027 | 0.2228 | 0.1585 | 150 / 150 | 0 | 4.72 | 5.013 | 9.426 |
| hard_seed_2028 | 0.2242 | 0.1588 | 150 / 150 | 0 | 3.93 | 5.013 | 9.148 |
| random_seed_2026 | 0.2083 | 0.1419 | 178 / 178 | 0 | 7.14 | 5.013 | 15.449 |
| random_seed_2027 | 0.2074 | 0.1419 | 178 / 178 | 0 | 5.56 | 5.010 | 12.811 |
| random_seed_2028 | 0.2048 | 0.1420 | 178 / 178 | 0 | 7.44 | 5.013 | 10.023 |
| easy_seed_2026 | 0.1990 | 0.1371 | 202 / 202 | 0 | 5.10 | 4.791 | 7.174 |
| easy_seed_2027 | 0.2001 | 0.1375 | 202 / 202 | 0 | 5.06 | 4.791 | 8.240 |
| easy_seed_2028 | 0.1974 | 0.1371 | 202 / 202 | 0 | 5.53 | 4.792 | 10.082 |

Hard/2027 device-wide sampled memory peak: **7.738281 GiB** (7924 MiB). There were 61 valid samples overall, 57 within the training window, and 0 query failures.

This exceeded 7.5 GiB but stayed below the reported 8 GiB capacity in the captured samples. The agreed condition for classifying the earlier 10.62 GiB value as allocator accounting was not met. Its cause remains unresolved; training settings were retained. This run did not establish the resident-memory peak of the earlier hard/2026 run.

The logger ran on hard/2027 only. Readings are device-wide, sampled every five seconds, and may miss brief peaks. Raw readings were recorded in MiB; the table uses GiB. PyTorch allocated and reserved values are retained exactly as reported.

Independent CPU validation passed for all eight adapters (705 protected input files checked). Every example was seen once per epoch; optimizer logs confirm constant learning rate, all successful updates, zero skips, and finite gradients. All 112 tensors in each final adapter are finite. The observer timer agrees with the trainer time to within 0.01 seconds.

Settings: rank 4, alpha 8, dropout 0, q_proj/v_proj, AdamW lr 1e-4, constant schedule, weight decay 0, two epochs, microbatch 1, accumulation 4, FP16 with gradient scaling, checkpointing, prompt masked, sequence limit 1,024. No intermediate adapter checkpoints or checkpoint selection.

The three seeds are paired across hard, random and easy. Together with the earlier hard/2026 adapter, all nine primary adapters are ready for a separately authorized evaluation. The reserved 63 families remain untouched.
