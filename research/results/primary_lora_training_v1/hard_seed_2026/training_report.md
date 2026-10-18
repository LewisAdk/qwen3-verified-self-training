# First primary training model — October 12, 2026

Completed hard / seed 2026 only. The other eight primary models have not started; user confirmation is required to continue.

## Memory probe

- Twelve longest distinct examples, maximum 849 tokens.
- Peak reserved memory: 5.168 GiB; allocated: 4.996 GiB.
- Attempted / successful / skipped updates: 3 / 3 / 0.
- Probe adapter discarded; full run initialized fresh from the pinned base in a separate process.

## Hard / seed 2026

- 300 problems, two epochs, 600 example exposures; 65,758 supervised tokens seen.
- Epoch mean losses: [0.22847311607872448, 0.15858708488444487].
- Overall mean training loss: 0.193530 (same per-example averaging as the pilot; not an evaluation accuracy result).
- Attempted updates: 150; successful: 150; skipped: 0.
- Peak reserved memory: 10.617 GiB; allocated: 5.013 GiB.
- Maximum sequence: 849; training time: 5.12 minutes.
- Supervised tokens contributing to successfully executed updates: 65758.
- Only the final two-epoch adapter was saved. No checkpoint selection or intermediate training state.

## Validation and preserved scope

Seven CPU tests, real-data preflight, GPU memory test and completed-run checks passed. Step-log counts reproduce the metrics; final adapter checksums match. All protected scripts, prepared sets, checkpoints, prior results and family split hashes still match. Learning rate stayed constant at 1e-4; all frozen pilot settings were preserved. The 63 reserved families were untouched. No evaluation has run and no model-quality claim follows from training loss.

The new runner supports paired seeds 2026/2027/2028 across hard/random/easy. Later runs require explicit confirmation. An interrupted run is preserved and must restart from the pinned base in a new folder. No settings were changed to make the memory test fit.
