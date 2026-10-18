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

## Reviewed memory qualification

The full-run raw counters are **5.013 GiB peak tensor allocation** and **10.617 GiB peak allocator reservation**, while the card reports approximately 8 GiB total. The short memory probe reserved only 5.168 GiB. Full training finished without a CUDA out-of-memory exception. These raw values are preserved; they do not establish that the entire run stayed inside dedicated VRAM.

The installed PyTorch 2.11 implementation reports reserved_bytes.all.peak, and a fresh process uses the native allocator with no PYTORCH_ALLOC_CONF/PYTORCH_CUDA_ALLOC_CONF override. Reserved memory measures allocator-managed memory; allocated memory measures tensor memory. See [PyTorch memory-management documentation](https://docs.pytorch.org/docs/main/notes/cuda.html).

Windows driver-managed shared-memory fallback is a possible explanation, but it was not observed directly during this WSL run and is not established as the cause. NVIDIA documents that its Windows driver can allow applications to continue through GPU-memory pressure using shared memory: [NVIDIA system-memory fallback](https://nvidia.custhelp.com/app/answers/detail/a_id/5490). Driver 610.74 reports the card as 8192 MiB after the run. Current GPU usage after completion cannot reconstruct the earlier physical-memory peak.

No allocator, driver, precision, batch-size, data or training settings were changed, and no additional GPU training was performed to investigate. The other eight models remain stopped. This discrepancy should be clarified before claiming a purely dedicated-VRAM memory fit or proceeding with the remaining runs.

## Independent completion checks

The exported optimizer log exactly reproduces the seed-2026 per-epoch shuffles, with each of the 300 problems seen once in each epoch. All 150 actual optimizer executions succeeded and the learning rate stayed at 1e-4. Both epochs' supervised-token totals sum to 65,758. All exported final-adapter checksums match. CPU inspection found all 112 adapter tensors finite, with rank 4 and q/v projections only. Only the memory-probe and hard/2026 run folders exist; no other primary model has started.
