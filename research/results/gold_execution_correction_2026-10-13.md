# Gold execution correction — October 13, 2026

The v1 preflight passed, but the memory probe stopped before loading the model or starting GPU training: CUDA was unavailable because the new entry point imported the CPU preparation module, which sets CUDA_VISIBLE_DEVICES to an empty value. The failed probe manifest, failure record, logs, stopped receipt and v1 scripts are preserved.

The v2 entry point extracts only the original reference-cleaning function and its regexes without executing the preparation module or changing CUDA visibility. Training execution, settings, seeds, prepared gold data and the dated analysis plan are unchanged. A regression test explicitly checks that the reference helper preserves CUDA visibility.

Use new output folders results/gold_control_training_v2 and results/gsm_symbolic_gold_control_v2, with a new analysis output. Repeat the approved probe from a fresh process. The same OOM/non-finite/skipped-update and training/evaluation timeout guards apply (10.24 training minutes, 26 evaluation minutes per adapter). This is an entry-point correction after a CPU-side failure, not a restart following an OOM, NaN, skipped update or timeout. No model trained in v1.
