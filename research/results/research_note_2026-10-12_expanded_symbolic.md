# Research note — October 12, 2026

The GSM-Symbolic expanded-development evaluation completed 1,600 generations: base Qwen3-1.7B-Base and three existing LoRA adapters, each evaluated on 200 P1 and 200 P2 questions from instances 5–14 of the fixed development templates. All 63 reserved original families were preserved.

The prespecified primary pooled three-adapter mean minus base was **+3.42 percentage points**, with a paired original-family bootstrap 95% interval of **−1.26 to +8.00**. The interval contains zero: the result is **inconclusive**, not evidence that the true effect is zero. The positive estimate is compatible with a modest benefit, but benefit is not established. Previously inspected instances 0–4 remain secondary.

String comparison is a genuine potential scoring defect in the batched evaluators. Numeric re-scoring changed **0/3,020 evaluation verdicts** and **0/100 original candidate verdicts**; all 43 accepted training solutions passed numeric verification. Formatting did not explain observed fixes or regressions. The original generator and sequential GSM8K evaluators already used numeric Decimal comparison.

The primary interval half-width was **4.63 points**, describing observed precision rather than a formal minimum detectable effect. Difficulty-targeted versus random training requires its own paired variance/power assessment; the training-versus-base interval does not determine that comparison's sensitivity. More within-family instances can improve precision, but do not add independent families.

Recommended next step, pending pilot and backup decisions: label 200 unused GSM8K training problems starting at shuffled index 336, with four sampled base-model attempts each. This measures runtime and the difficulty distribution; it cannot by itself supply 300–500 unique training problems. Keep the four-attempt labels separate from any later extra sampling. Preserve the reserved families during this pilot and record any later development/test repartition before comparing new training results.

This is outcome-verified self-training research, not demonstrated RSI. No new training has been performed.
