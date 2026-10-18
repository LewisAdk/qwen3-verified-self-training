# Verified Self-Training in a Small Language Model

**A prespecified case study on GSM-Symbolic** · Lewis Adkins · 2026 · preprint (not peer-reviewed)

📄 Paper: [PDF](paper/verified-self-training-qwen3-1.7b.pdf) · [DOCX](paper/verified-self-training-qwen3-1.7b.docx)

## Summary

Single-round verified self-training with LoRA on **Qwen3-1.7B-Base** (revision `ea980cb0`), using GSM8K training problems and 400 GSM-Symbolic development questions from 37 template families.

- Four sequentially prespecified primary tests: a 43-example pilot, difficulty targeting, human-reference supervision, and training on 2,000 verified examples. **None met its primary criterion.**
- The largest self-training condition gained +1.58 points over base (95% seed-and-family interval [−3.08, +6.50]).
- The human-reference positive control changed accuracy by −6.33 points (family interval [−14.52, +2.28]). Without a control that demonstrated improvement, recipe effectiveness and pipeline sensitivity remain unresolved.
- Changing only the inference batch size of the base model moved accuracy by 1.00 point.

This is a documented **null-result case study**. It reports three-seed results, fixed decision rules, completion diagnostics and scoring checks.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | The manuscript (PDF and DOCX) |
| `research/*.py` | Labeling, training-set construction, training, evaluation and analysis scripts |
| `research/gsm_symbolic_success_rule_2026-10-12.txt`, `research/results/**/*plan*.md`, `*decisions*.md` | Dated decision rules and plans, quoted in Appendix B |
| `research/results/` | Aggregate analyses, reviewed reports, GSM8K labeling batches, training sets and diagnostics |

Scripts reference the author's local paths (for example `~/rsi-research`). Adjust them to reproduce.

## What is not included, and why

- **GSM-Symbolic question text and per-question outputs** are not redistributed. GSM-Symbolic is licensed CC BY-NC-ND 4.0, which is incompatible with this repository's MIT licence. Download it from [apple/GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) (revision `93b5b3758d9d9841ffe81d6cd2ae2b030685b078`).
- **LoRA adapter weights and checkpoints** are not included (weights are kept out of git).
- Logs, lock files, virtual environments and backups.

## Reproduction settings

Base model `Qwen/Qwen3-1.7B-Base` @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; GSM8K @ `740312add88f781978c0658806c59bc2815b9866`; GSM-Symbolic @ `93b5b3758d9d9841ffe81d6cd2ae2b030685b078`. LoRA rank 4, alpha 8, q_proj/v_proj; AdamW 1e-4 constant; effective batch 4; two epochs; FP16; seeds 2026–2028; one RTX 2080 Super (8 GB). See paper Appendix A.

## Related work by the author

The same pinned base model was later fine-tuned for conversation, packaged for in-browser use and used in [PropertyIQ](https://propertyiq.pages.dev) ([source](https://github.com/LewisAdk/propertyiq)). Browser model: [AwesomeLewis64/propertyiq-qwen3-1.7b-research-onnx](https://huggingface.co/AwesomeLewis64/propertyiq-qwen3-1.7b-research-onnx). That model does not contain this study's adapters, data or results.

## AI assistance

Claude (Anthropic) and Codex (OpenAI) assisted with code, analyses, literature review, figures, tables and manuscript drafting. The author designed the study, chose the experiments and success rules, and is responsible for the results and text.

## Licence

MIT for this repository's code, documents and data produced by the author (see `LICENSE`). Third-party data keeps its own licence (see `DATA_NOTICE.md`).
