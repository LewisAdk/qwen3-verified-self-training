# Verified Self-Training in a Small Language Model

**Can a small AI model get better at math by studying its own correct answers?** This study tested that with pass/fail rules written down in advance, and found no reliable gain.

**Lewis Adkins** · NYU, B.A. Economics (expected 2030) · [lewisadkins@nyu.edu](mailto:lewisadkins@nyu.edu) · [LinkedIn](https://linkedin.com/in/lewisadkins)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23459643.svg)](https://doi.org/10.5281/zenodo.23459643)

📄 13-page preprint, not peer-reviewed: [PDF](paper/verified-self-training-qwen3-1.7b.pdf) · [DOCX](paper/verified-self-training-qwen3-1.7b.docx)

## In plain English

"Self-training" means letting an AI model answer questions, keeping the answers that turn out to be right, and training the model on them. If it works, a model can improve at little cost, without people writing new training material.

I tried a small, minimal version of this on math word problems, using one consumer graphics card. For each test, I wrote down what would count as success before running it. **None of the four main tests passed.** The largest training run gained 1.58 percentage points, too small to tell apart from chance.

The study also has a built-in check, and that check failed too: training on human-written solutions didn't show an improvement either. So the honest conclusion is limited. This setup showed no reliable gain, but the study can't say whether self-training itself doesn't work or the test was too insensitive to detect it. The paper reports it that way rather than highlighting the few comparisons that looked good.

## At a glance

| | |
|---|---|
| **Model versions trained** | 18: 15 in the main experiments (5 training setups × 3 random seeds), plus a 3-seed pilot |
| **Math problems attempted** | 3,500 (1,100 + 2,400), with four tries each and every answer checked for correctness. 3,258 problems got at least one correct answer. A separate 50-problem pilot came first |
| **Largest training set** | 2,000 verified solutions |
| **Final exam** | 400 questions from a harder test set the models didn't train on |
| **Starting score** | 39.25% correct before any training |
| **Hardware** | One NVIDIA RTX 2080 Super (8 GB) |
| **Released** | Paper, code, dated decision rules and aggregate results |

## The four main tests

| Test | Change in accuracy | 95% range | Result |
|---|---|---:|---|
| Train on 43 of its own correct answers (pilot) | +3.42 points | −1.26 to +8.00 | Inconclusive |
| Train on hard problems instead of random ones | −0.92 points | −5.00 to +2.87 | Inconclusive |
| Train on human-written solutions (the built-in check) | −6.33 points | −14.52 to +2.28 | Inconclusive |
| Train on 2,000 of its own correct answers | +1.58 points | −3.08 to +6.50 | Did not meet the success rule |

A range that includes zero means the result can't be told apart from no change. The first three rows show the family interval, which allows for variation across question types. The last row shows the wider seed-and-family interval, which also allows for variation between training runs. Its success rule required both intervals to be above zero.

For scale: changing only how many questions the original model answered at once moved its score by 1.00 point, with no training at all.

## Summary

Single-round verified self-training with LoRA on **Qwen3-1.7B-Base** (revision `ea980cb0`), using GSM8K training problems and 400 GSM-Symbolic development questions from 37 template families.

- Four sequentially prespecified primary tests: a 43-example pilot, difficulty targeting, human-reference supervision, and training on 2,000 verified examples. **None met its primary criterion.**
- The largest self-training condition gained +1.58 points over base (95% seed-and-family interval [−3.08, +6.50]).
- The human-reference positive control changed accuracy by −6.33 points (family interval [−14.52, +2.28]). Without a control that demonstrated improvement, recipe effectiveness and pipeline sensitivity remain unresolved.
- Changing only the inference batch size of the base model moved accuracy by 1.00 point.

This is a documented **null-result case study**. It reports three-seed results, fixed decision rules, completion diagnostics and scoring checks. The rules were recorded before each evaluation, after earlier development results had been seen; this is not a publicly registered or held-out confirmatory study.

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

The same pinned base model was later fine-tuned for conversation, packaged for in-browser use and used in [PropertyIQ](https://propertyiq.pages.dev) ([source](https://github.com/LewisAdk/propertyiq)), a free apartment-investment analysis tool. [Browser model on Hugging Face](https://huggingface.co/AwesomeLewis64/propertyiq-qwen3-1.7b-research-onnx). That model does not contain this study's adapters, data or results.

## AI assistance

Claude (Anthropic) and Codex (OpenAI) assisted with code, analyses, literature review, figures, tables and manuscript drafting. The author designed the study, chose the experiments and success rules, and is responsible for the results and text.

## Licence

MIT for this repository's code, documents and data produced by the author (see `LICENSE`). Third-party data keeps its own licence (see `DATA_NOTICE.md`).
