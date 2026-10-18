# Gold supervised-control dataset — October 13, 2026

Prepared only. No model weights loaded into a GPU; no training or evaluation performed. Same 300 problems and order as the frozen hard set.

| Set | Examples | Supervised tokens / epoch | Two epochs | Planned updates / model |
|---|---:|---:|---:|---:|
| Hard model-generated | 300 | 32,879 | 65,758 | 150 |
| Gold reference | 300 | 32,206 | 64,412 | 150 |

Gold − hard: -673 supervised tokens per epoch (-2.047%). Includes EOS; no token matching or shortening.

Gold solution tokens (excluding EOS): total 31,906, mean 106.35, maximum 282.
Gold full sequence tokens: mean 612.10, minimum 501, maximum 847; unchanged limit 1,024.
Original difficulty buckets preserved: {1: 58, 2: 95, 3: 147}.

This control replaces model-generated explanations with GSM8K references, stripping calculator annotations while preserving wording and answer values. It matches problems and optimizer steps; solution length, style and supervision source differ. It is supervised learning, not self-training.

The dated plan defines gold-minus-base as primary (three-seed mean, same 400 questions, 37-family bootstrap; works only if its 95% interval is entirely above zero). Gold-minus-hard is secondary. Existing outputs and the 63 reserved families remain protected. STOP before training.
