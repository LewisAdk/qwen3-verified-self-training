# Pilot answer cut-off audit — October 12, 2026

Read-only analysis of saved pilot records; no GPU inference and no original files changed during this check.

| Shuffled index | Attempt | Correct / 4 | Tokens | Hit 256-token limit? | End reason |
|---:|---:|---:|---:|---|---|
| 352 | 4 | 3/4 | 73 | No | eos |
| 353 | 1 | 2/4 | 256 | Yes | token_limit |
| 353 | 3 | 2/4 | 256 | Yes | token_limit |
| 367 | 1 | 2/4 | 256 | Yes | token_limit |
| 412 | 1 | 0/4 | 256 | Yes | token_limit |
| 412 | 3 | 0/4 | 256 | Yes | token_limit |
| 416 | 3 | 0/4 | 256 | Yes | token_limit |
| 419 | 4 | 1/4 | 256 | Yes | token_limit |
| 482 | 2 | 2/4 | 256 | Yes | token_limit |
| 488 | 1 | 0/4 | 256 | Yes | token_limit |
| 488 | 4 | 0/4 | 256 | Yes | token_limit |
| 490 | 1 | 0/4 | 256 | Yes | token_limit |
| 493 | 1 | 0/4 | 256 | Yes | token_limit |
| 493 | 3 | 0/4 | 256 | Yes | token_limit |
| 512 | 1 | 3/4 | 256 | Yes | token_limit |
| 515 | 3 | 3/4 | 256 | Yes | token_limit |
| 530 | 4 | 2/4 | 193 | No | eos |
| 531 | 2 | 2/4 | 256 | Yes | token_limit |

There were 16 cut-offs: 8 in 0/4, 1 in 1/4, 5 in 2/4, 2 in 3/4 and none in 4/4. Thus 9/16 (56.25%) were in 0/4 or 1/4 problems; all 16 were in the broader unsolved-or-hard 0–3/4 categories. This satisfies the requested majority rule under either reading. The other two unparsed attempts ended at EOS (73 and 193 tokens).

Decision: use 384 new tokens in the full labeling pass, explicitly changed from the 256-token pilot. This is a protocol choice based on truncation diagnostics, not evidence that an extra 128 tokens will necessarily solve these questions. Pilot labels remain separate from selection.
